%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd

import matplotlib.pyplot as plt
# Making pretrained weights work without needing to find the default filename

if not os.path.exists('/tmp/.cache/torch/checkpoints/'):

        os.makedirs('/tmp/.cache/torch/checkpoints/')

!cp '../input/resnet50/resnet50.pth' '/tmp/.cache/torch/checkpoints/resnet50-19c8e357.pth'
import os

os.listdir('../input')
print('Make sure cudnn is enabled:', torch.backends.cudnn.enabled)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True



SEED = 999

seed_everything(SEED)
base_image_dir = os.path.join('..', 'input/aptos2019-blindness-detection/')

train_dir = os.path.join(base_image_dir,'train_images/')

df = pd.read_csv(os.path.join(base_image_dir, 'train.csv'))

df['path'] = df['id_code'].map(lambda x: os.path.join(train_dir,'{}.png'.format(x)))

df = df.drop(columns=['id_code'])

df = df.sample(frac=1).reset_index(drop=True) #shuffle dataframe

df.head(10)
len_df = len(df)

print(f"There are {len_df} images")
df['diagnosis'].hist(figsize = (10, 5))
bs = 64 #smaller batch size is better for training, but may take longer

sz=224
tfms = get_transforms(do_flip=True,flip_vert=True,max_rotate=360,max_warp=0,max_zoom=1.1,max_lighting=0.1,p_lighting=0.5)

src = (ImageList.from_df(df=df,path='./',cols='path') #get dataset from dataset

        .split_by_rand_pct(0.2) #Splitting the dataset

        .label_from_df(cols='diagnosis') #obtain labels from the level column

      )

data= (src.transform(tfms,size=sz,resize_method=ResizeMethod.SQUISH,padding_mode='zeros') #Data augmentation

        .databunch(bs=bs,num_workers=4) #DataBunch

        .normalize(imagenet_stats) #Normalize     

       )
data.show_batch(rows=3, figsize=(7,6))
from sklearn.metrics import cohen_kappa_score

def quadratic_kappa(y_hat, y):

    return torch.tensor(cohen_kappa_score(y_hat.argmax(dim=-1), y, weights='quadratic'),device='cuda:0')
learn = cnn_learner(data, base_arch=models.resnet50, metrics = [quadratic_kappa])
learn.fit_one_cycle(4,max_lr = 1e-2)
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
learn.unfreeze()

learn.fit_one_cycle(6, max_lr=slice(1e-6,1e-3))
learn.recorder.plot_losses()

learn.recorder.plot_metrics()
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
idx = 1

im,cl = learn.data.dl(DatasetType.Valid).dataset[idx]

cl = int(cl)

im.show(title=f"pred. class: {interp.pred_class[idx]}, actual class: {learn.data.classes[cl]}")
xb,_ = data.one_item(im) #put into a minibatch of batch size = 1

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
m = learn.model.eval()
type(m)
len(m)
from fastai.callbacks.hooks import *
def hooked_backward(cat=cl):

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(cat)].backward()

    return hook_a,hook_g

hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu() #activation maps

acts.shape
grad = hook_g.stored[0][0].cpu() #gradients

grad.shape
grad_chan = grad.mean(1).mean(1) # importance weights

grad_chan.shape
mult = F.relu(((acts*grad_chan[...,None,None])).sum(0)) # GradCAM map

mult.shape
#Utility function to display heatmap:

def show_heatmap(hm):

    _,ax = plt.subplots()

    sz = list(xb_im.shape[-2:])

    xb_im.show(ax,title=f"pred. class: {interp.pred_class[idx]}, actual class: {learn.data.classes[cl]}")

    ax.imshow(hm, alpha=0.6, extent=(0,*sz[::-1],0),

              interpolation='bilinear', cmap='magma')

    return _,ax
show_heatmap(mult)
def GradCAM(idx:int,interp:ClassificationInterpretation, image = True):

    m = interp.learn.model.eval()

    im,cl = interp.learn.data.dl(DatasetType.Valid).dataset[idx]

    cl = int(cl)

    xb,_ = interp.data.one_item(im) #put into a minibatch of batch size = 1

    xb_im = Image(interp.data.denorm(xb)[0])

    xb = xb.cuda()

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(cl)].backward() 

    acts  = hook_a.stored[0].cpu() #activation maps

    grad = hook_g.stored[0][0].cpu()

    grad_chan = grad.mean(1).mean(1)

    mult = ((acts*grad_chan[...,None,None])).sum(0) #F.relu(((acts*grad_chan[...,None,None])).sum(0))

    if image:

        _,ax = plt.subplots()

        sz = list(xb_im.shape[-2:])

        xb_im.show(ax,title=f"pred. class: {interp.pred_class[idx]}, actual class: {learn.data.classes[cl]}")

        ax.imshow(mult, alpha=0.4, extent=(0,*sz[::-1],0),

              interpolation='bilinear', cmap='magma')

    return mult
_ = GradCAM(np.random.randint(len(learn.data.valid_ds)),interp)
return_fig = interp.plot_top_losses(6,heatmap=True,return_fig = True)