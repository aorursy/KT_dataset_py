# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as animation



from fastai import *

from fastai.vision import *

from fastai.callbacks import *



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


df_train = pd.read_csv(dirname+'/train.csv')

df_train['fn'] = df_train.index

df_train.head()
df_train['label'].hist()
def valid_split(df, valid_ratio = 0.1):

    valid_idx = []

    for i in df['label'].unique():

        valid_idx+=list(df[df['label']==i].sample(frac=valid_ratio, random_state=2020)['fn'].values)

    return valid_idx



valid_idx = valid_split(df_train, valid_ratio = 0.2)
class PixelImageItemList(ImageList):

    

    def __init__(self, myimages = {}, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.myimages = myimages 

    

    def open(self,fn):

        return self.myimages.get(fn)

    

    @classmethod

    def from_df(cls, df:DataFrame, path:PathOrStr, cols:IntsOrStrs=0, folder:PathOrStr=None, suffix:str='', **kwargs)->'ItemList':

        "Get the filenames in `cols` of `df` with `folder` in front of them, `suffix` at the end."

        res = super().from_df(df, path=path, cols=cols, **kwargs)

        

        # FULL LOAD of all images

        for i, row in df.drop(labels=['label','fn'],axis=1).iterrows():

            # Numpy to Image conversion from

            # https://www.kaggle.com/heye0507/fastai-1-0-with-customized-itemlist

            img_pixel = row.values.reshape(28,28)

            img_pixel = np.stack((img_pixel,)*1,axis=-1)

            ## mark for convolution test

            img_pixel[1,1]=255

            

            res.myimages[res.items[i]]=vision.Image(pil2tensor(img_pixel,np.float32).div_(255))



        return res
%%time

data = (PixelImageItemList.from_df(df=df_train,path='.',cols='fn')

        .split_by_idx(valid_idx=valid_idx) #.split_by_rand_pct()

        .label_from_df(cols='label')

        .databunch(bs=128))

data.dataset[0]
data.show_batch(rows=3, figsize=(5,5), cmap='gray')
#https://github.com/fastai/fastai/blob/master/fastai/layers.py#L111

def myconv_layer(ni:int, nf:int, ks:int=3, stride:int=1):

    layers = [init_default(nn.Conv2d(ni,nf,stride=stride,kernel_size=ks,padding=1, bias=False),nn.init.kaiming_normal_)]

    

    # make sure the ReLU doesn't override the conv2d activations, so we keep more details for later

    layers.append(nn.ReLU(inplace=False))

    layers.append(nn.BatchNorm2d(nf))

    

    return nn.Sequential(*layers)
# Adapted from https://www.kaggle.com/melissarajaram/fastai-pytorch-with-best-original-mnist-arch



leak = 0.15

model = nn.Sequential(

    

    myconv_layer(1,32),

    conv_layer(32,32,stride=1,ks=3,leaky=leak),

    conv_layer(32,32,stride=2,ks=5,leaky=leak),

    nn.Dropout(0.4),

    

    conv_layer(32,64,stride=1,ks=3,leaky=leak),

    conv_layer(64,64,stride=1,ks=3,leaky=leak),

    conv_layer(64,64,stride=2,ks=5,leaky=leak),

    nn.Dropout(0.4),

    

    Flatten(),

    nn.Linear(3136, 128),

    relu(inplace=True),

    nn.BatchNorm1d(128),

    nn.Dropout(0.4),

    nn.Linear(128,10)

)
learn = Learner(data, model, loss_func = nn.CrossEntropyLoss() , metrics=[accuracy])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(3, 0.5e-3)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(12, figsize=(7,6))

interp.plot_confusion_matrix()
conv_kernels=list(learn.model.parameters())[0] # alternativly: learn.model[0][0].weight.data

# convert to numpy

conv_kernels=conv_kernels.cpu().detach().numpy()

# example:

print('Shape kernels first con layer:', conv_kernels.shape)

print('First kernel:\n', conv_kernels[0,0,:,:])
for i in range(conv_kernels.shape[0]):

    ax = plt.subplot(conv_kernels.shape[0]/8, 8, i+1)

    ax.set_xticks([])

    ax.set_yticks([])

    conv_kernel = conv_kernels[i,0,:,:]

    plt.imshow(conv_kernels[i,0,:,:], cmap='gray')

        

plt.show()
learn.model[0:2]
example_img = 0



#

m = learn.model.eval()

x,y = data.train_ds[example_img]

xb,_ = data.one_item(x)

xb = xb.cuda()



data.train_ds.get(example_img)
mblock = 0 # first Sequential

inv_layer = m[mblock][0] # conv2d layer in first block



def hooked():

    with hook_output(inv_layer) as hook_forward:

        preds = m(xb)

    return hook_forward



acts = hooked().stored[0].cpu()



inv_layer
acts[0:5][0,:5,:5]
conv_kernels[0,0,:,:]


def img_activations(m:nn.Module, img_id:Image, data = data, ds=data.train_ds):

    # create batch with one image

    xb,_ = data.one_item(x)

    xb = xb.cuda()

    

    # flatten to get activations of children

    with hook_outputs(flatten_model(m)) as hook_forward:

        preds=m.eval()(xb)

    

    return [i.cpu() for i in hook_forward.stored[:]]



m = learn.model.eval()

x,_ = data.train_ds[0]

acts = img_activations(m, x,data)

    
# apply ReLU to first layer and compare to second layer

torch.max(torch.zeros(3,3),acts[0][0,0,:3,:3]) == acts[1][0,0,:3,:3]
import warnings

warnings.filterwarnings('ignore')



def scale_color(im,mn=None,mx=None):

    if mn == None:

        mn = im.min()

    if mx == None:

        mx = im.max()

    return (im-mn)/(mx-mn)



fig = plt.figure(figsize=(12, 6))

ims=[]

for i in range(32):#acts[0]):

    

    krnl = scale_color(np.copy(conv_kernels[i,0,:,:]))

    org_im = x.data.numpy()[0,:,:] 

    layer_1_im = acts[0][0,i,:,:].numpy()

    layer_2_im = acts[1][0,i,:,:].numpy()

    

    org_im[:3,:3] = krnl

    layer_1_im = scale_color(layer_1_im) 

    layer_1_im[:3,:3] = krnl #[:3,:3] 

    layer_2_im = scale_color(layer_2_im) 

    layer_2_im[:3,:3] = krnl #[:3,:3] 

    

    #layer_1_im=np.concatenate((org_im,layer_1_im,layer_2_im), axis=1)

    

    ax1 = plt.subplot(131, frameon=False)

    ax1.set_title('(0) Input')

    ax1.set_xticks([])

    ax1.set_yticks([])

    im1 = plt.imshow(org_im,animated=True, cmap='gray')

    

    ax2 = plt.subplot(132, frameon=False)

    ax2.set_title('(1) Conv2d(1,32)')

    ax2.set_xticks([])

    ax2.set_yticks([])

    im2 = plt.imshow(layer_1_im,animated=True, cmap='gray')

    

    ax3 = plt.subplot(133, frameon=False)

    ax3.set_title('(2) ReLU(32)')

    ax3.set_xticks([])

    ax3.set_yticks([])

    im3 = plt.imshow(layer_2_im,animated=True, cmap='gray')

      

    ims.append([im1,im2, im3]) #im1,



ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False, repeat_delay=1000)



from IPython.display import HTML

HTML(ani.to_jshtml())  


