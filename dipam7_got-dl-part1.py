from fastai import *

from fastai.vision import *
path= "../input/train/"
df = pd.read_csv('../input/train/train.csv')

df.head()
src = ImageList.from_csv(path, 'train.csv', folder='images').split_by_rand_pct(0.2, seed = 2)
tfms = get_transforms(max_rotate=20, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)
def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_df()

               .transform(tfms, size=size, padding_mode=padding_mode)

               .databunch(bs=bs, num_workers=0).normalize(imagenet_stats))
data = get_data(224, 16, 'zeros')
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir='/tmp/models')
learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)
learn.unfreeze()
learn.lr_find()
learn.recorder.plot(suggestion=True)
learn.fit_one_cycle(2, max_lr=slice(2e-5,1e-4))
data = get_data(352,8)

learn.data = data
learn.fit_one_cycle(2, max_lr=slice(2e-5,1e-4))
learn.save('/kaggle/working/352')
data = get_data(352,16)
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True, model_dir='/tmp/models').load('/kaggle/working/352')
idx=0

x,y = data.valid_ds[idx]

x.show()

# data.valid_ds.y[idx]
k = tensor([

    [0.  ,-5/3,1],

    [-5/3,-5/3,1],

    [1.  ,1   ,1],

]).expand(1,3,3,3)/6
k.shape
from fastai.callbacks.hooks import *
k.shape
t = data.valid_ds[0][0].data; t.shape
t[None].shape
edge = F.conv2d(t[None], k)
show_image(edge[0], figsize=(5,5));
m = learn.model.eval();
xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
def hooked_backward(cat=y):

    with hook_output(m[0]) as hook_a: 

        with hook_output(m[0], grad=True) as hook_g:

            preds = m(xb)

            preds[0,int(cat)].backward()

    return hook_a,hook_g
hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()

acts.shape
avg_acts = acts.mean(0)

avg_acts.shape
def show_heatmap(hm):

    _,ax = plt.subplots()

    xb_im.show(ax)

    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),

              interpolation='bilinear', cmap='magma');
show_heatmap(avg_acts)
learn.model
learn.summary()