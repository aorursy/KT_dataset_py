%reload_ext autoreload

%autoreload 2

%matplotlib inline



from fastai.vision import *
bs = 16
path = untar_data(URLs.PETS)/'images'
tfms = get_transforms(max_rotate=20, max_lighting=0.4, max_warp=0.4,

                      p_affine=1., p_lighting=1.)

doc(get_transforms)
src = ImageItemList.from_folder(path).random_split_by_pct(0.2, seed=2)
def get_data(size, bs, padding_mode='reflection'):

    return (src.label_from_re(r'([^/]+)_\d+.jpg$')

           .transform(tfms, size=size, padding_mode=padding_mode)

           .databunch(bs=bs).normalize(imagenet_stats))
data = get_data(224, bs, 'zeros')
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
data = get_data(224,bs)
plot_multi(_plot, 3, 3, figsize=(8,8))
gc.collect()

learn = create_cnn(data, models.resnet34, metrics=error_rate, bn_final=True)
learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3), pct_start=0.8)
data = get_data(224,8) # Error for greater size

learn.data = data
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.save('352')
data = get_data(352,16)
learn = create_cnn(data, models.resnet34, metrics=error_rate, bn_final=True).load('352')
idx=0

x,y = data.valid_ds[idx]

x.show()

data.valid_ds.y[idx]
k = tensor([

    [0.  ,-5/3,1],

    [-5/3,-5/3,1],

    [1.  ,1   ,1],

]).expand(1,3,3,3)/6
from fastai.callbacks.hooks import *
k
k.shape
t = data.valid_ds[0][0].data; t.shape
t[None].shape
edge = F.conv2d(t[None], k)
show_image(edge[0], figsize=(5,5));
data.c
learn.model
learn.summary()
m = learn.model.eval();
xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
from fastai.callbacks.hooks import *
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
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)

grad.shape,grad_chan.shape
mult = (acts*grad_chan[...,None,None]).mean(0)
show_heatmap(mult)
fn = path/'../other/bulldog_maine.jpg'
x = open_image(fn); x
xb,_ = data.one_item(x)

xb_im = Image(data.denorm(xb)[0])

xb = xb.cuda()
hook_a,hook_g = hooked_backward()
acts = hook_a.stored[0].cpu()

grad = hook_g.stored[0][0].cpu()



grad_chan = grad.mean(1).mean(1)

mult = (acts*grad_chan[...,None,None]).mean(0)
show_heatmap(mult)
data.classes[0]
hook_a,hook_g = hooked_backward(0)
acts = hook_a.stored[0].cpu()

grad = hook_g.stored[0][0].cpu()



grad_chan = grad.mean(1).mean(1)

mult = (acts*grad_chan[...,None,None]).mean(0)
show_heatmap(mult)