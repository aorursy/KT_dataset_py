%reload_ext autoreload

%autoreload 2

%matplotlib inline
import fastai

fastai.__version__
from fastai.vision import *

from fastai.callbacks.hooks import *
#path = untar_data(URLs.CAMVID)

#path.ls()
!mkdir -p /tmp/.fastai/data/

!cp -r ../input/pecha /tmp/.fastai/data/



path = Path('/tmp/.fastai/data/pecha/')

path.ls()
path_lbl = path/'labels'

path_img = path/'images'
fnames = get_image_files(path_img)

fnames[:3]
lbl_names = get_image_files(path_lbl)

lbl_names[:3], len(lbl_names)
img_f = fnames[5]

img = open_image(img_f)

img.show(figsize=(10,10)), img.size
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(10,10), alpha=1)
src_size = np.array(mask.shape[1:])

src_size,mask.data
codes = np.loadtxt(path/'codes.txt', dtype=str); codes, len(codes)
size = src_size//2

bs=4

size, src_size
src = (SegmentationItemList.from_folder(path_img)

       .split_by_fname_file('../valid.txt')

       .label_from_func(get_y_fn, classes=codes))
tfms = get_transforms(do_flip=False, flip_vert=False)

data = (src.transform(tfms, size=size, tfm_y=True)

        .databunch(bs=bs, num_workers=0)

        .normalize(imagenet_stats))
data.show_batch(2, figsize=(10,7))
data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)
name2id = {v:k for k,v in enumerate(codes)}

void_code = name2id['Void']



def acc_camvid(input, target):

    target = target.squeeze(1)

    mask = target != void_code

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics=acc_camvid
wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
x, y = next(iter(learn.data.train_dl))

x.shape, y.shape
y.min(), y.max()
import matplotlib.pyplot as plt



img = x.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')[0]

img = img.reshape(250, 410, 3)

mask = y.permute(0, 2, 3, 1).cpu().numpy().astype('uint8')[0]

mask = mask.reshape(250, 410)



plt.imshow(img)

plt.show()

plt.imshow(mask, cmap='gray')

plt.show()
out = learn.model(x.data); out.shape
!export CUDA_LAUNCH_BLOCKING=1
# %%debug

learn.loss_func(input=out, target=y)
lr_find(learn)

learn.recorder.plot()
# lr = slice(1e-06,1e-03)
learn.fit_one_cycle(10, slice(1e-06,1e-03), pct_start=0.9)
learn.recorder.plot_losses()
path = "."
learn.save('stage-1')
learn.show_results(rows=3, figsize=(40,50))
learn.unfreeze()
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(12, slice(1e-06,1e-05), pct_start=0.8)
learn.save('stage-2')
learn.show_results(rows=3, figsize=(40,50))