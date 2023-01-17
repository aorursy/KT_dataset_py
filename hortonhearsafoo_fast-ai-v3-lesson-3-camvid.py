%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.callbacks.hooks import *
path = untar_data(URLs.CAMVID)

path.ls()
path_lbl = path/'labels'

path_img = path/'images'
# path = Path('./data/camvid-small')



# def get_y_fn(x): return Path(str(x.parent)+'annot')/x.name



# codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',

#     'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])



# src = (SegmentationItemList.from_folder(path)

#        .split_by_folder(valid='val')

#        .label_from_func(get_y_fn, classes=codes))



# bs=8

# data = (src.transform(get_transforms(), tfm_y=True)

#         .databunch(bs=bs)

#         .normalize(imagenet_stats))
fnames = get_image_files(path_img)

fnames[:3]
lbl_names = get_image_files(path_lbl)

lbl_names[:3]
img_f = fnames[0]

img = open_image(img_f)

img.show(figsize=(5,5))
get_y_fn = lambda x: path_lbl/f'{x.stem}_P{x.suffix}'
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(5,5), alpha=1)
src_size = np.array(mask.shape[1:])

src_size,mask.data
codes = np.loadtxt(path/'codes.txt', dtype=str); codes
size = src_size//2

bs=8
src = (SegmentationItemList.from_folder(path_img)

       .split_by_fname_file('../valid.txt')

       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), size=size, tfm_y=True)

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

# metrics=accuracy
wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
lr_find(learn)

learn.recorder.plot()
lr=3e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('stage-1')
learn.load('stage-1');
learn.show_results(rows=3, figsize=(8,9))
learn.unfreeze()
lrs = slice(lr/400,lr/4)
learn.fit_one_cycle(12, lrs, pct_start=0.8)
learn.save('stage-2');
size = src_size

bs=3
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)
learn.load('stage-2');
lr_find(learn)

learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
learn.save('stage-1-big')
learn.load('stage-1-big');
learn.unfreeze()
lrs = slice(1e-6,lr/10)
learn.fit_one_cycle(10, lrs)
learn.save('stage-2-big')
learn.load('stage-2-big');
learn.show_results(rows=3, figsize=(10,10))