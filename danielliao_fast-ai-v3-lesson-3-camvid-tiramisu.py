%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from fastai.callbacks.hooks import *
path = Path('/kaggle/input/repository/alexgkendall-SegNet-Tutorial-bb68b64/CamVid')
path.ls()
fnames = get_image_files(path/'val')

fnames[:3]
lbl_names = get_image_files(path/'valannot')

lbl_names[:3]
img_f = fnames[0]

img = open_image(img_f)

img.show(figsize=(5,5))
def get_y_fn(x): return Path(str(x.parent)+'annot')/x.name



codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',

    'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(5,5), alpha=1)
src_size = np.array(mask.shape[1:])

src_size,mask.data
bs,size = 8,src_size//2
src = (SegmentationItemList.from_folder(path)

       .split_by_folder(valid='val')

       .label_from_func(get_y_fn, classes=codes))
data = (src.transform(get_transforms(), tfm_y=True)

        .databunch(bs=bs, num_workers=0)

        .normalize(imagenet_stats))
data.show_batch(2, figsize=(10,7))
name2id = {v:k for k,v in enumerate(codes)}

void_code = name2id['Void']



def acc_camvid(input, target):

    target = target.squeeze(1)

    mask = target != void_code

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
metrics=acc_camvid

wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir='/kaggle/working/models')
lr_find(learn)

learn.recorder.plot()
lr=2e-3
learn.fit_one_cycle(1, slice(lr), pct_start=0.8) # 10
learn.save('stage-1')
del learn

gc.collect()

# learn.destroy() # 释放内容， 但是1.0.45版本里没有这个method

# https://www.kaggle.com/danielliao/fastai-tutorial-13-dl-on-a-shoestring



learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir='/kaggle/working/models')

learn.load('stage-1',with_opt=False);
learn.unfreeze()
lrs = slice(lr/100,lr)
learn.fit_one_cycle(1, lrs, pct_start=0.8) # 12
learn.save('stage-2');
del learn

gc.collect()

# learn.destroy() # 释放内容， 但是1.0.45版本里没有这个method

# https://www.kaggle.com/danielliao/fastai-tutorial-13-dl-on-a-shoestring
size = src_size

bs=4
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs, num_workers=0) # 设置为0很关键

        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir="/kaggle/working/models").load('stage-2');
lr_find(learn)

learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(1, slice(lr), pct_start=0.8)  #10
learn.save('stage-1-big') # 这里是GPU disk 增加的原因, 

# at Console: go to /kaggle/working/models/ rm stage-1.pth 可以释放GPU disk 
del learn

gc.collect()

# learn.destroy() # 释放内容， 但是1.0.45版本里没有这个method

# https://www.kaggle.com/danielliao/fastai-tutorial-13-dl-on-a-shoestring



learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir='/kaggle/working/models')

learn.load('stage-1-big',with_opt=False);

# learn.load('stage-1-big');
learn.unfreeze()
lrs = slice(lr/1000,lr/10)
learn.fit_one_cycle(1, lrs) # 10
learn.save('stage-2-big')
del learn

gc.collect()

# learn.destroy() # 释放内容， 但是1.0.45版本里没有这个method

# https://www.kaggle.com/danielliao/fastai-tutorial-13-dl-on-a-shoestring



learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir='/kaggle/working/models')

learn.load('stage-2-big',with_opt=False);

# learn.load('stage-2-big');
learn.show_results(rows=3, figsize=(9,11))
# start: 480x360
learn.summary()