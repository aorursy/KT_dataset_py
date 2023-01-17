%reload_ext autoreload

%autoreload 2

%matplotlib inline
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
from fastai.vision import *

from fastai.callbacks.hooks import *

from fastai.utils.mem import *
path = Path('../input/obi-mouse-brain')

path.ls()
path_lbl = path/'masks'

path_img = path/'auto tone'
fnames = get_image_files(path_img)

fnames[:3]
lbl_names = get_image_files(path_lbl)

lbl_names[:3]
img_f = fnames[0]

img = open_image(img_f)

img.show(figsize=(5,5))
get_y_fn = lambda x: path_lbl/f'{x.stem}{x.suffix}'
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(5,5), alpha=1)
src_size = np.array(mask.shape[1:])

src_size,mask.data
codes = np.loadtxt(path/'codes.txt', dtype=str); codes
size = src_size//8



free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

if free > 8200: bs=8

else:           bs=4

print(f"using bs={bs}, have {free}MB of GPU RAM free")
class SegLabelListCustom(SegmentationLabelList):

    def open(self, fn): return open_mask(fn, div=True)



class SegItemListCustom(SegmentationItemList):

    _label_cls = SegLabelListCustom



codes = ['0','1']

src = (SegItemListCustom.from_folder(path_img)

       .random_split_by_pct(valid_pct=0.2, seed=33)

       .label_from_func(get_y_fn, classes=codes))



data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
# src = (SegmentationItemList.from_folder(path_img)

#        .split_by_rand_pct(valid_pct = .2, seed = 42)

# #        .label_empty()) #TODO ???

#        .label_from_func(get_y_fn, classes=codes))
# data = (src.transform(get_transforms(), size=size, tfm_y=True)

#         .databunch(bs=bs)

#         .normalize(imagenet_stats))
data.show_batch(2, figsize=(10,7))

data.show_batch(2, figsize=(10,7), ds_type=DatasetType.Valid)
metrics=accuracy_thresh

wd=1e-2
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir = '/kaggle/working')
lr_find(learn)

learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.9)
learn.save('/kaggle/working/one_cycle')
learn.show_results(rows=3, figsize=(8,9))
img = data.valid_ds[0][0]

img
from PIL import Image

# img.show(y=learn.predict(img)[0])

prediction = learn.predict(img)

think_np = np.array(prediction[1])

think_np.shape = (256,256)

think_np = think_np.astype(int)

think_np[think_np > 0] = 255

think_im = Image.fromarray((think_np).astype('uint8'), mode='L')

think_im
# learn.destroy()



size = src_size//2



free = gpu_mem_get_free_no_cache()

# the max size of bs depends on the available GPU RAM

if free > 8200: bs=1

else:           bs=1

print(f"using bs={bs}, have {free}MB of GPU RAM free")
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, model_dir = '/kaggle/working')
# learn.load('/kaggle/working/one_cycle');
lr_find(learn)

learn.recorder.plot()
lr=1e-3
learn.fit_one_cycle(10, slice(lr), pct_start=0.8)
learn.save('one-cycle-big')

learn.save('one-cycle-big.pkl')
learn.show_results(rows=3, figsize=(20,10))
img = data.valid_ds[0][0]

img
prediction = learn.predict(img)

think_np = np.array(prediction[1])

think_np.shape = (1024,1024)

think_np = think_np.astype(int)

think_np[think_np > 0] = 255

think_im = Image.fromarray((think_np).astype('uint8'), mode='L')

think_im
img.show(y=learn.predict(img)[0], figsize=(20,10))