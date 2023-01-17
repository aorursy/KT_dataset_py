import os

GPU_id = 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_id)
import warnings

warnings.filterwarnings("ignore")



import time

import os



from fastai.vision import *

from fastai.callbacks.hooks import *
!pwd
path = Path('../input/pennfudanped')

path.ls()
path_lbl = path/'PedMasks'

path_img = path/'PNGImages'
fnames = get_image_files(path_img)

fnames[:3]
lbl_names = get_image_files(path_lbl)

lbl_names[:3]
img_f = fnames[0]

img = open_image(img_f)

img.show(figsize=(5, 5))
print(img_f.stem,img_f.suffix)

get_y_fn = lambda x: path_lbl/f'{x.stem}_mask{x.suffix}'

get_y_fn(img_f)
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(5, 5), alpha=1)

src_size = np.array(mask.shape[1:])
df = pd.read_csv(path/'added-object-list.txt',skiprows=1,sep='\t')

df.columns = ['image','objects']

df.head()
df['objects'].max()
mask = df['objects']==8

df.loc[mask]
img_f = Path(path/'PNGImages/FudanPed00058.png')

img = open_image(img_f)

img.show(figsize=(5, 5))
mask = open_mask(get_y_fn(img_f))

mask.show(figsize=(5, 5), alpha=1)
codes = np.array(['background','person'])
size = src_size//2

bs = 32
class MySegmentationLabelList(SegmentationLabelList):

    def open(self, fn): 

        res = open_mask(fn)

        res.px = (res.px>0).float()

        return res



class MySegmentationItemList(ImageList):

    "`ItemList` suitable for segmentation tasks."

    _label_cls,_square_show_res = MySegmentationLabelList,False
src = (MySegmentationItemList.from_folder(path_img) # SegmentationItemList

       .split_by_rand_pct(0.2) # SegmentationItemList

       .label_from_func(get_y_fn, classes=codes)) # LabelLists
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))
data.show_batch(2, figsize=(5, 5))
name2id = {v:k for k,v in enumerate(codes)}

void_code = -1



def acc_camvid(input, target):

    target = target.squeeze(1)

    mask = target != void_code

    #print(input.size())

    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()
wd = 1e-2

metrics = acc_camvid

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd)

learn.model_dir = '/kaggle/working/models'
learn.lr_find()
learn.recorder.plot()
lr = 1e-4

learn.fit_one_cycle(10, slice(lr))
%%time

pred,truths = learn.get_preds()
class MyImageList(ImageList):

    def __init__(self, *args, imgs=None, **kwargs):

        self.imgs = imgs

        

    def get(self, i):

        res = self.imgs[i]

        return Image(res)
pred_masks = MyImageList(imgs = pred.argmax(dim=1,keepdim=True))

true_masks = MyImageList(imgs = truths)
def _plot(i,j,ax): true_masks[i*3+j].show(ax)

plot_multi(_plot, 3, 3, figsize=(8,8))
def _plot(i,j,ax): pred_masks[i*3+j].show(ax)

plot_multi(_plot, 3, 3, figsize=(8,8))