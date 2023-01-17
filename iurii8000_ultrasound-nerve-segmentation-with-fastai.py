!ls

%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *
import pandas as pd

import glob
path_images = Path("../input/train")

path_lbl = path_images
path_test = Path("../input/test")
fn_list = glob.glob('../input/train/*[!_mask].tif') #создает список с именами img(x)

print(fn_list[:3])

print(fn_list[3])

len(fn_list)
lbl_names = glob.glob('../input/train/*_mask.tif') #создает список с именами mask(y)

print(lbl_names[:3])

len(lbl_names)
def get_y_fn(x):  # выдает y(имя маски) по x

    x = Path(x)

    return path_lbl/f'{x.stem}_mask{x.suffix}'
img_f = fn_list[3]

img = open_image(img_f)

img.show(figsize=(5,5))

print(img.size)



mask = open_mask(get_y_fn(img_f),div=True)

mask.show(figsize = (5,5))

print(mask.size)
def mask2rle(img):

    img = img.resize(420,580)

    print(img.size())

    tmp_1 = np.flipud(img)

    print(tmp_1.shape)

    tmp = np.rot90(tmp_1, k=3)

    print(tmp.shape)

    rle = []

    lastColor = 0;

    startpos = 0

    endpos = 0



    tmp = tmp.reshape(-1,1)   

    for i in range( len(tmp) ):

        if (lastColor==0) and tmp[i]>0:

            startpos = i

            lastColor = 1

        elif (lastColor==1)and(tmp[i]==0):

            endpos = i-1

            lastColor = 0

            rle.append( str(startpos)+' '+str(endpos-startpos+1) )

    return rle
src_size = np.array(mask.shape[1:])

src_size,mask.data
filter_func = lambda x: str(x) in fn_list
from fastai.utils.mem import *

size = 128

bs=16
class SegLabelListCustom(SegmentationLabelList):

    def open(self, fn): return open_mask(fn, div=True)  # для каждого изображения x возврвщае маску(y)

class SegItemListCustom(SegmentationItemList):

    _label_cls = SegLabelListCustom # метка класса для этого изображения и есть маска

codes = ['0','1']

src = (SegItemListCustom.from_folder(path_images)  # path_images = Path("../input/train")

       .filter_by_func(filter_func) # Сохраняет только те элементы, для которых func возвращает True.filter_func = lambda x: str(x) in fnames х 

       .split_by_rand_pct()  # Разделите элементы случайным образом, поместив valid_pct в набор проверки

       .label_from_func(get_y_fn,classes=codes))
type(src)
data = (src.transform(get_transforms(), size=size, tfm_y=True)

        .databunch(bs=bs)

        .normalize(imagenet_stats))

data.path = Path('.')
data.show_batch(rows=4, figsize=(14,10))
def dice_func(input, target):

    smooth = 0

    input = input[:,1,:,:]

    iflat = input.flatten().float()

    tflat = target.flatten().float()

    intersection = (iflat * tflat).sum()

    return ((2. * intersection + smooth) /

              (iflat.sum() + tflat.sum() + smooth))



def dice(input:Tensor, targs:Tensor, iou:bool=False)->Rank0Tensor:

    n = targs.shape[0]

    input = input.argmax(dim=1).view(n,-1)

    targs = targs.view(n,-1)

    intersect = (input * targs).sum().float()

    union = (input+targs).sum().float()

    if not iou: return (2. * intersect / union if union > 0 else union.new([1.]).squeeze())

    else: return intersect / (union-intersect+1.0)

learn = unet_learner(data, models.resnet152, metrics=[dice], wd=1e-3)
lr_find(learn)

learn.recorder.plot()
learn.fit_one_cycle(5,max_lr = 1e-5) # 20
learn.recorder.plot_lr(show_moms=True)
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
learn.unfreeze()
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(15,max_lr = slice(6e-6,1e-4)) # 60
learn.recorder.plot_losses()
learn.recorder.plot_metrics()
# learn.save("trained_model")
df_1 = pd.read_csv("../input/sample_submission.csv")
def pred_rle(fn):

    a = learn.predict(open_image(fn))

    a_1 = a[0].data

    a_2 = open_mask_rle(rle_encode(a_1), (128, 128)).resize((1,420,580))

    a_3 = a_2.data

    a_3 = a_3.resize(420,580)

    print(a_3.size())

    a_3 = np.flipud(a_3)

    print(a_3.shape)

    a_3 = np.rot90(a_3, k=3)

    return rle_encode(a_3)
for i in df_1['img']:

    df_1['pixels'][df_1['img'] == i] = pred_rle('../input/test/' + str(i) + '.tif')   
df_1.to_csv('submission.csv', index=False)