%matplotlib inline
%reload_ext autoreload
%autoreload 2

import json
import pydicom
from pathlib import Path
from PIL import ImageDraw, ImageFont
from matplotlib import patches, patheffects
from fastai.conv_learner import *
from fastai.dataset import *

torch.cuda.set_device(0)
PATH = Path("../input")
list(PATH.iterdir())
train_bb_df = pd.read_csv(PATH/'stage_1_train_labels.csv')
# train_bb_df.head()
train_bb_df['duplicate'] = train_bb_df.duplicated(['patientId'], keep=False)
# train_bb_df[train_bb_df['duplicate']].head()
detailed_df = pd.read_csv(PATH/'stage_1_detailed_class_info.csv')
# merge two df
class_df = train_bb_df.merge(detailed_df, on="patientId")
csv_df = class_df.filter(['patientId', 'class'], )
# csv_df = csv_df.set_index('patientId', )
# detailed_df.head() , 
class_df.head()
# csv_df.head()
DCMS = 'stage_1_train_images'
IMG_PATH = PATH/DCMS
img_size = 1024
all_images = list(IMG_PATH.iterdir())
all_images[:5]
def open_image(loc):
    if isinstance(loc, str):
        loc = loc + '.dcm'
    else: # posix path
        loc = loc.as_posix()
    img_arr = pydicom.read_file(loc).pixel_array
    img_arr = img_arr/img_arr.max()
    img_arr = (255*img_arr).clip(0, 255)#.astype(np.int32)
    img_arr = Image.fromarray(img_arr).convert('RGB') # model expects 3 channel image
    return np.array(img_arr)
from fastai import dataset
dataset.open_image = open_image
im0 = all_images[0]
im = open_image(im0)
def show_img(im, figsize=None, ax=None):
    if isinstance(im, Path): # read image from loc
        im = open_image(im)
    if not ax: 
        fig,ax = plt.subplots(figsize=figsize)
    ax.imshow(im)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return ax

def draw_outline(o, lw):
  o.set_path_effects([patheffects.Stroke(
      linewidth=lw, foreground='black'), patheffects.Normal()])

def draw_rect(ax, b):
    patch = ax.add_patch(patches.Rectangle(b[:2], *b[-2:], fill=False, edgecolor='white', lw=2))
    draw_outline(patch, 4)

def draw_text(ax, xy, txt, sz=14, color='white'):
    text = ax.text(*xy, txt, verticalalignment='top', color=color, fontsize=sz, weight='bold')
    draw_outline(text, 1)
ax = show_img(im)
def resized (patients, resize):
    if not resize:
        return (patients.x, patients.y, patients.width, patients.height), patients.Target
    else:
        scale = img_size/resize
        return (patients.x/scale, patients.y/scale, patients.width/scale, patients.height/scale), patients.Target

def get_bb_category(im_loc, only_one=True, resize=None):
    patientId = im_loc.name.split('.dcm')[0]
    patients = class_df[class_df['patientId'] == patientId]

    patients.iloc[np.argsort(patients.width * patients.height).values]
    if only_one:
        patients = patients.iloc[-1:].iloc[0]
        return resized(patients, resize)
    else:
        return [resized(patients.iloc[i], resize )for i in range(len(patients))]
            

im_loc = IMG_PATH/'00436515-870c-4b36-a041-de91049b9ab4.dcm'   
get_bb_category(im_loc,only_one=False, resize=224)
ax = show_img(im_loc, figsize=(16,8))
draw_text(ax, (0,0), im_loc.name.split('.dcm')[0], color='green')
for b,c in get_bb_category(im_loc,only_one=False):
    draw_rect(ax, b)
    draw_text(ax, b[:2], c)
CSV = '/tmp/lrg.csv'
csv_df.sample(100, random_state=5500).to_csv(CSV, index=False)
f_model = resnet34
resize=sz=224
bs=64
tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, crop_type=CropType.NO)
md = ImageClassifierData.from_csv(PATH, DCMS, CSV, tfms=tfms, bs=bs)
x,y=next(iter(md.val_dl))
# does fastai internally uses denorm? 
show_img(md.val_ds.denorm(to_np(x))[0]);
# https://github.com/fastai/fastai/blob/921777feb46f215ed2b5f5dcfcf3e6edd299ea92/fastai/conv_learner.py
from fastai.learner import Learner
class CustomLearner(Learner):
    def __init__(self, data, models, opt_fn=None, tmp_name='/tmp', models_name='models', metrics=None, clip=None, crit=None):
        self.data_,self.models,self.metrics = data,models,metrics
        self.sched=None
        self.wd_sched = None
        self.clip = None
        self.opt_fn = opt_fn or SGD_Momentum(0.9)
        self.tmp_path = '/tmp' 
        self.models_path = '/tmp/models' 
        os.makedirs(self.tmp_path, exist_ok=True)
        os.makedirs(self.models_path, exist_ok=True)
        self.crit = crit if crit else self._get_crit(data)
        self.reg_fn = None
        self.fp16 = False
        
    def _get_crit(self, data): return F.mse_loss
    
from fastai.conv_learner import ConvLearner
class CustomConvLearner(CustomLearner, ConvLearner):
    def __init__(self, data, models, precompute=False, **kwargs):
        self.precompute = False
        super().__init__(data, models, **kwargs)
        if hasattr(data, 'is_multi') and not data.is_reg and self.metrics is None:
            self.metrics = [accuracy_thresh(0.5)] if self.data.is_multi else [accuracy]
        if precompute: self.save_fc1()
        self.freeze()
        self.precompute = precompute

learn = CustomConvLearner.pretrained(f_model, md, metrics=[accuracy])
learn.opt_fn = optim.Adam
# def accuracy(preds, targs):
#     preds,targs=preds.type(torch.LongTensor),targs.type(torch.LongTensor)
# #     preds = torch.max(preds, dim=1)[1]
#     return (preds==targs).float().mean()

# from fastai import metrics
# metrics.accuracy = accuracy
lrf=learn.lr_find(1e-5,100)
learn.sched.plot()
