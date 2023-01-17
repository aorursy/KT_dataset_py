! pip install fastai==1.0.61
# import numpy as np

# np.random.seed(12)
# path = '/content/drive/My Drive/F1years/data'
# folder = '2018'
# in_file = '/content/b3RhyVvn.csv'
# from fastai.vision.utils import Path

# path_to_save = Path(path)
# dest = path_to_save/folder
# dest.mkdir(parents=True, exist_ok=True)
# from fastai.vision.utils import download_images

# download_images(in_file, dest, max_pics=15, max_workers=0)
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")
import numpy as np

np.random.seed(12)
from fastai.vision import ImageDataBunch
from fastai.vision import imagenet_stats, get_transforms

path = '../input/f1-cars-by-years/f1_years_data'
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(max_warp=None), size=224, num_workers=4, bs=32)\
        .normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,8))
from fastai.vision import cnn_learner, error_rate
from fastai.vision.models import resnet34

learn = cnn_learner(data, resnet34, pretrained=True, metrics=error_rate)
learn.fit_one_cycle(7)
# unaviable at Kaggle
# learn.save('stage1_v0_2')
from fastai.vision import ClassificationInterpretation

interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
interp.plot_top_losses(3)
idx=11
x,y = data.valid_ds[idx]
x.show()
# data.valid_ds.y[idx]
x.shape
from fastai.vision import Image
import torch

m = learn.model.eval()
xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
if torch.cuda.is_available():
    xb = xb.cuda()
from fastai.callbacks.hooks import hook_output

def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g

hook_a,hook_g = hooked_backward()
acts  = hook_a.stored[0].cpu()
avg_acts = acts.mean(0)
acts.shape, avg_acts.shape
from matplotlib import pyplot as plt

def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,224,224,0),
              interpolation='bilinear', cmap='magma');

show_heatmap(avg_acts)