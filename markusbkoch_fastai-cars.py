%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai.vision import *

from fastai.metrics import error_rate
import torch



def my_setseed(s=42): #set seed, for reproducible results

    np.random.seed(s)

    torch.manual_seed(s)

    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)
bs = 64

# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart
# path = untar_data(URLs.PETS); path

path = Path('/kaggle/input/carros/data/data/cars'); path
path.ls()
# path_anno = path/'annotations'

# path_img = path/'images'
# fnames = get_image_files(path_img)

fnames = get_image_files(path, recurse=True)

fnames[:5]
my_setseed()

# pat = r'/([^/]+)_\d+.jpg$'

pat = r'/([^/]+)/[^/]+$' #regex to get class from folder name
# data = ImageDataBunch.from_name_re( path_img, fnames, pat, ds_tfms=get_transforms(), size=224, bs=bs

#                                   ).normalize(imagenet_stats)

data = ImageDataBunch.from_name_re(path, fnames, pat=pat, ds_tfms=get_transforms(), size=224, bs=bs, num_workers=1

                                  ).normalize(imagenet_stats)
data.show_batch(rows=3, figsize=(7,6))
print(data.classes)

len(data.classes),data.c
def _plot(i,j,ax):

    x,y = data.train_ds[3]

    x.show(ax, y=y)



plot_multi(_plot, 3, 3, figsize=(8,8))
my_setseed()

learn = cnn_learner(data, models.resnet34, metrics=error_rate, model_dir="/tmp/model/")

learn.fit_one_cycle(4)
learn.save('basico')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, slice(3e-4,3e-3))
learn.save('finetuned')
learn.fit_one_cycle(1)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))
my_setseed()

learn = cnn_learner(data, models.resnet50, metrics=error_rate, model_dir="/tmp/model/")

learn.fit_one_cycle(4)
learn.save('basico-50')
learn.lr_find()

learn.recorder.plot()
learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-4,2e-3))
learn.load('basico-50');
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()



len(data.valid_ds)==len(losses)==len(idxs)
interp.plot_confusion_matrix(figsize=(5,5), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))