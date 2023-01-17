# settings

%reload_ext autoreload

%autoreload 2

%matplotlib inline
# load libraries

from fastai import *

from fastai.vision import *

import pandas as pd

import cv2
size = 16 # ssize of input images

bs = 64 # batch size

tfms = get_transforms()
! ls ../input/fashionmnist/data/data
path = Path('../input/fashionmnist/data/data')

cv2.imread(str((path/'train/Coat').ls()[0])).shape
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='train',test='test',

                                 ds_tfms=tfms, size=size, bs=bs,valid_pct=.2).normalize(imagenet_stats)

data
data.show_batch(rows=3)
model = models.resnet18
data.path = '/tmp/.torch/models'
learn = cnn_learner(data, model, metrics=accuracy,callback_fns=[ShowGraph])
learn.summary()
learn.lr_find()

learn.recorder.plot()
learn.save("stage-1")
lr = 2e-2
learn.fit_one_cycle(9,slice(lr))
learn.fit_one_cycle(4,slice(lr))
learn.unfreeze()
lr = lr /100

learn.fit_one_cycle(4,slice(lr))
accuracy(*learn.TTA())
learn.save('stage-2')
learn.load('stage-2')

pass
size = 28
# Load data to DataBunch

data = ImageDataBunch.from_folder(path,train='train',test='test',

                                 ds_tfms=tfms, size=size, bs=bs,valid_pct=.2).normalize(imagenet_stats)

data
learn.data = data
learn.freeze()

learn.lr_find()

learn.recorder.plot()
lr = 3e-4
learn.fit_one_cycle(5,slice(lr))
learn.fit(6)
learn.unfreeze()
learn.fit_one_cycle(5,slice(1e-4))
accuracy(*learn.TTA())
learn.save('stage-3')
! wget https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/beautycheck/english/prototypen/w_sexy_gr.jpg
from PIL import Image

import cv2
from fastai.vision import Image,pil2tensor
def array2tensor(x):

    """ Return an tensor image from cv2 array """

    x = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)

    return Image(pil2tensor(x,np.float32).div_(255))
!wget https://i.pinimg.com/originals/c0/41/8f/c0418f5967b642a1f01864409fb2f86a.jpg
! ls
from PIL import Image as I
img = cv2.imread('c0418f5967b642a1f01864409fb2f86a.jpg')
I.open('c0418f5967b642a1f01864409fb2f86a.jpg')
img = array2tensor(img)
learn.predict(img)
interp = ClassificationInterpretation.from_learner(learn)



losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
interp.plot_top_losses(9, figsize=(15,11))
learn.export('')
learn =  load_model_from_export('')
learn.predict('')
interp.most_confused(min_val=2)
planet = untar_data(URLs.PLANET_TINY)

planet_tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
pd.read_csv(planet/"labels.csv").head()
data = ImageDataBunch.from_csv(planet, folder='train',csv_labels='labels.csv', size=128, suffix='.jpg', label_delim = ' ', ds_tfms=planet_tfms)
data.show_batch(rows=2, figsize=(9,7))
# folders = ['tesla','lambo','audi']

# files = ['urls_tesla.csv','urls_lambo.csv','urls_audi.csv']
# urls = """

# https://st.motortrend.com/uploads/sites/10/2017/09/2018-audi-r8-coupe-angular-front.png

# https://www.cstatic-images.com/car-pictures/xl/usc90aus061a021001.png

# https://st.motortrend.com/uploads/sites/10/2015/11/2015-audi-rs7-hatchback-angular-front.png

# https://upload.wikimedia.org/wikipedia/commons/d/d2/2018_Audi_A7_S_Line_40_TDi_S-A_2.0.jpg

# """
# pd.DataFrame(urls.strip().split('\n')).to_csv(path/'urls_audi.csv',index=False)
# for file,folder in zip(files,folders):

#     path = Path('data/cars')

#     dest = path/folder

#     dest.mkdir(parents=True, exist_ok=True)

#     download_images(path/file, dest, max_pics=200)
#download_images(path/file, dest, max_pics=200)