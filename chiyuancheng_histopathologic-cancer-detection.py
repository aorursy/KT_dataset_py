from fastai import *

from fastai.vision import *

from torchvision.models import * 



import os

import pathlib

import matplotlib.pyplot as plt
#data = pd.read_csv('/kaggle/input/train_labels.csv')

#train_path = '/kaggle/input/train/'

#test_path = '/kaggle/input/test/'

# quick look at the label stats

#data['label'].value_counts()



path = Path("../input")

labels = pd.read_csv(path/"train_labels.csv")

labels.head()



#path_w =Path("../directory")
ls
print(labels["label"].nunique()); classes = list(set(labels["label"])); classes

for i in classes:

    print("Number of items in class {} is {}".format(i,len(labels[labels["label"] == i])))
tfms = get_transforms(do_flip=True, flip_vert=True, max_rotate=.0, max_zoom=.1,

                      max_lighting=0.05, max_warp=0.)
np.random.seed(123)

sz = 32

data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test',suffix=".tif", size = sz,bs = 128,

                               ds_tfms = tfms)

#data.path = pathlib.Path('.')

data.normalize(imagenet_stats)
print(data.classes); data.c
data.show_batch(rows=3, figsize=(5,5))
from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    return score
arch = models.densenet169

learn = create_cnn(data,arch,pretrained = True,ps = 0.5,metrics = [auc_score,accuracy], model_dir='/tmp/models')
# learn.save('stage-0')
# #learn.lr_find()

# learn.lr_find()

# #lr_find??
# learn.recorder.plot(learn)
# lr = 0.001

# learn.fit_one_cycle(1, slice(lr))

# learn.save('stage-1')
# learn.load('stage-1')
# learn.unfreeze()

# learn.lr_find()
# learn.recorder.plot()
# learn.fit_one_cycle(3,max_lr = slice(5e-2,1e-3))
# interp = ClassificationInterpretation.from_learner(learn)
# interp.plot_top_losses(9)
# interp.plot_confusion_matrix()
# newTfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.5)

# newSz = 96

# newData = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

#                                test = 'test',suffix=".tif", size = newSz, ds_tfms = newTfms)

# newData.path = pathlib.Path('.')

# newData.normalize(imagenet_stats)

# learn.data = newData
# learn.freeze()

# learn.lr_find()
# learn.recorder.plot()
#learn.fit_one_cycle(5,max_lr = slice(5e-3,5e-5))