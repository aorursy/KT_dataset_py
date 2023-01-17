%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

from torchvision.models import * 



import os

import matplotlib.pyplot as plt
path = Path("../input")

labels = pd.read_csv(path/"train_labels.csv")

labels.head()
print(labels["label"].nunique()); classes = list(set(labels["label"])); classes
for i in classes:

    print("Number of items in class {} is {}".format(i,len(labels[labels["label"] == i])))
tfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.1)
np.random.seed(123)

sz = 32

data = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test',suffix=".tif", size = sz,bs = 256,

                               ds_tfms = tfms)

data.path = pathlib.Path('.')

data.normalize(imagenet_stats)
print(data.classes); data.c
data.show_batch(rows = 3)
from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    return score
arch = models.densenet121

learn = cnn_learner(data,arch,pretrained = True,ps = 0.45,metrics = [auc_score,accuracy])
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8,1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,max_lr = slice(1e-5,1e-3))
newTfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.25)

newSz = 64

newData = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test',suffix=".tif", size = newSz, ds_tfms = newTfms)

newData.path = pathlib.Path('.')

newData.normalize(imagenet_stats)

learn.data = newData
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(8,1e-2)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,max_lr = slice(1e-3/3,1e-3))
learn.save('stage-2')
preds,y = learn.TTA()

acc = accuracy(preds, y)

print('The validation accuracy is {} %.'.format(acc * 100))

pred_score = auc_score(preds,y).item()

print('The validation AUC is {}.'.format(pred_score))
newTfms = get_transforms(do_flip = True,flip_vert = True,max_zoom = 1.5)

newSz = 96

newData = ImageDataBunch.from_csv(path, folder = 'train', csv_labels = "train_labels.csv",

                               test = 'test',suffix=".tif", size = newSz, ds_tfms = newTfms)

newData.path = pathlib.Path('.')

newData.normalize(imagenet_stats)

learn.data = newData
learn.freeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(4,1e-4)
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(2,max_lr = slice(1e-5/3,1e-5))
learn.save('stage-3')
preds,y = learn.TTA()

acc = accuracy(preds, y)

print('The validation accuracy is {} %.'.format(acc * 100))

pred_score = auc_score(preds,y).item()

print('The validation AUC is {}.'.format(pred_score))
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(6)
interp.plot_confusion_matrix()
preds,y = learn.TTA()

acc = accuracy(preds, y)

print('The validation accuracy is {} %.'.format(acc * 100))

pred_score = auc_score(preds,y).item()

print('The validation AUC is {}.'.format(pred_score))
def generateSubmission(learner):

    submissions = pd.read_csv('../input/sample_submission.csv')

    id_list = list(submissions.id)

    preds,y = learner.TTA(ds_type=DatasetType.Test)

    pred_list = list(preds[:,1])

    pred_dict = dict((key, value.item()) for (key, value) in zip(learner.data.test_ds.items,pred_list))

    pred_ordered = [pred_dict[Path('../input/test/' + id + '.tif')] for id in id_list]

    submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})

    submissions.to_csv("submission_transferLearning_{}.csv".format(pred_score),index = False)
generateSubmission(learn)