#Import Libs

import numpy as np 

import pandas as pd 

from fastai import *

from fastai.vision import *

from torchvision.models import *

import os 

import path
# print(fastai.__version__)

imgs = os.listdir('../input/train')[:5]

arch = resnet50 

bs = 64

sz = 48 #gonna start out with 48x48 px and then gonna move up ro the original size 96x96 px 

path = Path('../input')

open_image('../input/train/'+imgs[1])
# look at the csv

df = pd.read_csv(path/'train_labels.csv')

df.head()
# Use data_block API to define the default specs for each databunch we create... 

tfms = get_transforms()

src = (ImageItemList.from_csv(path, 'train_labels.csv', folder = 'train', suffix = '.tif')

      .random_split_by_pct()

      .label_from_df()

      .add_test_folder('test'))
np.random.seed(50)

data = (src.transform(tfms, size = sz)

       .databunch().normalize(imagenet_stats)) 
data.show_batch(rows = 3, figsize=(11,12))
#thanks to Khoi Nguyen's kernel :https://www.kaggle.com/suicaokhoailang/wip-densenet121-baseline-with-fastai

from sklearn.metrics import roc_auc_score



def auc_score(y_pred,y_true,tens=True):

    score = roc_auc_score(y_true,torch.sigmoid(y_pred)[:,1])

    if tens:

        score = tensor(score)

    return score
learn = create_cnn(data, arch, metrics = [accuracy, auc_score], model_dir = '/tmp/models/')
learn.lr_find()

learn.recorder.plot()
lr = 1e-02 

learn.fit_one_cycle(5, slice(lr))
learn.save('model-1-rn50')
learn.unfreeze()

learn.lr_find() 

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-05, lr/5))
learn.save('model-2-rn50')
#increase the size and replace the old data w/ the new one and do transfer learning 

data = (src.transform(tfms, size = 96)

       .databunch().normalize(imagenet_stats))

learn.data = data 

data.train_ds[0][0].shape
learn.freeze()

learn.lr_find()

learn.recorder.plot()
lr = 1e-03 

learn.fit_one_cycle(5, slice(lr))
learn.save('model-big-1-rn50')
learn.unfreeze()

learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(3e-06, lr/5))
learn.save('model-big-2-rn50')
interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()
losses,idxs = interp.top_losses()
interp.plot_confusion_matrix(figsize=(12,12), dpi=60)
preds,y = learn.TTA()

acc = accuracy(preds, y)

print('The validation accuracy is {} %.'.format(acc * 100))
pred_score = auc_score(preds,y).item()

print('The validation AUC is {}.'.format(pred_score))
df1 = pd.read_csv(path/'sample_submission.csv')

id_list = list(df1.id)

preds,y = learn.TTA(ds_type=DatasetType.Test)

pred_list = list(preds[:,1])
pred_dict = dict((key, value.item()) for (key, value) in zip(learn.data.test_ds.items,pred_list))

pred_ordered = [pred_dict[Path('../input/test/' + id + '.tif')] for id in id_list]
submissions = pd.DataFrame({'id':id_list,'label':pred_ordered})

submissions.to_csv("submission_{}.csv".format(pred_score),index = False)
