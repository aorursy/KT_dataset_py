# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
%reload_ext autoreload

%autoreload 2

%matplotlib inline
from fastai import *

from fastai.vision import *

import pandas as pd
path = Path('/kaggle/input/aptos2019-blindness-detection')

path.ls()
df_train = pd.read_csv(path/'train.csv')

df_test = pd.read_csv(path/'test.csv')
df_train.head()
aptos19_stats = ([0.42, 0.22, 0.075], [0.27, 0.15, 0.081])

data = ImageDataBunch.from_df(df=df_train,

                              path=path, folder='train_images', suffix='.png',

                              valid_pct=0.1,

                              ds_tfms=get_transforms(flip_vert=True, max_warp=0.1, max_zoom=1.15, max_rotate=45.),

                              size=224,

                              bs=32, 

                              num_workers=os.cpu_count()

                             ).normalize(aptos19_stats)
data.c, data.train_ds, data.valid_ds, data.test_ds, data.classes
data.show_batch(rows=3, figsize=(7,7))
learn = cnn_learner(data, models.resnet34, metrics=error_rate)
learn.fit_one_cycle(4)
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_top_losses(9,figsize=(20,11))
learn.unfreeze()
learn.fit_one_cycle(1)
learn.unfreeze()

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))
learn.model_dir=Path('/kaggle/working')

learn.save('stage-1')
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix(figsize=(12,12), dpi=100)
interp.most_confused()
sample_submission = pd.read_csv('../input/aptos2019-blindness-detection/sample_submission.csv')

sample_submission.head()
preds, targs, loss = learn.get_preds(with_loss=True)
# get accuracy

acc = accuracy(preds, targs)

print('The accuracy is {0} %.'.format(acc))
from sklearn.metrics import roc_curve, auc

# probs from log preds

probs = np.exp(preds[:,1])

# Compute ROC curve

fpr, tpr, thresholds = roc_curve(targs, probs, pos_label=1)



# Compute ROC area

roc_auc = auc(fpr, tpr)

print('ROC area is {0}'.format(roc_auc))
plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([-0.01, 1.0])

plt.ylim([0.0, 1.01])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic')

plt.legend(loc="lower right")
learn.load('stage-1')
path
learn.data.add_test(ImageList.from_df(

    sample_submission, path,

    folder='test_images',

    suffix='.png'

))
# remove zoom from FastAI TTA

tta_params = {'beta':0.12, 'scale':1.0}
preds,y=learn.TTA(ds_type=DatasetType.Test,**tta_params)
sample_submission.diagnosis = preds.argmax(1)

sample_submission.head()
sample_submission.to_csv('submission.csv',index=False)

_ = sample_submission.hist()