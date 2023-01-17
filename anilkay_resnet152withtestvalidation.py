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

        break



# Any results you write to the current directory are saved as output.
from fastai.vision import *

data = ImageDataBunch.from_folder("/kaggle/input/messy-vs-clean-room/images/", train="train",valid='val', test='test', size=224, bs=48)

data.show_batch()
learn = cnn_learner(data, 

                     models.resnet152, 

                    metrics=accuracy)

learn.fit(25)
interp = ClassificationInterpretation.from_learner(learn)

interp.plot_top_losses(9, figsize=(14,14))
interp.plot_confusion_matrix()
interp.plot_multi_top_losses()
ytrue=interp.y_true

ypred=interp.pred_class

from sklearn.metrics import classification_report

print(classification_report(y_true=ytrue,y_pred=ypred))
from sklearn.metrics import accuracy_score

print(accuracy_score(y_true=ytrue,y_pred=ypred))
interp.ds_type
test_validation = ClassificationInterpretation.from_learner(learn,ds_type=DatasetType.Test)

test_validation.plot_confusion_matrix()
test_validation.plot_multi_top_losses(10)