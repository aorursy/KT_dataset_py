# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from fastai.tabular import *
from sklearn.metrics import roc_auc_score



def auroc_score(input, target):

    input, target = input.cpu().numpy()[:,1], target.cpu().numpy()

    return roc_auc_score(target, input)



class AUROC(Callback):

    _order = -20 #Needs to run before the recorder



    def __init__(self, learn, **kwargs): self.learn = learn

    def on_train_begin(self, **kwargs): self.learn.recorder.add_metric_names(['AUROC'])

    def on_epoch_begin(self, **kwargs): self.output, self.target = [], []

    

    def on_batch_end(self, last_target, last_output, train, **kwargs):

        if not train:

            self.output.append(last_output)

            self.target.append(last_target)

                

    def on_epoch_end(self, last_metrics, **kwargs):

        if len(self.output) > 0:

            output = torch.cat(self.output)

            target = torch.cat(self.target)

            preds = F.softmax(output, dim=1)

            metric = auroc_score(preds, target)

            return add_metrics(last_metrics, [metric])
train_df = pd.read_csv('../input/creditcard.csv')
cont_names = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',

       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',

       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',]



dep_var='Class'

procs=[ Normalize]
data = (TabularList.from_df(train_df, cont_names=cont_names , procs=procs,)

                .split_subsets(train_size=0.8, valid_size=0.2, seed=34)

                .label_from_df(cols=dep_var)

                .databunch())
learn = tabular_learner(data, layers=[200,100],metrics=accuracy, callback_fns=AUROC)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(4, 1e-02)
from imblearn.over_sampling import SMOTE
#save the column name

col_name = train_df.columns

x_col = col_name[:-1]

y_col = col_name[-1]



X = train_df.drop('Class', axis=1)

Y = train_df.Class

X_res, Y_res = SMOTE().fit_resample(X, Y)
smote_df = pd.DataFrame(X_res, columns = x_col)
smote_df = smote_df.assign(Class = Y_res)
smote_df.Class.value_counts()
data = (TabularList.from_df(smote_df, cont_names=cont_names , procs=procs,)

                .split_subsets(train_size=0.8, valid_size=0.2, seed=34)

                .label_from_df(cols=dep_var)

                #.add_test(test)

                .databunch())
learn = tabular_learner(data, layers=[200,100],metrics=accuracy, callback_fns=AUROC)
learn.lr_find()
learn.recorder.plot()
learn.fit_one_cycle(5, 5e-03)
learn.recorder.plot_losses()