# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

from fastai.vision import *

from sklearn.model_selection import StratifiedShuffleSplit

# Any results you write to the current directory are saved as output.

import warnings

warnings.simplefilter("ignore")

from pathlib import Path

path=Path('../input')

df_trn=pd.read_csv(path/'X_train.csv')

df_label=pd.read_csv(path/'y_train.csv')

df_test=pd.read_csv(path/'X_test.csv')

df_all=pd.concat([df_trn,df_test])

df_all['train']=[0]*len(df_trn)+[1]*len(df_test)

df_all.series_id[df_all.train==1]+=len(df_trn)

df_label=pd.DataFrame(data=[0]*len(np.unique(df_trn.series_id))+[1]*len(np.unique(df_test.series_id)),columns=['train'])
from tqdm import tqdm



import seaborn as sns
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['orientation_X','orientation_Y','orientation_Z','orientation_W'])
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z'])
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z'])
cols=['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

for col in cols:



    df_all[col]=(df_all[col])/(85)

cols=['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

def get(self,i):

    return tensor(np.append((self.items[i][1]['measurement_number'][:,None].astype(np.float32)-64)/512,self.items[i][1][cols].values.astype(np.float32),axis=1))
sample_list=df_all.groupby('series_id')
ItemList.get=get
src=(ItemList(sample_list,inner_df=df_label).split_by_rand_pct(0.2).label_from_df(cols='train'))

data=src.databunch(bs=32)

class LSTMClassifier(nn.Module):



    def __init__(self, in_dim, hidden_dim, num_layers, dropout, bidirectional, num_classes, batch_size):

        super(LSTMClassifier, self).__init__()

        self.in_dim = in_dim

        self.hidden_dim = hidden_dim

        self.batch_size = batch_size

        self.bidirectional = bidirectional

        self.num_dir = 2 if bidirectional else 1

        self.num_layers = num_layers

        self.dropout = dropout



        self.lstm = nn.LSTM(input_size=self.in_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers, dropout=self.dropout, bidirectional=self.bidirectional,

                            batch_first=True)

        self.gru = nn.GRU(self.hidden_dim * 2, self.hidden_dim, bidirectional=self.bidirectional, batch_first=True)

        self.fc = nn.Sequential(

            nn.Linear(2048, hidden_dim),

            nn.ReLU(True),

            nn.Dropout(p=dropout),

            nn.Linear(hidden_dim, num_classes),

        )



    def forward(self, x):



        lstm_out, _ = self.lstm(x)

        gru_out, _ = self.gru(lstm_out)

        avg_pool_l = torch.mean(lstm_out, 1)

        max_pool_l, _ = torch.max(lstm_out, 1)

        

        avg_pool_g = torch.mean(gru_out, 1)

        max_pool_g, _ = torch.max(gru_out, 1)

        x = torch.cat((avg_pool_g, max_pool_g, avg_pool_l, max_pool_l), 1)

        y = self.fc(x)

        return y
model = LSTMClassifier(11, 256, 2, 0.5, True, 2, 32)



learn=Learner(data,model,metrics=accuracy)



learn.lr_find(num_it=200)



learn.recorder.plot()



src_list=ItemList(sample_list,inner_df=df_label)

#for i,(idx_train,idx_val) in enumerate(sss.split(np.unique(df_trn.series_id), df_label.surface)):

from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,

                                     StratifiedKFold, GroupShuffleSplit,

                                     GroupKFold, StratifiedShuffleSplit)

sss = StratifiedKFold(n_splits=5, random_state=0)

sss.get_n_splits(sample_list, df_label.train)
for i,(idx_train,idx_val) in enumerate(sss.split(sample_list, df_label.train)):

    print(df_label.train[idx_train].mean(),df_label.train[idx_val].mean())
for i,(idx_train,idx_val) in enumerate(sss.split(sample_list, df_label.train)):

    src=(src_list.split_by_idxs(idx_train,idx_val).label_from_df(cols='train'))

    data=src.databunch(bs=32)

    model = LSTMClassifier(11, 256, 2, 0.5, True, 2, 32)

    learn=Learner(data,model,metrics=accuracy)

    learn.fit_one_cycle(15,1e-3)

    learn.recorder.plot_losses()

    learn.recorder.plot_metrics()

    x,y=learn.get_preds()

    
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix(figsize=(16,9))