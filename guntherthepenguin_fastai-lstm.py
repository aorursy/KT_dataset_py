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

df_all['train']=['train']*len(df_trn)+['test']*len(df_test)
df_all.columns
import seaborn as sns
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['orientation_X','orientation_Y','orientation_Z','orientation_W'])
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z'])
sns.pairplot(df_all.sample(frac=0.1),hue='train',vars=['linear_acceleration_X', 'linear_acceleration_Y', 'linear_acceleration_Z'])
cols=['linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

for col in cols:



    df_trn[col]=(df_trn[col])/(85)

    df_test[col]=(df_test[col])/(85)

cols=['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X', 'angular_velocity_Y', 'angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

test_list=df_test.groupby('series_id')

train_list=df_trn.groupby('series_id')
def open_image(self,i):

    mn=(np.hstack([self.items[i][1]['measurement_number'][:,None] for j in range(8)])-64)/256

    #mn=np.zeros_like(mn)

    feats=np.hstack([self.items[i][1][cols] for j in range(6)])

    feats_mean=(feats-feats.mean())/feats.std()

    img= np.append(np.append(feats,mn,axis=1),feats_mean,axis=1)

    img=np.stack([img,img.T,img[::-1,::-1]],axis=2)

    return pil2tensor(img,np.float32)
def get(self,i):

    return tensor(np.append((self.items[i][1]['measurement_number'][:,None].astype(np.float32)-64)/512,self.items[i][1][cols].values.astype(np.float32),axis=1))
ItemList.get=get
src=(ItemList(df_trn.groupby('series_id'),inner_df=df_label).split_by_rand_pct(0.2).label_from_df(cols='surface'))

src.add_test(test_list,label='concrete');

data=src.databunch(bs=32)

x,y=data.one_batch()
x.shape

plt.pcolor(to_np(x)[5,:,:])

plt.colorbar()
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
model = LSTMClassifier(11, 256, 2, 0.2, True, 9, 32)



learn=Learner(data,model,metrics=accuracy)



learn.lr_find(num_it=200)



learn.recorder.plot()



df_sub=pd.read_csv(path/'sample_submission.csv')
X=list(df_trn.groupby('series_id').indices.keys())

label,group=df_label.set_index('series_id').loc[df_trn.groupby('series_id').indices.keys(),'surface'],df_label.set_index('series_id').loc[df_trn.groupby('series_id').indices.keys(),'group_id']

src_list=ItemList(df_trn.groupby('series_id'),inner_df=df_label)

#for i,(idx_train,idx_val) in enumerate(sss.split(np.unique(df_trn.series_id), df_label.surface)):

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

sss.get_n_splits(X, label,group)
idx_trn,idx_val=next(sss.split(X, label,group))
len(idx_trn),len(idx_val)
def accuracy_mult(input:Tensor, targs:Tensor)->Rank0Tensor:

    "Compute accuracy with `targs` when `input` is bs * n_classes."

    n = targs.shape[0]

    input = input.argmax(dim=-1).view(n,-1)

    targs = targs.argmax(dim=-1).view(n,-1).long()

    return (input==targs).float().mean()
target_probs=[]
for i,(idx_train,idx_val) in enumerate(sss.split(X, label,group)):

    src=(src_list.split_by_idxs(idx_train,idx_val).label_from_df(cols='surface'))

    src.add_test(test_list);

    data=src.databunch(bs=32)

    model = LSTMClassifier(11, 256, 2, 0.2, True, 9, 32)

    learn=Learner(data,model,metrics=accuracy)



    learn.fit_one_cycle(20,1e-3)

    learn.recorder.plot_losses()

    learn.recorder.plot_metrics()

    x,y=learn.get_preds(ds_type=DatasetType.Test)

    target_probs.append(x)

    
preds,y,losses = learn.get_preds(with_loss=True)

interp = ClassificationInterpretation(learn, preds, y, losses)
interp.plot_confusion_matrix(figsize=(16,9))
np.unique(df_label.surface,return_counts=True)
target_probs_stacked=np.stack(target_probs,axis=2)

target_probs_stacked
x=(target_probs_stacked).mean(axis=2)

df_sub['mean']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]
x=np.median((target_probs_stacked),axis=2)

df_sub['median']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]
x=(target_probs_stacked).max(axis=2)

df_sub['max']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]
x=target_probs_stacked[(target_probs_stacked<target_probs_stacked.max(axis=2)[:,:,None])&(target_probs_stacked>target_probs_stacked.min(axis=2)[:,:,None])].reshape(list(target_probs_stacked.shape[0:2])+[3])
x=x.mean(axis=2)
df_sub['truncated']=[learn.data.classes[idx] for idx in np.argmax(x,axis=1)]
df_sub=df_sub.drop(columns='surface')
df_sub=df_sub.set_index('series_id')
df_sub['mean'].to_csv('mean.csv',header=['surface'])

df_sub['median'].to_csv('median.csv',header=['surface'])

df_sub['max'].to_csv('max.csv',header=['surface'])

df_sub['truncated'].to_csv('truncated.csv',header=['surface'])
for i in range(5):

    df_sub[f'sub_{i}']=[learn.data.classes[idx] for idx in np.argmax(target_probs_stacked[:,:,i],axis=1)]

    df_sub[f'sub_{i}'].to_csv(f'sub{i}.csv',header=['surface'])
