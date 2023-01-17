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
import numpy as np

import pandas as pd

from sklearn.model_selection import GroupKFold

from sklearn.metrics import mean_squared_error

import gc

from tqdm import tqdm_notebook as tqdm

import random



import torch

import torch.nn as nn

import torch.nn.functional as F

from torch.utils.data import TensorDataset, DataLoader



from kaggle.competitions import nflrush



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from fastai.basics import *
train_path = '/kaggle/input/nfl-big-data-bowl-2020/train.csv'

train = pd.read_csv(train_path)

print(train.shape)
use_cols = [

    'GameId', 

    'PlayId', 

    'Team',

    'Yards',

    'X',

    'Y',

    'PossessionTeam',

    'HomeTeamAbbr',

    'VisitorTeamAbbr',

    'Position',

]

train = train[use_cols]

train.head()
"""

Find an offense team.

ref: https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/112314#latest-648026

"""

def fe_is_offence(row):

    if row["Team"] == "home":

        if row["HomeTeamAbbr"] == row["PossessionTeam"]:

            return 1

        else:

            return 0

    elif row["Team"] == "away":

        if row["VisitorTeamAbbr"] == row["PossessionTeam"]:

            return 1

        else:

            return 0



def fe_is_offence_from_position(row, off_team):

    if row["Team"] == off_team:

        return 1

    else:

        return 0

        

# def run_fe_is_offence(df):

#     df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)

    

#     if (df['is_offence'].values == 0).all():

#         off_team = df[df['Position']=='QB']['Team'].values[0]

#         df['is_offence'] = df.apply(lambda row: fe_is_offence_from_position(row, off_team), axis=1)



"""

bugfix

"""

def run_fe_is_offence(df):

    df['is_offence'] = df.apply(lambda row: fe_is_offence(row), axis=1)

    

    check_is_offence = df.groupby('PlayId')['is_offence'].nunique()

    is_offence_not_found_idx = check_is_offence[check_is_offence!=2].index

    not_found_df = df[df['PlayId'].isin(is_offence_not_found_idx)]

    found_df = df[~df['PlayId'].isin(is_offence_not_found_idx)]

#     print('is_offence found: {}'.format(len(found_df)))

#     print('is_offence not found: {}'.format(len(not_found_df)))



    for u_play_id in not_found_df['PlayId'].unique():

        tmp_df = not_found_df[not_found_df['PlayId']==u_play_id]

        pos_list = [pos for pos in tmp_df['Position'].unique() if pos in ['QB', 'RB', 'WR', 'TE']]

        

        if len(pos_list) > 0:

            off_team = tmp_df[tmp_df['Position']==pos_list[0]]['Team'].values[0]

#         else:

#             print('Offence position not found')

#             import pdb;pdb.set_trace()



        target_idx = not_found_df.query('PlayId==@u_play_id and Team==@off_team').index

        not_found_df.loc[target_idx, 'is_offence'] = 1

    

    df = pd.concat([found_df, not_found_df], sort=False)

#     print('done df: {}'.format(df.shape))

    return df
def run_group_fe(df, group_key, aggs):

    

    group_df = df.groupby(group_key).agg(aggs)



    new_cols = [col[0]+'_'+col[1] for col in group_df.columns]

    group_df.columns = new_cols

    group_df.reset_index(inplace=True)

        

    return group_df



def adjust_group_df(group_df, is_train):

    offence_df = group_df[group_df['is_offence']==1]

    deffence_df = group_df[group_df['is_offence']==0]



    del group_df['is_offence']

    del offence_df['is_offence']

    del deffence_df['is_offence']

    

    if is_train:

        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]

        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId', 'Yards'] else col for col in group_df.columns]

    else:

        off_cols = ['off_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]

        deff_cols = ['deff_{}'.format(col) if col not in ['GameId', 'PlayId'] else col for col in group_df.columns]

        

    offence_df.columns = off_cols

    deffence_df.columns = deff_cols

    if is_train: del deffence_df['Yards']

    

    adjusted_group_df = pd.merge(offence_df, deffence_df, on=['GameId', 'PlayId'])

    

    return adjusted_group_df
train = run_fe_is_offence(train)

train.head()
train_group_key = ['GameId', 'PlayId', 'is_offence', 'Yards']

aggs = {

    'X': ['mean', 'max', 'min', 'median'],

    'Y': ['mean', 'max', 'min', 'median'],

}

is_train = True

group_df = run_group_fe(train, train_group_key, aggs)

adjusted_group_df = adjust_group_df(group_df, is_train)
print(adjusted_group_df.shape)

adjusted_group_df.head()
class NFL_NN(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()



        self.fc1 = nn.Linear(in_features, 216)

        self.bn1 = nn.BatchNorm1d(216)

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(216, 512)

        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(512, 216)

        self.relu3 = nn.ReLU()

        self.dout3 = nn.Dropout(0.2)

        self.out = nn.Linear(216, out_features)

        self.out_act = nn.Sigmoid()

        

    def forward(self, input_):

        a1 = self.fc1(input_)

        bn1 = self.bn1(a1)

        h1 = self.relu1(bn1)

        a2 = self.fc2(h1)

        h2 = self.relu2(a2)

        a3 = self.fc3(h2)

        h3 = self.relu3(a3)

        dout3 = self.dout3(h3)

        a5 = self.out(dout3)

        y = self.out_act(a5)

        return a5
epoch = 10

batch_size = 64
oof_crps_list = []

fold = GroupKFold(n_splits=5)





y = np.zeros(shape=(adjusted_group_df.shape[0], 199))

for i, yard in enumerate(adjusted_group_df['Yards'].values):

#     print(i, yard)

    y[i, yard+99:] = np.ones(shape=(1, 100-yard))



oof_preds = np.ones((len(adjusted_group_df), y.shape[1]))



feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]



print('use feats: {}'.format(len(feats)))



for n_fold, (train_idx, valid_idx) in enumerate(fold.split(adjusted_group_df, y, groups=adjusted_group_df['GameId'])):

        print('Fold: {}'.format(n_fold+1))

        

        train_x, train_y = adjusted_group_df[feats].iloc[train_idx].values, y[train_idx]

        valid_x, valid_y = adjusted_group_df[feats].iloc[valid_idx].values, y[valid_idx] 



#         train_x = torch.tensor(train_x, requires_grad=True).float()

#         train_y = torch.tensor(train_y, requires_grad=True).float()

#         valid_x = torch.tensor(valid_x, requires_grad=False).float()

#         valid_y = torch.tensor(valid_y, requires_grad=False).float()



        train_x = torch.from_numpy(train_x).float()

        train_y = torch.from_numpy(train_y).float()

        valid_x = torch.from_numpy(valid_x).float()

        valid_y = torch.from_numpy(valid_y).long()



        train_dataset = TensorDataset(train_x, train_y)

        valid_dataset = TensorDataset(valid_x, valid_y)



        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

        

        print('train: {}, valid: {}'.format(len(train_dataset), len(valid_dataset)))
data = DataBunch.create(train_dataset, valid_dataset, bs=batch_size, num_workers=0)
in_features = adjusted_group_df[feats].shape[1]

out_features = y.shape[1]
# loss_func = nn.MSELoss()

# optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
class CustomLoss(nn.Module):

    

    def __init__(self):

        super(CustomLoss, self).__init__()

        

    def forward(self,y_hat,target):

        return torch.sqrt(nn.MSELoss()(y_hat.float(), target.view((len(target), out_features)).float()))
class CRPS(Callback):

    def on_epoch_begin(self, **kwargs):

        self.crps, self.targ_count = 0, 0

        

    def on_batch_end(self, last_output, last_target, **kwargs):

        self.crps += np.sum(np.power(last_output.numpy() - last_target.numpy(), 2))

        self.targ_count += len(last_target)

        

    def on_epoch_end(self, last_metrics, **kwargs):

        

        return add_metrics(last_metrics, self.crps / (199*self.targ_count))
learn = Learner(data, NFL_NN(in_features, out_features), loss_func=CustomLoss(), metrics=CRPS())
learn.lr_find()

learn.recorder.plot()
learn.fit_one_cycle(5, slice(1e-2))
learn.recorder.plot_losses()
# x_bs,y_bs = next(iter(data.train_dl))

# x_bs.shape, y_bs.shape
def min_max_scaler(x):

    return (x - np.min(x)) / (np.max(x) - np.min(x))
env = nflrush.make_env()
result_df = None

is_train = False

test_group_key = ['GameId', 'PlayId', 'is_offence']



for (test_df, sample_prediction_df) in tqdm(env.iter_test()):

    

    test_df = run_fe_is_offence(test_df)

    test_group_df = run_group_fe(test_df, test_group_key, aggs)

    test_adjusted_group_df = adjust_group_df(test_group_df, is_train)

    

    feats = [

        "off_X_mean","off_X_max","off_X_min","off_X_median","off_Y_mean","off_Y_max","off_Y_min","off_Y_median",

        "deff_X_mean","deff_X_max","deff_X_min","deff_X_median","deff_Y_mean","deff_Y_max","deff_Y_min","deff_Y_median",

    ]

    test = torch.from_numpy(test_adjusted_group_df[feats].values)

    test_dataset = TensorDataset(test)

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    

    in_features = test_adjusted_group_df[feats].shape[1]

    out_features = 199

    

    learn.model.eval()

    preds = np.zeros((len(test_dataset), out_features))

    

    with torch.no_grad():

        for i, test_x_batch in enumerate(test_loader):

            test_values = test_x_batch[0].float()

            pred = learn.model(test_values)

            preds[i * batch_size:(i + 1) * batch_size] = pred

            

    y_pred = preds.copy()

    adjust_preds = np.zeros((len(y_pred), y_pred.shape[1]))

    for idx, pred in enumerate(y_pred):

        

        prev = 0

        for i in range(len(pred)):

            if pred[i]<prev:

                pred[i]=prev

            prev=pred[i]

        x = min_max_scaler(pred)

        adjust_preds[idx, :] = x



    adjust_preds[:, -1] = 1

    adjust_preds[:, 0] = 0



    preds_df = pd.DataFrame(data=adjust_preds.reshape(-1, 199), columns=sample_prediction_df.columns)

    env.predict(preds_df)



    if result_df is None:

        result_df = preds_df

    else:

        result_df = pd.concat([result_df, preds_df], sort=False)
env.write_submission_file()
result_df.drop_duplicates().shape
result_df.head(30)