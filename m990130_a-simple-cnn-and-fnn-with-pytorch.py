# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

from sklearn import preprocessing

from datetime import datetime

# load the data as DataFrame



data = {

    'tra': pd.read_csv('../input/air_visit_data.csv'),

    'as': pd.read_csv('../input/air_store_info.csv'),

    'hs': pd.read_csv('../input/hpg_store_info.csv'),

    'ar': pd.read_csv('../input/air_reserve.csv'),

    'hr': pd.read_csv('../input/hpg_reserve.csv'),

    'id': pd.read_csv('../input/store_id_relation.csv'),

    'tes': pd.read_csv('../input/sample_submission.csv'),

    'hol': pd.read_csv('../input/date_info.csv').rename(columns={'calendar_date':'visit_date'})

    }



# take the common part of hpg reservation, id relation with key hpg_store_id



data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])



# convert the time info into datetime format, add diff_time col

# group the data with id then visit time, then rename them

# done with the DataFrame 'ar' and 'hr'





for df in ['ar','hr']:

    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime']).dt.date

    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime']).dt.date

    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)

    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})

    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})

    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])



# divide visit_date info into 4 cols 

    

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])

data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek

data['tra']['year'] = data['tra']['visit_date'].dt.year

data['tra']['month'] = data['tra']['visit_date'].dt.month

data['tra']['visit_date'] = data['tra']['visit_date'].dt.date



# apply the same convention on data['tes']



data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])

data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])

data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek

data['tes']['year'] = data['tes']['visit_date'].dt.year

data['tes']['month'] = data['tes']['visit_date'].dt.month

data['tes']['visit_date'] = data['tes']['visit_date'].dt.date





# stores is the aggregate info of 'tra'

# rename

# merge the new DataFrame with air_reserve on key air_id



stores = data['tra'].groupby(['air_store_id','dow']).agg({'visitors' : [np.min,np.mean,np.median,np.max,np.size]}).reset_index()

stores.columns = ['air_store_id', 'dow', 'min_visitors', 'mean_visitors', 'median_visitors','max_visitors','count_observations']

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id']) 



# label the styles and the areas 



# NEW FEATURES FROM Georgii Vyshnia

stores['air_genre_name'] = stores['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))

stores['air_area_name'] = stores['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

lbl = preprocessing.LabelEncoder()

for i in range(4):

    stores['air_genre_name'+str(i)] = lbl.fit_transform(stores['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))

    stores['air_area_name' +str(i)] = lbl.fit_transform(stores['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))



# merge information above into stores as 2 new cols   

    

stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])

stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])



# same work(time &labels) with hol



data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])

data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])

data['hol']['visit_date'] = data['hol']['visit_date'].dt.date



train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date']) 

test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date']) 



train = pd.merge(train, stores, how='left', on=['air_store_id','dow']) 

test = pd.merge(test, stores, how='left', on=['air_store_id','dow'])



for df in ['ar','hr']:

    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 

    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])



train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1)



train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']

train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2

train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2



test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']

test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2

test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2



# NEW FEATURES FROM JMBULL

train['date_int'] = train['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

test['date_int'] = test['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)

train['var_max_lat'] = train['latitude'].max() - train['latitude']

train['var_max_long'] = train['longitude'].max() - train['longitude']

test['var_max_lat'] = test['latitude'].max() - test['latitude']

test['var_max_long'] = test['longitude'].max() - test['longitude']



# NEW FEATURES FROM Georgii Vyshnia

train['lon_plus_lat'] = train['longitude'] + train['latitude'] 

test['lon_plus_lat'] = test['longitude'] + test['latitude']



lbl = preprocessing.LabelEncoder()

train['air_store_id2'] = lbl.fit_transform(train['air_store_id'])

test['air_store_id2'] = lbl.transform(test['air_store_id'])



test_index = test['id']

#test_index_idx = test.index

train['visitors'] = np.log1p(train['visitors'].values)



##### preprocess steps to feed into the network #################

#################################################################



train['cat'] = 'train'

test['cat'] = 'test'

    

hot_enc_cols_cat = ['air_genre_name0','air_genre_name1','air_genre_name2','air_genre_name3',

                 'air_area_name0','air_area_name1','air_area_name2','air_area_name3',

                 'air_genre_name','air_area_name','day_of_week','dow','year','month']



full_df = pd.concat((train,test), axis=0, ignore_index=False)

    

df_index = full_df.index

    

full_df = pd.get_dummies(full_df, columns=hot_enc_cols_cat)



scale_cols = ['lon_plus_lat','var_max_long','var_max_lat','date_int','total_reserv_dt_diff_mean','total_reserv_mean',

             'total_reserv_sum','rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rs2_y','rv2_y','latitude','longitude',

             'count_observations','max_visitors','median_visitors','min_visitors','holiday_flg','rv1_y',

              'mean_visitors','air_store_id2','date_int','var_max_long']



full_df = full_df.fillna(0)



from scipy.special import erfinv



def rank_gauss(x):

    # x is numpy vector

    N = x.shape[0]

    temp = x.argsort()

    rank_x = temp.argsort() / N

    rank_x -= rank_x.mean()

    rank_x *= 2 # rank_x.max(), rank_x.min() should be in (-1, 1)

    efi_x = erfinv(rank_x) # np.sqrt(2)*erfinv(rank_x)

    efi_x -= efi_x.mean()

    return efi_x



for coln in scale_cols:

    full_df[coln] = rank_gauss(np.array(full_df[coln]))



full_df.index = df_index    

        

train = full_df[full_df['cat']=='train']

test = full_df[full_df['cat']=='test']    

    

drop_cols = ['cat','id', 'air_store_id', 'visit_date','visitors']



targets = train['visitors']

train = train.drop(train[drop_cols],axis=1)

test = test.drop(test[drop_cols],axis=1)



print('Pre-processing done!')



print('train',train.shape)

print('test',test.shape)

print(targets.shape)
from sklearn.model_selection import train_test_split

train, valid, y_train, y_valid = train_test_split(train, targets, test_size=0.3, random_state=2018)



print(train.shape)

print(valid.shape)

print(y_train.shape)

print(y_valid.shape)

print('test shape',test.shape)

print('\n-------------------\n')
import torch 

import torch.nn as nn

import torchvision.transforms as transforms

from torch.autograd import Variable

import torch.nn.functional as F



# Hyper-Parameters



EPOCH = 20

BATCH_SIZE = 200

L_R = 0.001

CUDA = torch.cuda.is_available() # if not run, set here to False manually
# expand the train and valid data to fit the expected Tensor



train_np = np.expand_dims(train.values.astype(np.float32), axis=1)

print(train_np.shape)

valid_np = np.expand_dims(valid.values.astype(np.float32), axis=1)

print(valid_np.shape)



train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_np),

                                               torch.from_numpy(y_train.values))



train_loader = torch.utils.data.DataLoader(dataset=train_dataset,

                                           batch_size = BATCH_SIZE,

                                           shuffle = True)



test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(valid_np),

                                               torch.from_numpy(y_valid.values))



test_loader = torch.utils.data.DataLoader(dataset=test_dataset,

                                           batch_size =500,

                                           shuffle = False)



print(type(train_dataset),type(test_dataset))



# every batch: BATCH_SIZE * 274(featrues) data and BATCH_SIZE*1 labels
# a toy cnn, works worse than fnn below....



class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv1d(1, 16, kernel_size=18),

            nn.BatchNorm1d(16),

            nn.ReLU(),

            nn.Dropout(.2),

            nn.MaxPool1d(2))

        self.layer2 = nn.Sequential(

            nn.Conv1d(16, 32, kernel_size=8),

            nn.BatchNorm1d(32),           

            nn.ReLU(),

            nn.Dropout(.2),

            nn.MaxPool1d(2))

        self.fc = nn.Linear(32*60, 32*10)

        self.fc2 = nn.Linear(32*10, 1)



#input size (BATCH_SIZE * 1 *274)--(conv1d(1, 16, 18))-->(BATCH_SIZE * 16 * 274-18)--(maxpool1d(2))-->(BATCH_SIZE * 16 * (274-18)/2=128)



       

        

    def forward(self, x):

        #print(x.shape)

        out = self.layer1(x)

        #print(out.shape)

        out = self.layer2(out)

        #print(out.shape)

        out = out.view(-1, 32*60)

        out = self.fc(out)

        out = self.fc2(out)

        #print(out.shape)

        return out

class Net(nn.Module):

    def __init__(self):

        super(Net, self).__init__()

        self.layer = nn.Sequential(

            nn.Linear(274, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(.3),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(.2),

            nn.Linear(64, 32),

            nn.ReLU(inplace=True),

            nn.Dropout(.2),

            nn.Linear(32, 1)

        )           

        

        

    def forward(self, x):

        return self.layer(x)

#model = CNN()

model = Net()



if CUDA:

    model.cuda()



# -----------------------------empty the cache of GPU-----------------------------

torch.cuda.empty_cache()

# Optimizer

#criterion = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr=L_R)
def train_step(epoch):

    model.train()

    for batch_idx, (x, y) in enumerate(train_loader):        

        if CUDA:

            x, y = x.cuda(), y.cuda()

        x, y = Variable(x), Variable(y).float()

        #print(x.shape)

        optimizer.zero_grad()

        outputs = model(x)

        loss = F.mse_loss(outputs, y)

        loss.backward()

        optimizer.step()

        

        if batch_idx % 256 == 0:

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

                epoch, batch_idx * len(x), len(train_loader.dataset),

                100. * batch_idx / len(train_loader), loss.data[0]))



            

# ToDo: plot the result of test_loss



def test_step():

    model.eval()

    test_loss = 0

    correct = 0

    for x, y in test_loader:

        if CUDA:

            x, y = x.cuda(), y.cuda()

        x, y = Variable(x, volatile=True), Variable(y).float()

        outputs = model(x)

        

        # Problem F.mse_loss use size_average=True so the value is wrong in early version

        

        test_loss += F.mse_loss(outputs, y, size_average=False).data[0] # sum up the batch loss 

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))

    loss_p.append(test_loss)
# run training



loss_p = []



for epoch in range(EPOCH):

    train_step(epoch)

    test_step()
# result loader

res_np = np.expand_dims(test.values.astype(np.float32), axis=1)



res_tensor = torch.from_numpy(res_np)



res_loader = torch.utils.data.DataLoader(dataset=res_tensor,

                                           batch_size =BATCH_SIZE,

                                           shuffle = False)
results = []



for x in res_loader:

    x = Variable(x).cuda()

    outputs = model(x)

    results.append(outputs.cpu().data.numpy())



# turn results into np.array and flattend them then get the real predict with np.exp

results_n = np.array(results)

flattened = np.exp(np.array([val for sublist in results_n for val in sublist]).reshape(-1))



# transform it into DataFrame to save of visualize

nn_df = pd.DataFrame(flattened,columns=['visitors'],index=test_index)

print(nn_df.head())

nn_df.to_csv('submit_nn.csv')