import os

import numpy as np 

import pandas as pd

from sklearn.preprocessing import OneHotEncoder as ohe

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse

import time

import joblib



from fastFM.mcmc import FMRegression as FMRmcmc

from fastFM.als import FMRegression as FMRals

from fastFM.sgd import FMRegression as FMRsgd
ds_dir = '../input/pda2019'
train = pd.read_csv(os.path.join(ds_dir,"train-PDA2019.csv"))

test = pd.read_csv(os.path.join(ds_dir,"test-PDA2019.csv"))

train.drop('timeStamp',inplace=True,axis=1)
itemIDs = train.itemID.unique()
userIDs = train.userID.unique()
watchedList = train.groupby('userID')['itemID'].apply(list)
train.head(5)
test.head(5)
y_train = train[['rating']].values.reshape(-1)

x_train = train.drop('rating',axis=1)
x_test = test.copy()

x_test.drop('recommended_itemIDs',axis=1,inplace=True)

x_test = list(x_test.userID.values)* len(itemIDs)

x_test = pd.DataFrame(x_test, columns=['userID'])
dupeItem = np.repeat(itemIDs, len(test))

dupeItem = dupeItem.tolist()

x_test['itemID'] = dupeItem
def get_top10(row):

    if row.userID in userIDs:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = pred[~pred.itemID.isin(watchedList[row.userID])]

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred

    else:

        pred = grouped.get_group(row.userID).sort_values(by=['prediction'],ascending=False)

        pred = ' '.join(map(str, pred.head(10).itemID.values))

        return pred
encoder = ohe(handle_unknown='ignore').fit(x_train)

x_train_e = encoder.transform(x_train)

x_test_e = encoder.transform(x_test)
model = FMRmcmc(n_iter=120, init_stdev=0.2, rank=14)

t_start = time.time()

prediction = model.fit_predict(x_train_e, y_train, x_test_e)

fit_dur = pred_dur = time.time()-t_start



x_test['prediction'] = prediction

grouped = x_test.groupby(x_test.userID)

test['recommended_itemIDs'] = test.apply(get_top10, axis=1)

test.to_csv('sub_pda_fastFM_MCMC.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')
model = FMRals()

t_start = time.time()

model.fit(x_train_e, y_train)

fit_dur = time.time()-t_start



t_start = time.time()

prediction = model.predict(x_test_e)

pred_dur = time.time()-t_start





x_test['prediction'] = prediction

grouped = x_test.groupby(x_test.userID)

test['recommended_itemIDs'] = test.apply(get_top10, axis=1)

test.to_csv('sub_pda_fastFM_ALS.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')
model = FMRsgd(n_iter=26000000, init_stdev=0.2, l2_reg_V=0.1, l2_reg_w=0.001, rank=12, step_size=0.005)

t_start = time.time()

model.fit(x_train_e, y_train)

fit_dur = time.time()-t_start



t_start = time.time()

prediction = model.predict(x_test_e)

pred_dur = time.time()-t_start



x_test['prediction'] = prediction

grouped = x_test.groupby(x_test.userID)

test['recommended_itemIDs'] = test.apply(get_top10, axis=1)

test.to_csv('sub_pda_fastFM_SGD.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')