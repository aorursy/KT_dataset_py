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
ds_dir = '../input/goodreads-ucsds-files'
train = pd.read_csv(os.path.join(ds_dir,"train_Interactions.csv"))

test = pd.read_table(os.path.join(ds_dir,"pairs_Rating.txt"))

test.rename(columns={'userID-bookID,prediction':'userID-bookID'},inplace=True)
train.head(5)
test.head(5)
train.loc[(train.rating == 0),'rating']=1e-9
y_train = train[['rating']].values.reshape(-1)

x_train = train.drop('rating',axis=1)
testset = test.copy()

testset[['userID','bookID']] = testset['userID-bookID'].str.split('-',expand=True)

testset.drop('userID-bookID',inplace=True,axis=1)
encoder = ohe(handle_unknown='ignore').fit(x_train)

x_train = encoder.transform(x_train)

testset = encoder.transform(testset)
model = FMRmcmc(n_iter=120, init_stdev=0.2, rank=14)

t_start = time.time()

prediction = model.fit_predict(x_train, y_train, testset)

fit_dur = pred_dur = time.time()-t_start



test['prediction'] = prediction

test.to_csv('sub_ucsd_fastFM_MCMC.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')

print('Last Submission')

print('Private :',1.10931, ' Rank 232')

print('Public :',1.15162)
model = FMRals()

t_start = time.time()

model.fit(x_train, y_train)

fit_dur = time.time()-t_start



t_start = time.time()

prediction = model.predict(testset)

pred_dur = time.time()-t_start



test['prediction'] = prediction

test.to_csv('sub_ucsd_fastFM_ALS.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')

print('Last Submission')

print('Private :',3.73618, ' Rank 422')

print('Public :',3.78752)
model = FMRsgd(n_iter=26000000, init_stdev=0.2, l2_reg_V=0.1, l2_reg_w=0.001, rank=12, step_size=0.005)

t_start = time.time()

model.fit(x_train, y_train)

fit_dur = time.time()-t_start



t_start = time.time()

prediction = model.predict(testset)

pred_dur = time.time()-t_start



test['prediction'] = prediction

test.to_csv('sub_ucsd_fastFM_SGD.csv',index=False)



print('Training duration : ', fit_dur,'s')

print('Predict duration : ', pred_dur,'s')

print('Last Submission')

print('Private :',1.27958, ' Rank 419')

print('Public :',1.26406)