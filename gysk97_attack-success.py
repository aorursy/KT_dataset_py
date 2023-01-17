import pandas as pd
import numpy as np

import torch
import torchvision.datasets as data
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt
import seaborn as sns

train_csv = pd.read_csv('train.csv')
test_csv = pd.read_csv('test.csv')
plt.figure(figsize=(12, 12))
sns.heatmap(train_csv.corr().loc[:'alcohol', 'quality':])
import copy
train_D = train_csv.drop(columns = ['index','quality','density'], axis = 1)
train_L = copy.deepcopy(train_csv.quality)
test_D = test_csv.drop(columns =  ['index','density'], axis = 1)

train_D.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler
mscaler = MinMaxScaler()
scaler = StandardScaler()
train_D = scaler.fit_transform(np.array(train_D))
test_D = scaler.transform(np.array(test_D))
train_D = torch.FloatTensor(train_D)
train_L = torch.LongTensor(train_L)
test_D = torch.FloatTensor(test_D)
from sklearn.ensemble import RandomForestClassifier


RF = RandomForestClassifier()
RF
from sklearn.model_selection import GridSearchCV
params = {'max_depth':[None, 3,4,5,6], 'min_samples_leaf':[1,2,3],'n_estimators' :[100,200,500,1000]}
cv= GridSearchCV(RF, params)
cv.fit(train_D, train_L)
cv.best_params_
RF.fit(train_D, train_L)
pred = cv.predict(test_D)
pred
sample = pd.read_csv('sample_submission.csv')
sample['quality'] = pred
sample
sample.to_csv('happyRF.csv', index = False)