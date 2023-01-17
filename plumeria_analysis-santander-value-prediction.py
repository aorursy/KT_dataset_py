import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold,cross_val_score
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv",usecols=['ID'])
test['ID'].unique()
sample_submission=pd.read_csv("../input/sample_submission.csv")
# separate array into input and output components
array = train.values
X = array[:,1:4992]
Y = array[:,4992]
seed=7
#Check the datatypes we have
m=train.dtypes
Count=Counter(m).most_common()
print(Count)
# Load train dataset
print(train.head(10))
print('The Data in train is : {}'.format(train.shape))
# Load test dataset
print(test.head(5))
print('The Data in test is : {}'.format(test.shape))
train.dropna()
plt.scatter(X[:,0], X[:,1], c='m')
plt.show()
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X)
print(X_train_minmax)
min_max_scaler.scale_
min_max_scaler.min_  
# Calculate the spearman's correlation between two variables
# prepare data
# Select the random column from the train.csv
data1=train.iloc[:,[10]]
data2=train.iloc[:,[12]]
# calculate spearman's correlation
coef,p=spearmanr(data1,data2)
print('Spearmans coefficient : %3f'%coef) 
# interpret the significance
if(p>0.05):
    print('Sample are uncorrelated (fail to reject Ho) p= %3f'%p) 
else:
    print('Sample are correlated (reject Ho) p=%3f'%p) 
kfold=KFold(n_splits=10,random_state=seed)
model=KNeighborsRegressor()
scoring='neg_mean_squared_error'
results=cross_val_score(model,X,Y,cv=kfold,scoring=scoring)