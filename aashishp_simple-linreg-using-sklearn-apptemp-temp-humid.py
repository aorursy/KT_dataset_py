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
# some imports

import seaborn as sns

from matplotlib import pyplot as plt

from scipy import stats

from sklearn import metrics

# Reading only required columns

columns_to_read = ['Temperature (C)','Apparent Temperature (C)','Humidity']

df = pd.read_csv('../input/weatherHistory.csv',usecols = columns_to_read)
df.sample(10)
# Lets check missing values

df.isnull().sum()
# Lets check distributions

fig,ax = plt.subplots(ncols=3,figsize = [20,8])

plt.subplots_adjust(hspace=0.3,wspace=0.3)

sns.distplot(df['Temperature (C)'],ax= ax[0],fit = stats.norm)

sns.distplot(df['Apparent Temperature (C)'],ax= ax[1],fit = stats.norm)

sns.distplot(df['Humidity'], ax = ax[2],fit = stats.norm)
df['Humidity'] = np.log1p(df['Humidity'])**3

sns.distplot((df['Humidity']),fit = stats.norm)
mask = np.ones(df.shape[1])

mask = np.triu(mask,k =1)

#print(mask)



heatmap = sns.heatmap(np.transpose(df.corr()),annot=True,mask=mask,yticklabels = True,xticklabels=True);

#heatmap.set_yticklabels(rotation = 90)
y_original = df['Apparent Temperature (C)']

X_original = df.drop('Apparent Temperature (C)',axis = 1)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split



X,X_test,y,y_test = train_test_split(X_original,y_original,test_size = 0.33,random_state = 42)



X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size = 0.33,random_state = 42)
# single linear reg 

linreg = LinearRegression()

model_1 = linreg.fit(X_train,y_train)



pred_valids = model_1.predict(X_valid)



print("R2 score on validation set is {:.4}".format(metrics.r2_score(y_valid,pred_valids)))
pred_test = model_1.predict(X_test)

print("R2 score on test set is {:.4}".format(metrics.r2_score(y_test,pred_test)))