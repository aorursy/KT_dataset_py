# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# set the working directory

os.chdir("../input")
# get the data

df=pd.read_csv("birthwt.csv")
df.head(10)

df.shape
# divide the data set in two parts : train and test

train,test=train_test_split(df,test_size=0.2)
train.shape, test.shape
#train the model

fit_DT=DecisionTreeRegressor(max_depth=2).fit(train.iloc[:,0:9],train.iloc[:,9])

fit_DT
# apply model on the test data

predictions_DT=fit_DT.predict(test.iloc[:,0:9])

predictions_DT
#model evaluation

#calculate the MAPE 

def MAPE(y_true,y_pred):

    mape=np.mean(np.abs((y_true-y_pred)/y_true))*100

    return mape
MAPE(test.iloc[:,9],predictions_DT)
# import required library and module

import statsmodels.api as sm

# train the model using training dataset

model= sm.OLS(train.iloc[:,9],train.iloc[:,0:9]).fit()
# get the summary of the model

model.summary()
# make predictions using the above models

predictions_LR=model.predict(test.iloc[:,0:9])
# model evaluation

MAPE(test.iloc[:,9],predictions_LR)