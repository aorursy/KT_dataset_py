# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from sklearn.model_selection import train_test_split

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "../input/dont-overfit-ii/train.csv"
data = pd.read_csv(path)
data.head()
data.describe()
data.count()
data.corr()

cor = data.iloc[:,0:5].corr()

print(cor)

train, test = train_test_split(data, test_size = 0.3)

print(train.shape)

print(test.shape)

train_x = train.iloc[:,0:120]; train_y = train.iloc[:,120]

test_x  = test.iloc[:,0:120];  test_y = test.iloc[:,120]
print(train_x.shape)

print(train_y.shape)

print(test_x.shape)

print(test_y.shape)

train_x.head()

train_y.head()

train_y.head(20)
train.head()
train.dtypes
lm1 = sm.OLS(train_y, train_x).fit()

pdct1 = lm1.predict(test_x)

print(pdct1)
actual = list(test_y.head(50))

type(actual)
predicted = np.round(np.array(list(pdct1.head(50))),2)

print(predicted)

type(predicted)
df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

print(df_results.head(115))
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1))) 
def RMSE(predict, target):

    return np.sqrt(((predict - target) ** 2).mean())

print ('My RMSE: ' + str(RMSE(test_y,pdct1)) )
def MAPE(predict,target):

    return ( abs((target - predict) / target).mean()) * 100

print ('My MAPE: ' + str(MAPE(test_y,pdct1)) )
import math



#A function to calculate Root Mean Squared Logarithmic Error (RMSLE)

def rmsle(pdct1, test_y):

    assert len(pdct1) == len(test_y)

    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(test_y)]

    return (sum(terms_to_sum) * (1.0/len(pdct1))) ** 0.5

print ('My RMSLE: ' + str(RMSE(test_y,pdct1)) )
from sklearn.metrics import mean_absolute_error

print ('Sk MAE: ' + str(mean_absolute_error(pdct1,test_y)) )

def MAE(predict,target):

    return (abs(predict-target)).mean()

print ('My MAE: ' + str(MAE(test_y,pdct1)))
def R2(predict, target):

    return 1 - (MAE(predict,target) / MAE(target.mean(),target))

def R_SQR(predict, target):

    r2 = R2(predict,target)

    return np.sqrt(r2)

print ('My R2         : ' + str(R2(test_y,pdct1)) )

print ('My R          : ' + str(R_SQR(test_y,pdct1)) )
def R2_ADJ(predict, target, k):

    r2 = R2(predict,target)

    n = len(target)

    return (1 -  ( (1-r2) *  ( (n-1) / (n-(k+1)) ) ) )

k= len(data.columns)

print ('My R2 adjusted: ' + str(R2_ADJ(test_y,pdct1,k)) )