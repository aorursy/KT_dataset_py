# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import train_test_split

import statsmodels.api as sm



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path = "../input/indian-liver-patient-records/indian_liver_patient.csv"

data = pd.read_csv(path)
data.head()
data.info()
data.describe(include = 'all')
sns.boxplot(x = 'Total_Bilirubin' , y = 'Direct_Bilirubin' , data=data)
sns.pairplot(data)
data.corr()
data.isnull().sum()
dummy = pd.get_dummies(data['Gender'])

dummy.head()
data = pd.concat([data , dummy] , axis=1)

data.head()
data.drop(['Gender'] , axis=1 , inplace=True)
# split the dataset into train and test

# --------------------------------------

train, test = train_test_split(data, test_size = 0.3)

print(train.shape)

print(test.shape)

# split the train and test into X and Y variables

# ------------------------------------------------

train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]

test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]

print(train_x)

print(test_x)
train_x.shape
train_y.shape
test_x.shape
test_y.shape
train_x.head()
train_y.head()
train.head()
train.tail()
train.dtypes
lm1 = sm.OLS(train_y, train_x).fit()

pdct1 = lm1.predict(test_x)

print(pdct1)
actual = list(test_y.head(5))

type(actual)

predicted = np.round(np.array(list(pdct1.head(5))),2)

print(predicted)

type(predicted)

df_results = pd.DataFrame({'actual':actual, 'predicted':predicted})

print(df_results)
from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  

print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  