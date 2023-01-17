# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
path = "../input/novel-corona-virus-2019-dataset/covid_19_data.csv"
df = pd.read_csv(path)
df.head()
df.describe
df.columns
df.dtypes
df.tail()
df.columns
df.shape
#summarize the dataset
desc = df.describe()
desc
#find null values
df.isnull().sum()
df.isna().sum()
df.corr()
df.corr()
cor = df.iloc[:,0:7].corr()
print(cor)
df.corr()
cor = df.iloc[:,0:12].corr()
print(cor)
# correlation using visualization
sns.heatmap(cor, xticklabels=cor.columns, yticklabels=cor.columns)

df.dtypes
# Drop the 'Province_State' and ' County_Region' and  'Date' columns
df.drop(['ObservationDate','Province/State' , 'Country/Region' , 'Last Update'], axis='columns', inplace=True)

# Examine the shape of the DataFrame (again)
print(df.shape)
# split the dataset into train and test
# --------------------------------------
train, test = train_test_split(df, test_size = 0.3)
print(train.shape)
print(test.shape)

# split the train and test into X and Y variables
# ------------------------------------------------
train_x = train.iloc[:,0:3]; train_y = train.iloc[:,3]
test_x  = test.iloc[:,0:3];  test_y = test.iloc[:,3]
print(train_x)
print(test_x)


print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)

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

#To Check the Accuracy:
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(test_y, pdct1))  
print('Mean Squared Error:', metrics.mean_squared_error(test_y, pdct1))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_y, pdct1)))  
