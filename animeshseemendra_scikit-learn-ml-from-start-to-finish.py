import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#reading data
data_train = pd.read_csv('../input/train.csv')
data_test = pd.read_csv('../input/test.csv')
#data print
data_train.describe()

data_train.head(10)

data_train.isnull().sum()
from sklearn.linear_model import LinearRegression
data_isnull=data_train[data_train['Age'].isnull()]
data_nullage=data_isnull['Age']
data_isnull=data_isnull.drop(['Age'], axis = 1, inplace = True)
clf=LinearRegression()
clf.fit(data_isnull,data_nullage)
plt.scatter(data_isnull, data_nullage,  color='black')
plt.plot(data_isnull, data_nullage, color='blue', linewidth=3)
