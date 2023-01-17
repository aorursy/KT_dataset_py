# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train_set = pd.read_csv("../input/train.csv")

train_set.head()
train_set.describe()
train_set.columns
plt.hist(train_set['SalePrice'], bins=100)
#trying the same with seaborn

sns.distplot(train_set['SalePrice'], hist_kws={'alpha': 0.9}, kde=False)
corr = train_set.select_dtypes(include = ['float64', 'int64']).iloc[:, 1:].corr()

plt.figure(figsize=(12, 12))

sns.heatmap(corr, vmax=1, square=True)
corr.columns
abs(corr['SalePrice']).sort_values(ascending=False)
plt.plot(train_set['SalePrice'],train_set['OverallQual'], 'bo')
plt.plot(train_set['SalePrice'],train_set['GrLivArea'], 'bo')
corr['GarageArea']['GarageCars']
plt.plot(train_set['SalePrice'],train_set['GarageArea'], 'bo')
plt.plot(train_set['SalePrice'],train_set['TotalBsmtSF'], 'bo')
#First quick and dirty linear regression
#Data set has been loaded but it need to be separated in 2 parts: 

    #- trainning set ~70%

    #- Cross validation ~30%

#Combination of this two data set will enable us to know if we are dealing with a bias or variance issue 

#Sometimes the data set is divided in 3 parts including the test set but here testy set is given in another csv file. 
nb_trainning_ex = int(train_set['Id'].count()*0.7)

nb_cv_ex = train_set['Id'].count() - nb_trainning_ex

print(nb_trainning_ex, nb_cv_ex, (nb_cv_ex + nb_trainning_ex)==train_set['Id'].count())
train_set_2 = train_set[1:nb_trainning_ex+1]

cv_set_2 = train_set[nb_trainning_ex:]
X_train = train_set_2.ix[:,['OverallQual', 'GarageArea', 'TotalBsmtSF','GrLivArea', '1stFlrSF']]

y_train = train_set_2['SalePrice']
#Feature scaling

X_train = (X_train - X_train.mean())/X_train.std()

y_train = (y_train - y_train.mean())/y_train.std()
X_cv = cv_set_2.ix[:,['OverallQual', 'GarageArea', 'TotalBsmtSF','GrLivArea', '1stFlrSF']]

y_cv = cv_set_2['SalePrice']
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 

clf = lr.fit(X_train, y_train) 
clf.score(X_cv, y_cv)
plt.plot(train_set['GarageCars'],train_set['GarageArea'], 'bo')
sns.boxplot(x="YearBuilt", y="SalePrice", data=train_set)