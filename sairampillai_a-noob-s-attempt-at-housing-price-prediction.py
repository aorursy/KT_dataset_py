%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv")
dataset.describe()

dataset.columns

dataset['SalePrice'].describe()

print(dataset['YearRemodAdd'].describe())
# Plotting prices of house based on OverallQuality

plt.figure()

plt.plot(dataset.groupby(["OverallQual"])["SalePrice"].mean())

print(dataset.groupby(["OverallQual"])["SalePrice"].mean())



plt.figure()

plt.plot(dataset.groupby(["OverallCond"])["SalePrice"].mean())

print(dataset.groupby(["OverallCond"])["SalePrice"].mean())
#scatterplot

sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(dataset[cols], size = 2.5)

plt.show();
plt.scatter(dataset['TotalBsmtSF'],dataset['SalePrice'])

for column in list(dataset):

    if dataset[column].dtype != 'int64':

        pass

    else:

        print(column)

        plt.scatter(dataset[column],dataset['SalePrice'])

        plt.show()

        print()
columnsToDrop = [x for x in list(dataset) if dataset[x].dtype!='int64']

columnsToDrop.append('Id')

#columnsToDrop.append('YearBuilt')

#columnsToDrop.append('YearRemodAdd')

#columnsToDrop.append('MoSold')

#columnsToDrop.append('YrSold')

print(columnsToDrop)



numerical_data  = dataset.drop(columnsToDrop,axis=1)

labels = numerical_data['SalePrice']

numerical_data = numerical_data.drop('SalePrice',axis=1)

numerical_data = numerical_data.fillna(0)

print(list(numerical_data))

#load test data

test_data = pd.read_csv('../input/test.csv')

test_data_id = test_data['Id']

test_data = test_data.drop(columnsToDrop,axis=1)

test_data = test_data.fillna(0)

test_data.head(10)
from sklearn.cross_validation import train_test_split 

x_train,x_test,y_train,y_test = train_test_split(numerical_data,labels,test_size=0.10,random_state=3)
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, mean_squared_log_error



regr = LinearRegression()

regr.fit(x_train,y_train)

answers = regr.predict(x_test)

print(mean_squared_error(y_test,answers))



dtr = DecisionTreeRegressor()

dtr.fit(x_train,y_train)

dtr_answers = dtr.predict(x_test)

print(mean_squared_error(y_test,dtr_answers))



answers = dtr.predict(test_data)

my_submission = pd.DataFrame({'Id':test_data_id,'SalePrice':answers})

my_submission.to_csv('decision_tree_regressor_submission_with_years.csv',index=False)