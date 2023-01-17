# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#By Roman Vardugin
import seaborn as sns
import matplotlib.pyplot as plt

Data_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
Data_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
Submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
Data_train.head()
Data_test.head()
Submission.head()
Data_train.isnull().sum()
Data_train.info()
Data_test.info()
Cor_heat = Data_train.corr()
plt.figure(figsize=(16,16))
sns.heatmap(Cor_heat, vmax=0.9, square=True)
Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_train = Data_train.drop(Columns, axis=1)
Data_train.head()

Data_train = pd.get_dummies(Data_train)
Data_train = Data_train.fillna(method='ffill')
Data_train.head()
Columns = ['Alley','FireplaceQu','PoolQC','Fence','MiscFeature']

Data_test = Data_test.drop(Columns, axis=1)
Data_test.head()
Data_test.shape
Data_train.SalePrice.shape
Data_train = Data_train.drop([1460])
Data_train.shape

Data_test = pd.get_dummies(Data_test)
Data_test = Data_test.fillna(method='ffill')
Data_test
TrainX = Data_train.drop('SalePrice', axis=1)
TrainY = Data_train.SalePrice
trainX = np.asarray(TrainX)
trainY = np.asarray(TrainY)
  #from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(Data_train.drop('SalePrice',axis=1), Data_train.SalePrice,test_size = .3, random_state=0)
#from sklearn import linear_model 
#linear = linear_model.LinearRegression()
#linear.fit(x_train, y_train)
#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)
#print('R² Value: \n', linear.score(x_train, y_train))
#predicted = linear.predict(x_test)
#print(predicted)
Data_train.SalePrice
n_columns = Data_train.drop(columns=Data_test.columns)
n_columns
X = Data_train.drop(columns=n_columns.columns)
X
from sklearn import linear_model 
linear = linear_model.LinearRegression()
linear.fit(X, trainY)
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
print('R² Value: \n', linear.score(X, trainY))

predicted = linear.predict(Data_test)
print(predicted)
test_id = Submission['Id']
output = pd.DataFrame({"Id": test_id, "SalePrice": predicted})
output.to_csv("my_submission.csv", index=False)
output