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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

price_pre = pd.read_csv('../input/train.csv')

k=10

corrmatrix = price_pre.corr()


cols = corrmatrix.nlargest(k,'SalePrice')['SalePrice'].index

cm1 = price_pre[cols].corr()

#hm2 = sns.heatmap(cm1,square=True,annot=True,cmap='RdPu',fmt='.2f',annot_kws={'size':10})


cols1 = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

total = price_pre.isnull().sum().sort_values(ascending=False)
percent = (price_pre.isnull().sum()/price_pre.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent],axis=1,keys = ['Total','Percent'])

data1 = price_pre.drop(missing_data[missing_data['Total']>1].index,axis=1)

data2 = data1.drop(data1.loc[data1['Electrical'].isnull()].index)

data2.isnull().sum().max()

feature_data = data2.drop(['SalePrice'],axis=1)
target_data = data2['SalePrice']
X_train = feature_data[['OverallQual', 'GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']]
val_model = DecisionTreeRegressor()
train_X, val_X, train_y, val_y = train_test_split(X_train, target_data, random_state = 100)
val_model.fit(train_X, train_y)

test_data = pd.read_csv('../input/test.csv')
test_input = test_data[['OverallQual', 'GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']]

#val_predictions = val_model.predict(test_input)
#mean_absolute_error(val_y, val_predictions)


test_input[test_input.isnull().values==True]
X_in = test_input.fillna(0)
val_predictions = val_model.predict(X_in)
submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': val_predictions})
submission.to_csv('submission.csv', index=False)
