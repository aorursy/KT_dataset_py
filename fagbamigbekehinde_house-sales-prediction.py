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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

plt.style.use(style = 'ggplot')

%matplotlib inline
training_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
training_data.head()
test_data.head()
plt.hist(training_data.SalePrice, color='green')

plt.show()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

plt.style.use(style = 'ggplot')

%matplotlib inline
target = np.log(training_data.SalePrice)

plt.hist(target, color = 'yellow')
numeric_features = training_data.select_dtypes(include= [np.number])
numeric_features.head()
categorical_features = training_data.select_dtypes(exclude= [np.number])
categorical_features.head()
correlation = numeric_features.corr()
correlation.head()
print (correlation['SalePrice'].sort_values(ascending = False)[:5],  '\n')
print (correlation['SalePrice'].sort_values(ascending = False)[-5:],  '\n')
quality_pivot = training_data.pivot_table(index = 'OverallQual', values = 'SalePrice', aggfunc = np.median )
plt.scatter(x = training_data['GrLivArea'], y = np.log(training_data['SalePrice']))

plt.xlabel('Ground Living Area')

plt.ylabel('Sales Price')
plt.scatter(x = training_data['GarageArea'], y = np.log(training_data['SalePrice']))

plt.xlabel('Garage Area')

plt.ylabel('Sales Price')
training_data = training_data[training_data['GarageArea'] <1200 ]

plt.scatter(x = training_data['GarageArea'], y = np.log(training_data['SalePrice']))

plt.xlabel('Garage Area')

plt.ylabel('Sales Price')
training_data = training_data[training_data['GarageArea'] <1200 ]

plt.scatter(x = training_data['GarageArea'], y = np.log(training_data['SalePrice']))

plt.xlim(-200,1600)

plt.xlabel('Garage Area')

plt.ylabel('Sales Price')
nulls = pd.DataFrame(training_data.isnull())

nulls
nulls = pd.DataFrame(training_data.isnull().sum().sort_values(ascending = False ))

nulls
nulls = pd.DataFrame(training_data.isnull().sum().sort_values(ascending = False ))

nulls.columns = ['Null Counts']

nulls.index.name = 'Features'

nulls
categorical_features.head()
training_data['enc_Street'] = pd.get_dummies(training_data.Street, drop_first= True)

training_data.head()
test_data['enc_Street'] = pd.get_dummies(test_data.Street, drop_first= True)

test_data.head()
data = training_data.select_dtypes(include = [np.number]).interpolate().dropna()

data.head()
X = data.drop(['SalePrice', 'Id'], axis = 1)

y = np.log(training_data.SalePrice)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , random_state = 42, test_size = 0.3)
from sklearn import linear_model
lr = linear_model.LinearRegression()
lr= lr.fit(X_train, y_train)
lr.score(X_test, y_test)
y_pred = lr.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, y_pred)
actual_value = y_test
plt.scatter(x = y_pred, y = actual_value )

plt.xlabel('Predicted Value')

plt.ylabel('Actual Value')

plt.title('Linear Regression Model')
data_for_test = pd.DataFrame()

data_for_test['Id'] = test_data.Id

feats = test_data.select_dtypes(include = [np.number]).drop(['Id'], axis = 1).interpolate()
predictions =  lr.predict(feats)
final_predictions = np.exp(predictions)
data_for_test['SalePrice'] = final_predictions

data_for_test.head()
data_for_test.to_csv ('Submission.csv', index = False )