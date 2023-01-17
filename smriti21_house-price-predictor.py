# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor

from sklearn import metrics

from matplotlib import pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_data = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')


train_data.shape
#histogram

sns.distplot(train_data['SalePrice']);

#skewness and kurtosis

print("Skewness: %f" % train_data['SalePrice'].skew())

print("Kurtosis: %f" % train_data['SalePrice'].kurt())
print(train_data.columns)

train_data.head()

train_data.drop('Id',axis=1,inplace=True)
#scatter plot grlivarea/saleprice

var = 'GrLivArea'

data = pd.concat([train_data['SalePrice'], train_data[var]], axis=1)

data.plot.scatter(x=var, y='SalePrice', ylim=(0,1000000));
var = 'TotalBsmtSF'

data = pd.concat([train_data['SalePrice'], train_data[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,600000))
var = 'GarageArea'

data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)

data.plot.scatter(x=var,y='SalePrice',ylim=(0,800000))
plt.figure(figsize=(20,5))

var = 'Neighborhood'

data = pd.concat([train_data['SalePrice'],train_data[var]],axis=1)

ax = sns.barplot(x=var, y="SalePrice", data=data)

#C-Commercial has the lowest sales price
#correlation matrix

corrmat = train_data.corr()

f, ax = plt.subplots(figsize=(20, 10))

sns.heatmap(corrmat, vmax=.8, square=True, linecolor='black')
train_data.drop(['1stFlrSF','GarageCars'],axis=1,inplace=True)
col_with_missing_values = [column for column in train_data.columns if train_data[column].isnull().any()]

print(col_with_missing_values)
for col in col_with_missing_values:

    if train_data[col].dtype == float:

                train_data[col] = train_data[col].fillna(train_data[col].median())

    else:

                train_data[col] = train_data[col].fillna(train_data[col].value_counts().index[0])

            

            
train_data.isnull().sum()
categorical_columns = train_data.select_dtypes(include=['object'])

print(categorical_columns.head(1))

LE = LabelEncoder()

for col in categorical_columns:

    train_data[col] = LE.fit_transform(train_data[col])
train_data.head()
scaler = StandardScaler()

scaled_dataframe = scaler.fit_transform(train_data)

train_data = pd.DataFrame(scaled_dataframe)

print(train_data.head())



   
x = train_data.iloc[:,1:-1]

y = train_data.iloc[:, -1]
x_train , x_test , y_train , y_test = train_test_split(x , y ,test_size = 0.3,random_state = 0)
regressor = SGDRegressor(alpha=0.0002)

regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
df = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})

print(df.head())
print('Root Mean Squared Error:',np.sqrt(metrics.mean_absolute_error(y_test,y_pred)))

print(regressor.score(x_test,y_test))