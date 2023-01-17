# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

sample_submission = pd.read_csv("../input/allstate-claims-severity/sample_submission.csv")

test_data = pd.read_csv("../input/allstate-claims-severity/test.csv")

train_data = pd.read_csv("../input/allstate-claims-severity/train.csv")
print("Train data dimensions: ", train_data.shape)

print("Test data dimensions: ", test_data.shape)
#Exploring the train_data

train_data.head()
print('Number of missing values', train_data.isnull().sum().sum())
#Exploring the data stastically 

train_data.describe()
#Exploring the columns of data

train_data.columns
test_data.columns
train_data.info()
# Counting the Feature of train_data

cont_Featureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    #print(x)

    if(not str(x).isalpha()):

        cont_Featureslist.append(colName)
print(cont_Featureslist)
cont_Featureslist.remove('id')
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
# To find correlation between features and target feature

correlationMatrix = train_data[cont_Featureslist].corr().abs()



plt.subplots(figsize=(15, 10))

sns.heatmap(correlationMatrix,annot=True)



# Mask unimportant features

sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)

plt.show()
plt.figure(figsize=(15,10))

sns.distplot(train_data["loss"])

sns.boxplot(train_data["loss"])
plt.figure(figsize=(12,8))

sns.distplot(np.log1p(train_data["loss"]))
catCount = sum(str(x).isalpha() for x in train_data.iloc[1,:])

print("Number of categories: ",catCount)
cat_Featureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        cat_Featureslist.append(colName)
print(train_data[cat_Featureslist].apply(pd.Series.nunique))
from sklearn.preprocessing import LabelEncoder

for cf1 in cat_Featureslist:

    le = LabelEncoder()

    le.fit(train_data[cf1].unique())

    train_data[cf1] = le.transform(train_data[cf1])
train_data.head(5)
featureslist = []

for colName,x in train_data.iloc[1,:].iteritems():

    #print(x)

    if(not str(x).isalpha() or str(x).isalpha):

        featureslist.append(colName)
featureslist.remove('id')
# Import nessery models

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error as MSE

from sklearn.model_selection import train_test_split



X = train_data[featureslist]

y = train_data.loss



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

dtr_model = DecisionTreeRegressor()

dtr_model.fit(X_train, y_train)

y_pred = dtr_model.predict(X_test)

val_mae = mean_absolute_error(y_pred, y_test)

mse_test = MSE(y_test, y_pred)

rmse_test = mse_test**(1/2)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

val_mae = mean_absolute_error(y_pred, y_test)

mse_test = MSE(y_test, y_pred)

rmse_test = mse_test**(1/2)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))

print('Test set RMSE of sgbr: {:.3f}'.format(rmse_test))