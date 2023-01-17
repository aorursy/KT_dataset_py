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
import matplotlib.pyplot as plt

#load the data set
df_train=pd.read_csv('/kaggle/input/house_train.csv',index_col='Id')
df_train.head()
df_test=pd.read_csv('/kaggle/input/house_test.csv',index_col='Id')
df_test.head()
#checking the missing values in train data set
df_train.isnull().sum()[df_train.isnull().any()]
#cheking the missing values in test data set

df_test.isnull().sum()[df_test.isnull().any()]
#this is not necessary, just for some statistics of data
df_train.describe()
df_test.describe()
#name of all columns
df_train.columns
#this is also not necessary, just little bit visualization
df_train['BedroomAbvGr'].value_counts().plot(kind='bar')
#here we will plot the relationship between number of beds and price
bedroom_saleprice=df_train.head(20)
plt.scatter(bedroom_saleprice['BedroomAbvGr'],bedroom_saleprice['SalePrice'])
plt.title('The relationship between number of bedrooms and price',color='green')
plt.xlabel('Number of rooms',color='green',size=15)
plt.ylabel('Price',color='green',size=15)
#lets drop the target column
y = df_train.SalePrice
df_train.drop(['SalePrice'], axis=1, inplace=True)
#we will train our model using the following columns
features=['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
x_train=df_train[features].copy()
x_test=df_test[features].copy()
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
x_train,x_test,y_train,y_test=train_test_split(x_train,y,train_size=0.8,test_size=0.2,random_state=0)
x_train.head()
x_test.head()
#Predictive model 1, model 2, model 3, model 4, model 5: 
#Random Forest Regressor using differenet number of estimator and other feaures

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
# Function for comparing different models
def score_model(model, X_t=x_train, X_v=x_test, y_t=y_train, y_v=y_test):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d RandomForestRegressor : %d" % (i+1, mae))
#Predictive model 6: AdaBoost Regressor

from sklearn.ensemble import AdaBoostRegressor
model_6=AdaBoostRegressor(n_estimators=3000, learning_rate=0.01)
model_6.fit(x_train,y_train)
preds_6=model_6.predict(x_test)
print("Model 6 AdaBoost Regressor: " + str(mean_absolute_error(y_test, preds_6)))
#the best model here is model 3
my_model = model_3

# Fit the model to the training data
my_model.fit(x_train, y_train)

# Generate test predictions
preds_test = my_model.predict(x_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': x_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
