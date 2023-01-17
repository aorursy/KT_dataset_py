# import libraries

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from scipy import stats

import warnings
warnings.filterwarnings("ignore")
# load the data

data = pd.read_csv("../input/avocado-prices/avocado.csv")
data.shape
data.head(3)
data.isnull().sum()/len(data.index)
data.describe()
data.info()
# lets observe the distinct values in categorical cols

data.type.value_counts()
# lets drop unnamed col and date

data = data.drop(['Unnamed: 0','Date'],axis=1)
# lets convert categorical variables into 0 and 1 

data['type'] =  data['type'].apply(lambda x: 1 if x == 'conventional' else 0)
labels1,levels1 = pd.factorize(data['region'])
data['region_num'] = pd.DataFrame(labels1)
data = data.drop(['region'],axis=1)
data.head(3)
data.columns
X = data.loc[:,['Total Volume', '4046', '4225', '4770', 'Total Bags',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year',
       'region_num']]
y = data.AveragePrice
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.85,random_state=100)
model_lr01 = LinearRegression()
model_lr01.fit(X_train,y_train)
y_predict = model_lr01.predict(X_test)
print("r2 score of the model is {}".format(r2_score(y_true = y_test, y_pred = y_predict)))
# correlation

data.corr()
X = data.loc[:,['4046', '4225', '4770', 'Total Bags',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year',
       'region_num']]
y = data.AveragePrice

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.85,random_state=100)

model_lr02 = LinearRegression()
model_lr02.fit(X_train,y_train)

y_predict = model_lr02.predict(X_test)
print("r2 score of the model is {}".format(r2_score(y_true = y_test, y_pred = y_predict)))
X = data.loc[:,['type', 'year',
       'region_num']]
y = data.AveragePrice

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.85,random_state=100)

model_lr03 = LinearRegression()
model_lr03.fit(X_train,y_train)
y_predict = model_lr03.predict(X_test)
print("r2 score of the model is {}".format(r2_score(y_true = y_test, y_pred = y_predict)))
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
data.columns
X = data.loc[:,['Total Volume', '4046', '4225', '4770', 'Total Bags',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'type', 'year',
       'region_num']]

y = data.AveragePrice

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.85,random_state=100)

model_s1 = sm.OLS(y_train,X_train).fit()
model_s1.summary()
vif = pd.DataFrame()

vif['Features'] =X_train.columns

vif['VIF'] = [variance_inflation_factor(X_train.values,i)for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif=vif.sort_values(by='VIF',ascending=False)
vif
y_predict = model_s1.predict(X_train)
y_train.head(2)
y_predict.head(2)


