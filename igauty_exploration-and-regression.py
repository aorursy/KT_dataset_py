# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import datetime

import time

import warnings

warnings.filterwarnings('ignore')

import os

print(os.listdir("../input"))





from scipy.stats import norm

from scipy import stats

from sklearn import preprocessing



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib

matplotlib.style.use('ggplot')

sns.set_style('whitegrid')

%matplotlib inline



#VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor

from patsy import dmatrices



#Modelling

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import r2_score,mean_squared_error

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from lightgbm import LGBMRegressor

import xgboost

data = pd.read_csv('../input/insurance.csv')
data.head()
data.shape
data.info()
data.describe()
data.dtypes.value_counts()
print(data['sex'].value_counts())

ax = sns.countplot(x=data['sex'], data=data)
print(data['children'].value_counts())

ax = sns.countplot(x=data['children'], data=data)
print(data['smoker'].value_counts())

ax = sns.countplot(x=data['smoker'], data=data)
print(data['region'].value_counts())

ax = sns.countplot(x=data['region'], data=data)
data['age_bin'] = pd.cut(data.age,[15,20,25,30,35,40,45,50,55,60,65],labels=[20,25,30,35,40,45,50,55,60,65])

print(data['age_bin'].value_counts())

ax = sns.countplot(x=data['age_bin'], data=data)
data['bmi_bin'] = pd.cut(data.bmi,[10,15,20,25,30,35,40,45,50,55],labels=[15,20,25,30,35,40,45,50,55])

print(data['bmi_bin'].value_counts())

ax = sns.countplot(x=data['bmi_bin'], data=data)
data.head()
data.charges.hist()
plt.figure(figsize=(10,6))

ax = sns.scatterplot(x='bmi',y='charges',data=data,palette='magma',hue='smoker')

ax.set_title('Scatter plot of charges and bmi')



sns.lmplot(x="bmi", y="charges", hue="smoker", data=data, palette = 'magma', size = 8)
sns.catplot(data=data, x='age_bin', y='charges',  height=8, aspect=12/8)
sns.catplot(data=data, x='sex', y='charges',  height=8, aspect=12/8)
sns.catplot(data=data, x='children', y='charges',  height=8, aspect=12/8)
sns.catplot(data=data, x='smoker', y='charges',  height=8, aspect=12/8)
sns.catplot(data=data, x='region', y='charges',  height=8, aspect=12/8)
sns.catplot(data=data, x='bmi_bin', y='charges',  height=8, aspect=12/8)
data.groupby(['age_bin']).agg({'charges':'mean'})
data.groupby(['sex']).agg({'charges':'mean'})
data.groupby(['bmi_bin']).agg({'charges':'mean'})
data.groupby(['children']).agg({'charges':'mean'})
data.groupby(['smoker']).agg({'charges':'mean'})
data.groupby(['region']).agg({'charges':'mean'})
data.groupby(['age_bin', 'bmi_bin']).agg({'charges':'mean'})
data.groupby(['age_bin', 'children']).agg({'charges':'mean'})
data.groupby(['age_bin', 'smoker']).agg({'charges':'mean'})
data.groupby(['age_bin', 'region']).agg({'charges':'mean'})
data.groupby(['children', 'bmi_bin']).agg({'charges':'mean'})
data.groupby(['smoker', 'bmi_bin']).agg({'charges':'mean'})
data.groupby(['region', 'bmi_bin']).agg({'charges':'mean'})
data.groupby(['children', 'smoker']).agg({'charges':'mean'})
data.groupby(['children', 'region']).agg({'charges':'mean'})
data.groupby(['region', 'smoker']).agg({'charges':'mean'})
data.groupby(['sex', 'smoker']).agg({'charges':'mean'})
data.head()
data['smoker'].replace('yes', 1, inplace=True)

data['smoker'].replace('no', 0, inplace=True)
sns.lmplot(x='age_bin', y='charges', hue='smoker', col='sex',data=data, palette='husl')
sns.pairplot(data, vars= ['age_bin','bmi_bin','children', 'smoker','charges'], hue='sex')
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

le.fit(data.sex.drop_duplicates()) 

data.sex = le.transform(data.sex)

le.fit(data.region.drop_duplicates()) 

data.region = le.transform(data.region)
df = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]
data = data[['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges']]
data_corr = data.corr()['charges'][:-1] # -1 because the latest row is Target

golden_features_list = data_corr[abs(data_corr) > 0.1].sort_values(ascending=False)

print("There is {} strongly correlated values with Target:\n{}".format(len(golden_features_list), golden_features_list))
corr = data.corr() 

plt.figure(figsize=(12, 10))



sns.heatmap(corr, 

            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,

            annot=True, annot_kws={"size": 8}, square=True);


x = data.drop(['charges'], axis = 1)

y = data.charges



x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression().fit(x_train,y_train)



y_train_pred = lr.predict(x_train)

y_test_pred = lr.predict(x_test)



print(lr.score(x_test,y_test))
dt = DecisionTreeRegressor().fit(x_train,y_train)



y_train_pred = dt.predict(x_train)

y_test_pred = dt.predict(x_test)



print(dt.score(x_test,y_test))
dt = DecisionTreeRegressor().fit(x_train,y_train)



y_train_pred = dt.predict(x_train)

y_test_pred = dt.predict(x_test)



print(dt.score(x_test,y_test))
rf = RandomForestRegressor().fit(x_train,y_train)



y_train_pred = rf.predict(x_train)

y_test_pred = rf.predict(x_test)



print(rf.score(x_test,y_test))
gbm = GradientBoostingRegressor().fit(x_train,y_train)



y_train_pred = gbm.predict(x_train)

y_test_pred = gbm.predict(x_test)



print(gbm.score(x_test,y_test))
lbm = LGBMRegressor().fit(x_train,y_train)



y_train_pred = lbm.predict(x_train)

y_test_pred = lbm.predict(x_test)



print(lbm.score(x_test,y_test))
feat_imp = pd.DataFrame({'importance':gbm.feature_importances_})    

feat_imp['feature'] = x_train.columns

feat_imp.sort_values(by='importance', ascending=False, inplace=True)

feat_imp