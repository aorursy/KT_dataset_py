# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
from matplotlib import pyplot as plt
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from statsmodels.api import OLS
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


train = pd.read_csv("../input/big-mart-sales/Train.csv")
test = pd.read_csv("../input/big-mart-sales/Test.csv")
test1 = test.copy()
print('Train data shape:', train.shape)
print('Test data shape:',test.shape)
train.head()
test["Outlet_Size"].unique()
train.nunique()
test.nunique()
train.isna().sum()
map1 = {"Small":1,"Medium":2,"High":3}
train["Outlet_Size"] = train["Outlet_Size"].map(map1)
train["Item_Weight"] = train["Item_Weight"].fillna(train.Item_Weight.mean())
train["Outlet_Size"] = train["Outlet_Size"].fillna(train["Outlet_Size"].median())
train.isna().sum()
map1 = {"Small":1,"Medium":2,"High":3}
test["Outlet_Size"] = test["Outlet_Size"].map(map1)
test["Item_Weight"] = test["Item_Weight"].fillna(test.Item_Weight.mean())
test["Outlet_Size"] = test["Outlet_Size"].fillna(test["Outlet_Size"].median())
train.head()
train['Item_Outlet_Sales'].plot(legend=True,label='SALES',figsize=(12,8))
f, ax = plt.subplots(1,1, figsize=(10, 6))
plot = sns.lineplot(y='Item_Outlet_Sales', x='Outlet_Establishment_Year', data=train)
plot.set(title='Sales Data')
plt.figure(figsize=(15,6))
plt.scatter(y='Outlet_Establishment_Year', x='Outlet_Identifier', data=train, marker = '*')
plt.suptitle('Establishment of Outlet in Year')
import matplotlib.pyplot as plt
import seaborn as sns

plt.hist(train["Item_Outlet_Sales"],bins = 100)

plt.hist(train["Item_MRP"],bins = 100, color='brown')
fig,axes=plt.subplots(1,1,figsize=(12,8))
sns.scatterplot(x='Item_MRP',y='Item_Outlet_Sales',hue='Item_Fat_Content',size='Item_Weight',data=train)
plt.figure(figsize=(7,4))
sns.countplot(data= train, x= "Outlet_Type")
sns.countplot(train["Outlet_Location_Type"])
plt.show()
plt.figure(figsize=(25,8))
sns.countplot(data= train, x= "Item_Type")
sns.countplot(train["Outlet_Size"])
plt.show()
sns.countplot(train["Outlet_Type"],palette = 'RdYlGn')
plt.xticks(rotation = 90)
plt.show()
sns.violinplot(x=train["Outlet_Size"],y=train["Item_Outlet_Sales"],hue = train["Outlet_Size"],palette = "RdYlGn")
plt.legend()
plt.show()

plt.figure(figsize= (9,6))
corr= train.corr()
sns.heatmap(corr, linewidths=1.5, annot= True, cmap='RdYlGn')
train.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
test.drop(labels = ["Outlet_Establishment_Year"],inplace = True,axis =1)
feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X = pd.get_dummies(train[feat])
train = pd.concat([train,X],axis=1)
train.head()
feat = ['Outlet_Size','Outlet_Type','Outlet_Location_Type','Item_Fat_Content',"Item_Type"]
X1 = pd.get_dummies(test[feat])
test = pd.concat([test,X1],axis=1)
train.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)
test.drop(labels = ["Outlet_Size",'Outlet_Location_Type',"Outlet_Type",'Item_Fat_Content','Outlet_Identifier','Item_Identifier',"Item_Type"],axis=1,inplace = True)
train.head()
X_train = train.drop(labels = ["Item_Outlet_Sales"],axis=1)
y_train = train["Item_Outlet_Sales"]
X_train.shape,y_train.shape
train.head()
train.shape, test.shape
## Checking the presence of outliers
pos = 1
plt.figure(figsize=(20,8))
for i in train.columns:
    plt.subplot(8, 4, pos)
    sns.boxplot(train[i],color="red")
    pos += 1
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(train))
print('Z-score of column values:\n', z)
# Setting threshold to identify an outlier
threshold = 3
print(np.where(z > 3))
print(np.where(z < -3))
train_outliers= train
print('Shape of sales with outliers:', train_outliers.shape)
train = train[(z < 3).all(axis=1)]
print('Shape of sales data after removing outliers:', train.shape)
## Checking the presence of outliers
pos = 1
plt.figure(figsize=(20,8))
for i in train.columns:
    plt.subplot(8, 4, pos)
    sns.boxplot(train[i],color="orange")
    pos += 1
## Detect outliers using IQR and Handling outliers
Q1 = train.quantile(0.25)
Q3 = train.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
## Checking for outliers presence of data points with "True"
bool_outs= (train < (Q1 - 1.5 * IQR)) |(train > (Q3 + 1.5 * IQR))
print(bool_outs)
## Removing outliers from dataframe
train = train[~bool_outs.any(axis=1)]
print('Shape of dataframe without outliers: {}'.format(train.shape))
## Checking the presence of outliers
pos = 1
plt.figure(figsize=(20,8))
for i in train.columns:
    plt.subplot(8, 4, pos)
    sns.boxplot(train[i],color="green")
    pos += 1
y_train.head()
from sklearn import preprocessing

x = X_train.values #returns a numpy array
test_s = test.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled_train = min_max_scaler.fit_transform(x)
x_scaled_test = min_max_scaler.fit_transform(test_s)
df_train = pd.DataFrame(x_scaled_train)
df_test = pd.DataFrame(x_scaled_test)
df_train.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.25)

pipelines = []
pipelines.append(('ScaledLR', Pipeline([('Scaler', StandardScaler()),('LR',LinearRegression())])))
pipelines.append(('ScaledElastic', Pipeline([('Scaler', StandardScaler()),('EN',ElasticNet())])))
pipelines.append(('ScaledKNN', Pipeline([('Scaler', StandardScaler()),('KNN', KNeighborsRegressor())])))
pipelines.append(('ScaledDT', Pipeline([('Scaler', StandardScaler()),('DT', DecisionTreeRegressor())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestRegressor())])))
pipelines.append(('ScaledGBM', Pipeline([('Scaler', StandardScaler()),('GB', GradientBoostingRegressor())])))
pipelines.append(('ScaledAda', Pipeline([('Scaler', StandardScaler()),('Ada', AdaBoostRegressor())])))
pipelines.append(('ScaledETR', Pipeline([('Scaler', StandardScaler()),('ETR', ExtraTreesRegressor())])))
pipelines.append(('ScaledSVR', Pipeline([('Scaler', StandardScaler()),('SVM', SVR())])))
pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGBR', XGBRegressor())])))
pipelines.append(('ScaledMLP', Pipeline([('Scaler', StandardScaler()),('NNW', MLPRegressor())])))
results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=10, random_state=21)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='neg_root_mean_squared_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
# Algorithm comparison
fig = plt.figure(figsize=(18,5))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
xgb_model= XGBRegressor(n_estimators=15).fit(X_train, y_train)
xgb_preds= xgb_model.predict(X_test)
print('RMSE of XGB:', np.sqrt(mean_squared_error(xgb_preds, y_test)))
plt.scatter(y_test, xgb_preds)
plt.show()
xgb_predictions = xgb_model.predict(df_test)
xgb_final = pd.DataFrame({"Item_Identifier":test1["Item_Identifier"],"Outlet_Identifier":test1["Outlet_Identifier"],"Item_Outlet_Sales":abs(xgb_predictions)})
xgb_final.head()
xgb_final.to_csv('/kaggle/working/BigMart_Submission(XGB).csv',index=False)
gbm= GradientBoostingRegressor(n_estimators=15).fit(X_train, y_train)
gbm_preds= xgb_model.predict(X_test)
print('RMSE of GB:', np.sqrt(mean_squared_error(gbm_preds, y_test)))
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
lr_preds = model.predict(X_test)

plt.scatter(y_test, lr_preds)
plt.show()
lr_predictions = model.predict(df_test)
lr_df = pd.DataFrame({"Item_Identifier":test1["Item_Identifier"],"Outlet_Identifier":test1["Outlet_Identifier"],"Item_Outlet_Sales":abs(lr_predictions)})
lr_df.head()
sns.distplot((y_test-lr_preds),bins=50)
plt.show()

from sklearn import metrics
print('RMSE of Linear Regression:', np.sqrt(metrics.mean_squared_error(y_test, lr_preds)))
