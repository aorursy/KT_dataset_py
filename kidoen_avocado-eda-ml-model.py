

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/avocado-prices/avocado.csv")
df.head()
df = df.drop(columns=['Unnamed: 0'],axis=1)
df['Date'] = pd.to_datetime(df['Date'])
df.head()
import seaborn as sns

sns.heatmap(df.isnull())  # to check null values in the data
df.describe()
df['type'].unique()
sns.countplot(df['type'])
sns.boxplot(data=df,y='type',x='AveragePrice')
sns.heatmap(df.corr(),cmap='viridis',annot=True)

sns.distplot(df.AveragePrice)
df.head(2)
df.region.unique()
organic = df[df['type']=='organic']
sns.factorplot('AveragePrice','region',data=organic,hue='year',join=False,size=15)
conventional = df[df['type']=='conventional']
sns.factorplot('AveragePrice','region',data=conventional , size=15,join=False,hue='year')
import matplotlib.pyplot as plt
plt.figure(figsize=(14,10))
sns.scatterplot(x='Total Volume', y = 'AveragePrice', hue= 'type', data= df)
variant = df[['4046', '4225', '4770']].groupby(df.year).sum()
variant.plot(kind='line')

bags = df[['Small Bags','Large Bags']].groupby(df['region']).sum()
bags.plot(kind='line',figsize=(14,8))
plt.show()
import plotly
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
grouped = df.groupby('Date').mean()
plot = go.Scatter(x=grouped.AveragePrice.index, y = grouped.AveragePrice)
data = [plot]
layout=go.Layout(title="Time Series Plot for Mean Daily Price across all regions", xaxis={'title':'Date'}, yaxis={'title':'Prices'})
figure=go.Figure(data=data,layout=layout)
iplot(figure)
conventional = df[df['type']=='conventional']
organic = df[df['type']=="organic"]

group_conv = conventional.groupby('Date').mean()
plot1 = go.Scatter(x=group_conv.AveragePrice.index,y=group_conv.AveragePrice,name='conventional')

organic_conv = organic.groupby('Date').mean()
plot2 = go.Scatter(x=organic_conv.AveragePrice.index,y=organic_conv.AveragePrice,name='organic')

data=[plot1,plot2]
layout = go.Layout(title='time series for conventional and organic types(mean)',xaxis = {'title':'Date'},yaxis={'title':'Prices'})
figure = go.Figure(data=data,layout =layout )
iplot(figure)
conventional = df[df['type']=='conventional']
organic = df[df['type']=="organic"]

group_conv = conventional.groupby('Date').mean()
plot1 = go.Scatter(x=group_conv['Total Volume'].index,y=group_conv['Total Volume'],name='conventional')

organic_conv = organic.groupby('Date').mean()
plot2 = go.Scatter(x=organic_conv['Total Volume'].index,y=organic_conv['Total Volume'],name='organic')

data=[plot1,plot2]
layout = go.Layout(title='time series for conventional and organic types(mean) of total volume',xaxis = {'title':'Date'},yaxis={'title':'Prices'})
figure = go.Figure(data=data,layout =layout )
iplot(figure)
groupbyregion_conv = conventional.groupby('region').median()
groupbyregion_conv = groupbyregion_conv.sort_values('AveragePrice')

plot = go.Bar(x = groupbyregion_conv.AveragePrice,
             y=groupbyregion_conv.AveragePrice.index,
             orientation='h')
data = [plot]
layout=go.Layout(title="Median Price of Conventional Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'},height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupbyregion_organic = organic.groupby('region').median()
groupbyregion_organic = groupbyregion_organic.sort_values('AveragePrice')

plot = go.Bar(x=groupbyregion_organic.AveragePrice,
             y = groupbyregion_organic.AveragePrice.index,
             orientation='h')
data = [plot]
layout=go.Layout(title="Median Price of Organic Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'},height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)
groupbyregion_conv = conventional.groupby('region').median()
groupbyregion_conv = groupbyregion_conv.sort_values('Total Volume')

plot = go.Bar(x = groupbyregion_conv['Total Volume'],
             y=groupbyregion_conv['Total Volume'].index,
             orientation='h')
data = [plot]
layout=go.Layout(title="volume of Conventional Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'},height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)

groupbyregion_organic = organic.groupby('region').median()
groupbyregion_organic = groupbyregion_organic.sort_values('Total Volume')

plot = go.Bar(x=groupbyregion_organic['Total Volume'],
             y = groupbyregion_organic['Total Volume'].index,
             orientation='h')
data = [plot]
layout=go.Layout(title="volume of Organic Avocado by Region", xaxis={'title':'Date'}, yaxis={'title':'Region'},height=1200)
figure=go.Figure(data=data,layout=layout)
iplot(figure)

import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Lasso,Ridge,BayesianRidge,ElasticNet,HuberRegressor,LinearRegression,LogisticRegression,SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor


df.head()
X = df.drop('AveragePrice',1)
y = df.AveragePrice
X = X[:-1].reset_index(drop=True)
y = y[1:].reset_index(drop=True)


ml_df = pd.concat([X,y], 1)
ml_df =  ml_df.groupby('Date').mean()


X = ml_df.drop('AveragePrice',1)
y = ml_df.AveragePrice

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.10, shuffle=False)

X_train.shape,X_test.shape
classifiers = [['DecisionTree :',DecisionTreeRegressor()],
               ['RandomForest :',RandomForestRegressor()],
               ['KNeighbours :', KNeighborsRegressor(n_neighbors = 2)],
               ['SVM :', SVR()],
               ['AdaBoostClassifier :', AdaBoostRegressor()],
               ['GradientBoostingClassifier: ', GradientBoostingRegressor()],
               ['Xgboost: ', XGBRegressor()],
               ['CatBoost: ', CatBoostRegressor(logging_level='Silent')],
               ['Lasso: ', Lasso()],
               ['Ridge: ', Ridge()],
               ['BayesianRidge: ', BayesianRidge()]]
for name,classifier in classifiers:
    classifier = classifier
    classifier.fit(X_train,y_train)
    preds = classifier.predict(X_test)
    print(name,(np.sqrt(mean_squared_error(preds,y_test))))
    
    
new_df = df.copy()
new_df.head(2)


# CLASSIFICATION ON TYPE 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc,accuracy_score,classification_report
X = new_df.drop(['Date','type','region'],axis=1)
y = new_df.type

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 0)

X_train.shape,X_test.shape
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
rf_pred = rf.predict(X_test)
print(classification_report(rf_pred,y_test))

xgboost = xgb.XGBClassifier()
xgboost.fit(X_train,y_train)
xgb_pred = xgboost.predict(X_test)
print(classification_report(xgb_pred,y_test))
