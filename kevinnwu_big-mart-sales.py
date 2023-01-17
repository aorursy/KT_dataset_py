import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

%matplotlib inline
train = pd.read_csv('../input/big-mart-sales-prediction/Train.csv')
train.head()
train.info()
train.nunique()
train.isnull().sum()
train.isnull().sum()/train.count()*100
train.groupby(['Outlet_Identifier','Outlet_Size']).count()
train = train.drop('Outlet_Size', axis =1)
train.groupby('Item_Identifier').mean().sort_values('Item_Weight')
train[train['Item_Identifier'].isin(['FDE52','FDK57','FDN52','FDQ60'])]
train = train.drop(train[train['Item_Identifier'].isin(['FDE52','FDK57','FDN52','FDQ60'])].index)
train[train['Item_Identifier']=='FDX49']
Item_Spec = train.groupby(['Item_Identifier', 'Item_Weight']).sum().reset_index()[['Item_Identifier','Item_Weight']]

Item_Spec = pd.Series(Item_Spec['Item_Weight'].values, index=Item_Spec['Item_Identifier']).to_dict()
Item_Spec
train['Item_Weight'] = train['Item_Weight'].fillna(train['Item_Identifier'].map(Item_Spec))
train.isnull().sum()
train['Item_Fat_Content'].unique()
Fat_Content = {'Low Fat':'Low Fat', 'low fat':'Low Fat', 'LF':'Low Fat', 'Regular':'Regular', 'reg':'Regular'}
Fat_Content
train['Item_Fat_Content'] = train['Item_Fat_Content'].map(Fat_Content)
train['Item_Fat_Content'].unique()
train
plt.figure(figsize=(14,6))

sns.distplot(train['Item_Outlet_Sales'], bins =14)
data = train

index = train.groupby('Item_Type').nunique().index

fig = px.pie(data,names='Item_Type',  title='zz')

fig.show()
plt.figure(figsize=(14,6))

train.groupby('Item_Type').nunique()['Item_Identifier'].plot(kind='bar')
fig = px.histogram(train,

                   x="Item_Weight", 

                   color="Item_Fat_Content", 

                   marginal="box",

                   #title='Reading Score - Gender', 

                   #barmode='overlay',

                   nbins=20

                  )

fig.update_layout(yaxis=dict(title=''))

fig.show()
fig = px.histogram(train,

                   x='Item_Type', 

                   y='Item_Outlet_Sales',

                   color="Item_Type", 

                   histfunc='avg',

                   nbins=20

                  ).update_xaxes(categoryorder='total descending')

#fig.updatelayout(xaxis=dict())

fig.show()
fig = px.scatter(train,

                 x='Item_MRP', 

                 y='Item_Outlet_Sales',

                )

fig.show()
fig = px.scatter(train,

                 x='Outlet_Establishment_Year', 

                 y='Outlet_Location_Type',

                 color='Outlet_Type',

                 symbol='Outlet_Type',

                 text='Outlet_Identifier',

                ).update_yaxes(categoryorder='total ascending')



fig.update_traces(marker=dict(size=12,),

                  textposition='top center',

                  textfont=dict(family='Arial',size=12),

              

                 )

fig.update_layout(

    height=600,

)



fig.show()
fig3 = plt.figure(constrained_layout=True,figsize=(14,12))

gs = fig3.add_gridspec(3, 4)

f3_ax1 = fig3.add_subplot(gs[0, 0:2])

f3_ax2 = fig3.add_subplot(gs[0, 2:4])

f3_ax3 = fig3.add_subplot(gs[1, :])

f3_ax4 = fig3.add_subplot(gs[2, :])



sns.boxplot(x='Outlet_Location_Type',

            y='Item_Outlet_Sales',

            data = train,

            order=['Tier 1', 'Tier 2', 'Tier 3'],

            ax=f3_ax1,        

           )



sns.boxplot(x='Outlet_Type',

            y='Item_Outlet_Sales',

            data=train,

            order=['Grocery Store', 'Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3'],

            ax=f3_ax2

           )



sns.boxplot(x='Outlet_Establishment_Year',

            y='Item_Outlet_Sales',

            data = train,

            ax=f3_ax3

           )



sns.boxplot(x='Outlet_Identifier',

            y='Item_Outlet_Sales',

            data=train,

            order=['OUT019', 'OUT010', 'OUT018', 'OUT049', 'OUT035', 'OUT045', 'OUT017', 'OUT046', 'OUT013', 'OUT027'],

            ax=f3_ax4

           )



train.corr()
train.nunique()
data_train = train[['Item_Fat_Content', 'Item_Type', 'Item_MRP', 'Outlet_Identifier', 'Outlet_Location_Type', 'Outlet_Type','Item_Outlet_Sales']]
data_train = pd.get_dummies(data_train)
data_train.corr()['Item_Outlet_Sales']
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, SGDRegressor, Lasso

from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics

from sklearn.pipeline import make_pipeline
X = data_train.drop('Item_Outlet_Sales', axis=1).values

y = data_train['Item_Outlet_Sales'].values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()

lm.fit(X_train, y_train)
predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)
np.sqrt(metrics.mean_squared_error(y_test, predictions))
train.describe()
reg = make_pipeline(StandardScaler(),SGDRegressor(max_iter=1000, tol=1e-3))
reg.fit(X_train, y_train)
SGD_Predictions = reg.predict(X_test)
plt.scatter(y_test, SGD_Predictions)
np.sqrt(metrics.mean_squared_error(y_test, SGD_Predictions))
ls = Lasso(alpha=1)
ls.fit(X_train, y_train)
Lasso_Predictions = ls.predict(X_test)
plt.scatter(y_test, Lasso_Predictions)
np.sqrt(metrics.mean_squared_error(y_test, Lasso_Predictions))
result = [10000,0,0]

for sample in range(1,40):

    for leaf in range(1,sample):

        DCT = DecisionTreeRegressor(min_samples_split=sample, min_samples_leaf=leaf)

        DCT.fit(X_train, y_train)

        DecisionTree_Predictions = DCT.predict(X_test)

        rmse = np.sqrt(metrics.mean_squared_error(y_test, DecisionTree_Predictions))

        if rmse < result[0]:

            result[0] = rmse

            result[1] = sample

            result[2] = leaf

    
result
DCT = DecisionTreeRegressor(min_samples_split=37, min_samples_leaf=36)

DCT.fit(X_train, y_train)

DecisionTree_Predictions = DCT.predict(X_test)

plt.scatter(y_test, DecisionTree_Predictions)
np.sqrt(metrics.mean_squared_error(y_test, DecisionTree_Predictions))