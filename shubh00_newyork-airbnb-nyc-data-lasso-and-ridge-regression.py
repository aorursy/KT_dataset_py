## import required libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly

from plotly import tools

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head() ##sample dataset
df.describe().T    ## basic information about dataset
## creating table of freuency through plotly 

cat = ff.create_table(df.describe(include=['O']).T, index=True, index_title='Categorical columns')

iplot(cat)
df.info() ## dataset information
df.shape
df.isnull().sum() ## checking for null values
df.columns ## columns present in dataset
df.drop(['id','name','host_id','host_name','last_review'], inplace=True , axis=1) ## droping features
df.room_type.unique() ##checking for unique values in the feature
df.neighbourhood.nunique()
df.neighbourhood_group.unique()
cont = pd.crosstab(df.neighbourhood_group, df.room_type)

cont
contt = ff.create_table(pd.crosstab(df.neighbourhood, df.room_type), index=True)

iplot(contt)
df.reviews_per_month.fillna(0,inplace=True)
plt.figure(figsize=(22,10))



plt.subplot(1,2,1)

plt.title('Price Distribution Plot')

sns.distplot(df.price)



plt.subplot(1,2,2)

plt.title('Price Spread')

sns.boxplot(y=df.price)



plt.show()
from scipy import stats

z = np.abs(stats.zscore(df['price']))

print(z)
threshold = 3

print(np.where(z > 3))
from scipy.stats import skew

print(skew(df.price))
sns.countplot(df.room_type)
import plotly.express as px



fig = px.violin(df, y="price", x="neighbourhood_group", points="all")

fig.show()
plt.figure(figsize=(8,8))

sns.set(style="whitegrid")



ax = sns.barplot(x="neighbourhood_group", y="price",data=df)
plt.figure(figsize=(9,8))

sns.set(style="whitegrid")



ax = sns.countplot(x="room_type", hue="neighbourhood_group",data=df)
print(skew(df.minimum_nights))

plt.figure(figsize=(8,8))

sns.distplot(df.minimum_nights)

plt.title('Minimum no. of nights distribution')

plt.show()
brook = df[df.neighbourhood_group=="Brooklyn"].availability_365

queen = df[df.neighbourhood_group=="Queens"].availability_365

manh = df[df.neighbourhood_group=="Manhattan"].availability_365

island = df[df.neighbourhood_group=="Staten Island"].availability_365

bronx = df[df.neighbourhood_group=="Bronx"].availability_365



fig = go.Figure()

# Use x instead of y argument for horizontal plot

fig.add_trace(go.Box(x=brook, name='Brooklyn'))

fig.add_trace(go.Box(x=queen, name='Queens'))

fig.add_trace(go.Box(x=manh, name='Manhattan'))

fig.add_trace(go.Box(x=island, name='Staten Island'))

fig.add_trace(go.Box(x=bronx, name='Bronx'))



fig.show()
plt.figure(figsize=(10,6))

sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group)

plt.ioff()
sns.clustermap(df.corr(),annot = False);
g = sns.PairGrid(df, hue='neighbourhood_group')

g.map_diag(plt.hist,edgecolor = 'w')

g.map_offdiag(plt.scatter,edgecolor = 'w')

g.add_legend()

plt.subplots_adjust(top=0.9)

g.fig.suptitle(' Attributes distribution based on location')
avg_price = df.groupby(['neighbourhood_group'])['price'].mean().plot(kind='bar')
df.groupby(['neighbourhood_group'])['minimum_nights'].count().plot(kind='bar')
avg_price = df.groupby(['neighbourhood_group'])['availability_365'].count().plot(kind='bar')                       
from sklearn import preprocessing

le = preprocessing.LabelEncoder()



le.fit(df['neighbourhood_group'])

df['neighbourhood_group']=le.transform(df['neighbourhood_group'])



le.fit(df['neighbourhood'])

df['neighbourhood']=le.transform(df['neighbourhood'])



le.fit(df['room_type'])

df['room_type']=le.transform(df['room_type'])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

lm = LinearRegression()
X = df.drop(['price'], inplace=False, axis=1)

y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)



model=lm.fit(X_train, y_train)
lm.intercept_
predictions = lm.predict(X_test)
from sklearn import metrics

mae = metrics.mean_absolute_error(y_test, predictions)

mse = metrics.mean_squared_error(y_test, predictions)

rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))

r2 = metrics.r2_score(y_test, predictions)



print('MAE (Mean Absolute Error): %s' %mae)

print('MSE (Mean Squared Error): %s' %mse)

print('RMSE (Root mean squared error): %s' %rmse)

print('R2 score: %s' %r2)
from sklearn.model_selection import cross_val_score

mse_ng=cross_val_score(lm,X,y,scoring='neg_mean_squared_error',cv=5)

mse_ng

mean_mse=np.mean(mse_ng)

print(mean_mse)
plt.figure(figsize=(13,8))

sns.regplot(y=y_test, x=predictions, color='brown')

plt.title('Model prediction')
error = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': predictions.flatten()})

error.head(10)
from sklearn.feature_selection import RFE
lr2 = LinearRegression()
rfe_selector = RFE(lr2, 8, verbose=True)
rfe_selector.fit(X_train, y_train)
rfe_selector.support_
rfe_selector.ranking_
cols_keep = X_train.columns[rfe_selector.support_]
cols_keep
lr2 = LinearRegression()
lr2.fit(X_train[cols_keep],y_train)
y_train_pred = lr2.predict(X_train[cols_keep])
r3= metrics.r2_score(y_train, y_train_pred)

r3
from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV



ridge=Ridge()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

ridge_regressor=GridSearchCV(ridge,parameters,scoring='neg_mean_squared_error',cv=5)

ridge_regressor.fit(X,y)
print(ridge_regressor.best_params_)

print(ridge_regressor.best_score_)
from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

lasso=Lasso()

parameters={'alpha':[1e-15,1e-10,1e-8,1e-3,1e-2,1,5,10,20,30,35,40,45,50,55,100]}

lasso_regressor=GridSearchCV(lasso,parameters,scoring='neg_mean_squared_error',cv=5)



lasso_regressor.fit(X,y)

print(lasso_regressor.best_params_)

print(lasso_regressor.best_score_)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
prediction_lasso=lasso_regressor.predict(X_test)

prediction_ridge=ridge_regressor.predict(X_test)
sns.distplot(y_test-prediction_lasso)
sns.distplot(y_test-prediction_ridge)