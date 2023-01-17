# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.offline as py

from scipy.stats import norm

from scipy import stats

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px

import plotly.tools as tls

import plotly.figure_factory as ff

from sklearn.metrics import r2_score, mean_squared_error



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
df.head()
df.info()
df.describe()
pd.DataFrame(round(df.isnull().sum()/df.shape[0] * 100,3), columns = ['Missing'])
categorical_col = []

for column in df.columns:

    if df[column].dtype == object and len(df[column].unique()) <= 50:

        categorical_col.append(column)

        print(f"{column} : {df[column].unique()}")

        print("====================================")
numerical_col = []

for column in df.columns:

    if df[column].dtype != object and len(df[column].unique()) <= 50:

        numerical_col.append(column)

        print(f"{column} : {df[column].unique()}")

        print("====================================")
df.head()
df.shape
plt.rcParams['figure.figsize'] = 14,7

sns.countplot(df['price'][:200], palette='Set1')

plt.title("Count plot of the price variable")

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize=(14,10))

sns.set_style("darkgrid")

sns.jointplot(x = 'price', y = 'number_of_reviews', data=df, color = 'darkgreen',height = 8, ratio = 4)
df.head()
fig = go.Figure(go.Bar(y=df['name'][:30], x=df['number_of_reviews'], # Need to revert x and y axis

                      orientation="h")) # default orentation value is "v" - vertical ,we need to change it as orientation="h"

fig.update_layout(title_text='Top 30 Hotel with their reviews',xaxis_title="Count",yaxis_title="New York City Air BnB Hotels")

fig.show()
df.head()
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", hover_name="neighbourhood", hover_data=["neighbourhood_group"],

                        color_discrete_sequence=["fuchsia"], zoom=8,center=dict(lat=40.74765, lon=-73.89445), height=300)

fig.update_layout(mapbox_style="open-street-map")

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
data=df['neighbourhood_group'].value_counts().to_frame().reset_index().rename(columns={'index':'neighbourhood_group','neighbourhood_group':'count'})
colors=['red','green','yellow','light blue', 'pink']

fig = go.Figure([go.Pie(labels=data['neighbourhood_group'], values=data['count'])])

fig.update_traces(hoverinfo='label+percent', textinfo='percent+label', textfont_size=15,

                 marker=dict(colors=colors, line=dict(color='#000000', width=2)))

fig.update_layout(title="PERCENTAGE WISE NEIGHBOURHOOD GROUPS",title_x=0.5)

fig.show()
plt.figure(figsize=(20,10))

df.groupby('room_type')['price'].value_counts().nlargest(20).sort_values(ascending=True).plot(kind='barh')

plt.xlabel('Price Of The Room')

plt.show
fig = px.scatter(df, x='name', y='price',

                 color='availability_365') # Added color to previous basic 

fig.update_layout(title='NAME & PRICE OF THE HOTELS WITH AVAILABILITY STATUS',xaxis_title="HOTEL NAMES",yaxis_title="PRICE")

fig.show()
df.head()
plt.figure(figsize=(10,10))

sns.distplot(df['price'], fit=norm)

plt.title("Price Distribution Plot",size=15, weight='bold')
df['price'] = np.log(df.price+1)
plt.figure(figsize=(12,10))

sns.distplot(df['price'], fit=norm)

plt.title("Log-Price Distribution Plot",size=15, weight='bold')
plt.figure(figsize=(7,7))

stats.probplot(df['price'], plot=plt)

plt.show()
df.head()
df.room_type.value_counts()




from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()



df['room_type'] = le.fit_transform(df['room_type'])

df['neighbourhood_group'] = le.fit_transform(df['neighbourhood_group'])
df = df.drop(columns=['name','id' ,'host_id','host_name', 'last_review', 'neighbourhood'])

df.head()
sns.heatmap(df.isnull(), cmap='viridis')
df.room_type.value_counts()
df['reviews_per_month'] = df['reviews_per_month'].fillna(df['reviews_per_month'].mean())
plt.figure(figsize=(15,12))

palette = sns.diverging_palette(20, 220, n=256)

corr=df.corr(method='pearson')

sns.heatmap(corr, annot=True, fmt=".2f", cmap=palette, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}).set(ylim=(11, 0))

plt.title("Correlation Matrix",size=15, weight='bold')
x = df.drop(['price'], axis=1) ## Independent variable

y = df['price'] ## Dependent variable
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 42)
from sklearn.preprocessing import StandardScaler



sc = StandardScaler()



x_train = sc.fit_transform(x_train)

x_test = sc.transform(x_test)
from sklearn.linear_model import LinearRegression



lr = LinearRegression()



lr.fit(x_train, y_train)



lr_pred = lr.predict(x_test)



r2 = r2_score(y_test,lr_pred)

print('R-Square Score: ',r2*100)





# Calculate the absolute errors

lr_errors = abs(lr_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(lr_pred), 2), 'degrees.')





# Calculate mean absolute percentage error (MAPE)

mape = 100 * (lr_errors / y_test)







sns.distplot(y_test-lr_pred)
from sklearn.metrics import mean_absolute_error,mean_squared_error



print('mse:',mean_squared_error(y_test, lr_pred))

print('mae:',mean_absolute_error(y_test, lr_pred))

print('rmse', np.sqrt(mean_absolute_error(y_test, lr_pred)))
from sklearn.tree import DecisionTreeRegressor



dtree = DecisionTreeRegressor(criterion='mse')

dtree.fit(x_train, y_train)





dtree_pred = dtree.predict(x_test)



r2 = r2_score(y_test,dtree_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

dtree_errors = abs(dtree_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(dtree_pred), 2), 'degrees.')

print('mse:',mean_squared_error(y_test, dtree_pred))

print('mae:',mean_absolute_error(y_test, dtree_pred))

print('rmse', np.sqrt(mean_absolute_error(y_test, dtree_pred)))
from sklearn.ensemble import RandomForestRegressor



random_forest_regressor = RandomForestRegressor()

random_forest_regressor.fit(x_train, y_train)



rf_pred = random_forest_regressor.predict(x_test)



r2 = r2_score(y_test,rf_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

rf_errors = abs(rf_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(rf_pred), 2), 'degrees.')
print('mse:',mean_squared_error(y_test, rf_pred))

print('mae:',mean_absolute_error(y_test, rf_pred))

print('rmse', np.sqrt(mean_absolute_error(y_test, rf_pred)))
import xgboost as xgb



xg_boost = xgb.XGBRegressor()



xg_boost.fit(x_train, y_train)



xgb_pred = xg_boost.predict(x_test)



r2 = r2_score(y_test,xgb_pred)

print('R-Square Score: ',r2*100)



# Calculate the absolute errors

xgb_errors = abs(xgb_pred - y_test)

# Print out the mean absolute error (mae)

print('Mean Absolute Error:', round(np.mean(xgb_pred), 2), 'degrees.')
print('mse:',mean_squared_error(y_test, xgb_pred))

print('mae:',mean_absolute_error(y_test, xgb_pred))

print('rmse', np.sqrt(mean_absolute_error(y_test, xgb_pred)))