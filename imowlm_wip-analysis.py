!pip install bubbly
!pip install dabl
# for basic operations

import numpy as np

import pandas as pd



# for visualizations

import matplotlib.pyplot as plt

import seaborn as sns



# for interactive visualizations

import plotly.offline as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.offline as offline

offline.init_notebook_mode()

from plotly import tools

import plotly.figure_factory as ff

import plotly.express as px



from bubbly.bubbly import bubbleplot



import plotly.tools as tls

import squarify

from mpl_toolkits.basemap import Basemap

from numpy import array

from matplotlib import cm

import dabl



# for providing path

import os

print(os.listdir('../input/'))
data = pd.read_csv('../input/who_suicide_statistics.csv')



data = data.sort_values(['year'], ascending = True)



print(data.shape)
# let's check the total number of countries' data available for suicidal analysis



print("No. of Countries available for analysis :", data['country'].nunique())
# checking the head of the table



dat = ff.create_table(data.head())

py.iplot(dat)
# let's describe the data



dat = ff.create_table(data.describe())

py.iplot(dat)

# renaming the columns



data.rename({'sex' : 'gender', 'suicides_no' : 'suicides'}, inplace = True, axis = 1)



data.columns
# checking the percentage of missing values in the dataset



missing_percentage = data.isnull().sum()/data.shape[0]

print(missing_percentage*100)
# lets also check the Correlation of Suicides and Population



corr = data['suicides'].corr(data['population'])

print("Correlation Between Suicides and Population : {0:.2f}".format(corr))
# lets check the countries with more than 5000 suicides in age groupb 15-24 years 





x = data[(data['age'] == '15-24 years') 

     & (data['suicides'] > 5000)][['country',

                                   'suicides',

                                   'gender',

                                   'year']].sort_values(by = 'suicides', ascending = False)

x.style.background_gradient(cmap = 'Wistia')
# Top 10 Countries wrt Suicides



data[['country',

      'suicides']].groupby(['country']).agg('sum').sort_values(by = 'suicides',

                                                               ascending = False).head(10).style.background_gradient(cmap = 'Wistia')
# Countries with least number of suicides



data[['country',

      'suicides']].groupby(['country']).agg('sum').sort_values(by = 'suicides',

                                                               ascending = True).head(10).style.background_gradient(cmap = 'Wistia')
# lets check in which we highest no. of suicides



data[['suicides',

      'year']].groupby(['year']).agg('sum').sort_values(by = 'suicides',

                                                        ascending = False).head(10).style.background_gradient(cmap = 'Wistia')
# lets check the suicides according to gender



data[['gender','suicides']].groupby(['gender']).agg(['min','median','max']).style.background_gradient(cmap = 'Wistia')
# filling missing values



data['suicides'].fillna(0, inplace = True)

# data['population'].mean()

data['population'].fillna(1664090, inplace = True)



# checking if there is any null value left

data.isnull().sum().sum()



# converting these attributes into integer format

data['suicides'] = data['suicides'].astype(int)

data['population'] = data['population'].astype(int)
plt.rcParams['figure.figsize'] = (18, 3)

plt.style.use('fivethirtyeight')

dabl.plot(data, target_col = 'suicides')
df = px.data.gapminder()

df.head()
x = pd.merge(data, df, on = 'country')

x = x[['country', 'suicides','year_x','population','iso_alpha']]

x.head()
import plotly.express as px



plt.rcParams['figure.figsize'] = (18, 8)

plt.style.use('fivethirtyeight')



df = px.data.gapminder()

fig = px.choropleth(x,

                    locations="iso_alpha", 

                    color="suicides", 

                    hover_name="country",

                    animation_frame="year_x", 

                    range_color=[20,80],

                    )

fig.show()
# visualising the different countries distribution in the dataset



plt.style.use('seaborn-dark')

plt.rcParams['figure.figsize'] = (15, 9)



color = plt.cm.winter(np.linspace(0, 10, 100))

x = pd.DataFrame(data.groupby(['country'])['suicides'].sum().reset_index())

x.sort_values(by = ['suicides'], ascending = False, inplace = True)



sns.barplot(x['country'].head(10), y = x['suicides'].head(10), data = x, palette = 'winter')

plt.title('Top 10 Countries in Suicides', fontsize = 20)

plt.xlabel('Name of Country')

plt.xticks(rotation = 90)

plt.ylabel('Count')

plt.show()
# visualising the different year distribution in the dataset



plt.style.use('seaborn-dark')

plt.rcParams['figure.figsize'] = (18, 9)



x = pd.DataFrame(data.groupby(['year'])['suicides'].sum().reset_index())

x.sort_values(by = ['suicides'], ascending = False, inplace = True)



sns.lineplot(x['year'], y = x['suicides'], data = x, palette = 'cool')

plt.title('Distribution of suicides from the year 1985 to 2016', fontsize = 20)

plt.xlabel('year')

plt.xticks(rotation = 90)

plt.ylabel('count')

plt.show()


color = plt.cm.Blues(np.linspace(0, 1, 2))

data['gender'].value_counts().plot.pie(colors = color, figsize = (10, 10), startangle = 75)



plt.title('Gender', fontsize = 20)

plt.axis('off')

plt.show()
# visualising the different year distribution in the dataset



plt.style.use('seaborn-dark')

plt.rcParams['figure.figsize'] = (18, 9)



x = pd.DataFrame(data.groupby(['gender'])['suicides'].sum().reset_index())

x.sort_values(by = ['suicides'], ascending = False, inplace = True)



sns.barplot(x['gender'], y = x['suicides'], data = x, palette = 'afmhot')

plt.title('Distribution of suicides wrt Gender', fontsize = 20)

plt.xlabel('year')

plt.xticks(rotation = 90)

plt.ylabel('count')

plt.show()


suicide = pd.DataFrame(data.groupby(['country','year'])['suicides'].sum().reset_index())



count_max_sui=pd.DataFrame(suicide.groupby('country')['suicides'].sum().reset_index())



count = [ dict(

        type = 'choropleth',

        locations = count_max_sui['country'],

        locationmode='country names',

        z = count_max_sui['suicides'],

        text = count_max_sui['country'],

        colorscale = 'Cividis',

        autocolorscale = False,

        reversescale = True,

        marker = dict(

            line = dict (

                color = 'rgb(180,180,180)',

                width = 0.5

            ) ),

)]

layout = dict(

    title = 'Suicides happening across the Globe',

    geo = dict(

        showframe = True,

        showcoastlines = True,

        projection = dict(

            type = 'orthographic'

        )

    )

)

fig = dict( data=count, layout=layout )

iplot(fig, validate=False, filename='d3-world-map')
# looking at the Suicides in USA.



data[data['country'] == 'United States of America'].sample(20)
data['age'].value_counts()
df = data.groupby(['country', 'year'])['suicides'].mean()

df = pd.DataFrame(df)



# looking at the suicides trends for any 3 countries

plt.rcParams['figure.figsize'] = (20, 30)

plt.style.use('dark_background')



plt.subplot(3, 1, 1)

color = plt.cm.hot(np.linspace(0, 1, 40))

df['suicides']['United States of America'].plot.bar(color = color)

plt.title('Suicides Trends in USA wrt Year', fontsize = 30)



plt.subplot(3, 1, 2)

color = plt.cm.spring(np.linspace(0, 1, 40))

df['suicides']['Russian Federation'].plot.bar(color = color)

plt.title('Suicides Trends in Russian Federation wrt Year', fontsize = 30)



plt.subplot(3, 1, 3)

color = plt.cm.PuBu(np.linspace(0, 1, 40))

df['suicides']['Japan'].plot.bar(color = color)

plt.title('Suicides Trends in Japan wrt Year', fontsize = 30)



plt.show()
df2 = data.groupby(['country', 'age'])['suicides'].mean()

df2 = pd.DataFrame(df2)



# looking at the suicides trends for any 3 countries

plt.rcParams['figure.figsize'] = (20, 30)



plt.subplot(3, 1, 1)

df2['suicides']['United States of America'].plot.bar()

plt.title('Suicides Trends in USA wrt Age Groups', fontsize = 30)

plt.xticks(rotation = 0)



plt.subplot(3, 1, 2)

color = plt.cm.jet(np.linspace(0, 1, 6))

df2['suicides']['Russian Federation'].plot.bar(color = color)

plt.title('Suicides Trends in Russian Federation wrt Age Groups', fontsize = 30)

plt.xticks(rotation = 0)



plt.subplot(3, 1, 3)

color = plt.cm.Wistia(np.linspace(0, 1, 6))

df2['suicides']['Japan'].plot.bar(color = color)

plt.title('Suicides Trends in Japan wrt Age Groups', fontsize = 30)

plt.xticks(rotation = 0)



plt.grid()

plt.show()


plt.rcParams['figure.figsize'] = (18, 7)

plt.style.use('dark_background')



sns.stripplot(data['year'], data['suicides'], palette = 'cool')

plt.title('Year vs Suicides', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
# age-group vs suicides



plt.rcParams['figure.figsize'] = (18, 7)





sns.stripplot(data['gender'], data['suicides'], palette = 'Wistia')

plt.title('Age groups vs Suicides', fontsize = 20)

plt.grid()

plt.show()
# label encoding for gender



from sklearn.preprocessing import LabelEncoder



# creating an encoder

le = LabelEncoder()

data['gender'] = le.fit_transform(data['gender'])

data['age'] = le.fit_transform(data['age'])

# deleting unnecassary column



data = data.drop(['country'], axis = 1)



data.columns
#splitting the data into dependent and independent variables



x = data.drop(['suicides'], axis = 1)

y = data['suicides']



print(x.shape)

print(y.shape)
# splitting the dataset into training and testing sets



from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 45)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
# min max scaling



import warnings

warnings.filterwarnings('ignore')



# importing the min max scaler

from sklearn.preprocessing import MinMaxScaler



# creating a scaler

mm = MinMaxScaler()



# scaling the independent variables

x_train = mm.fit_transform(x_train)

x_test = mm.transform(x_test)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score



# creating the model

model = LinearRegression()



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)

from sklearn.ensemble import RandomForestRegressor



# creating the model

model = RandomForestRegressor()



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)

from sklearn.tree import DecisionTreeRegressor



# creating the model

model = DecisionTreeRegressor()



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)

from sklearn.ensemble import AdaBoostRegressor



# creating the model

model = AdaBoostRegressor()



# feeding the training data into the model

model.fit(x_train, y_train)



# predicting the test set results

y_pred = model.predict(x_test)



# calculating the mean squared error

mse = np.mean((y_test - y_pred)**2)

print("MSE :", mse)



# calculating the root mean squared error

rmse = np.sqrt(mse)

print("RMSE :", rmse)



#calculating the r2 score

r2 = r2_score(y_test, y_pred)

print("r2_score :", r2)

r2_score = np.array([0.385, 0.851, 0.745, 0.535])

labels = np.array(['Linear Regression', 'Random Forest', 'Decision Tree', 'AdaBoost Tree'])

indices = np.argsort(r2_score)

color = plt.cm.rainbow(np.linspace(0, 1, 9))



plt.style.use('seaborn-talk')

plt.rcParams['figure.figsize'] = (18, 7)

plt.bar(range(len(indices)), r2_score[indices], color = color)

plt.xticks(range(len(indices)), labels[indices])

plt.title('R2 Score', fontsize = 30)

plt.grid()

plt.tight_layout()

plt.show()
rmse = np.array([600, 295, 388, 521])

labels = np.array(['Linear Regression', 'Random Forest', 'Decision Tree', 'AdaBoost Tree'])

indices = np.argsort(rmse)

color = plt.cm.spring(np.linspace(0, 1, 9))



plt.style.use('seaborn-talk')

plt.rcParams['figure.figsize'] = (18, 7)



plt.bar(range(len(indices)), rmse[indices], color = color)

plt.xticks(range(len(indices)), labels[indices])

plt.title('RMSE', fontsize = 30)



plt.grid()

plt.tight_layout()

plt.show()