# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print("folder names: ")

print(check_output(["ls", "../input"]).decode("utf8"))



print("files (corruption-index): ")

print(check_output(["ls", "../input/corruption-index"]).decode("utf8"))



print("files (world-happiness): ")

print(check_output(["ls", "../input/world-happiness"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import seaborn as sns 

import matplotlib as mpl 

import plotly.graph_objs as go 

from plotly.graph_objs import *

from matplotlib import pyplot as plt 

from plotly.offline import iplot, init_notebook_mode





sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()

init_notebook_mode(connected=True)

%matplotlib inline 
df_15 = pd.read_csv('../input/world-happiness/2015.csv')

df_15.head()
print(df_15.dtypes)

print(df_15.shape)
df_15.describe()
sns.distplot(df_15['Happiness Score'])
corrmat = df_15.corr() 

corrmat
sns.heatmap(corrmat, square=True)
# choropleth map across all countries 

scl = [[0.0, 'rgb(242,240,247)'],[0.2, 'rgb(218,218,235)'],[0.4, 'rgb(188,189,220)'],\

            [0.6, 'rgb(158,154,200)'],[0.8, 'rgb(117,107,177)'],[1.0, 'rgb(84,39,143)']]



data = dict(

    type='choropleth',

    colorscale = scl,

    autocolorscale = False,

    locations = df_15['Country'],

    locationmode = 'country names',

    z = df_15['Happiness Score'],

    text = df_15['Country'],

    colorbar = {'title':'Happiness Score'}

)



layout = dict(

    title = 'Global Happiness Score',

    geo = dict(

        showframe = False,

        showcoastlines = False,

        projection = {'type':'Mercator'}

    )

)



cmap = go.Figure(data=[data], layout=layout)

iplot(cmap)
# let's look at regional level 

print(df_15.groupby('Region')['Happiness Rank'].nunique())

print("Total countries: {} countries".format(df_15.shape[0]))
# let's look at Southeast Asia

# df_sea = df_15[df_15['Region']=='Southeastern Asia']

# print("Total countries in Southeastern Asia: {}".format(df_sea.shape[0]))

# print([country for country in df_sea['Country']])
# CPI: 100 (very clean) to 0 (highly corrupt)

corr_df = pd.read_csv('../input/corruption-index/index.csv')

corr_df.head()
print(corr_df.dtypes)

print(corr_df.shape)
# filter based on the countries exists in df_sea 

# i1 = corr_df.set_index(['Country']).index

# i2 = df_sea.set_index(['Country']).index

# corr_country_df = corr_df[i1.isin(i2)]

df = corr_df.merge(df_15, left_on='Country', right_on='Country')

print("Countries that are somehow excluded: \n{}".format(set(corr_df['Country']).symmetric_difference(set(df_15['Country']))))
# let's sanitize the data little 



nan_cols = df.columns[df.isnull().all()].tolist()

print("Columns with all NaNs: \n{}".format(nan_cols))

df.drop(nan_cols, axis=1, inplace=True)

df.fillna(0, inplace=True)



corrmat = df.corr()

sns.heatmap(corrmat, square=True)


from sklearn.model_selection import KFold, train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression 

from sklearn import metrics



X = df.drop(['Happiness Score','Country','Region_x','Region_y','Country Code','Happiness Rank','CPI Rank'],axis=1)

y = df['Happiness Score']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lm = LinearRegression()

model = lm.fit(X_train, y_train)
# how correlated are the explanatory variables to the dependent variable

predictions = lm.predict(X_test)

coef = pd.DataFrame(lm.coef_, X.columns)

coef.columns = ['Coefficients']

print(coef)
mse = metrics.mean_squared_error(y_test, predictions)

variance = metrics.r2_score(y_test, predictions)

score = model.score(X_test, y_test)

print("\nMean Squared Error: {}\nVariance: {}\nScore: {}".format(mse, variance, score))