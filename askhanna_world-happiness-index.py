# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/2017.csv')
df.info()
df.head()
df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
fig = plt.figure(figsize=(16,10))
sns.heatmap(data=df.iloc[:,2:].corr(), annot=True)
df = df.drop(['Whisker.high','Whisker.low'], axis=1)
print(df.columns)
fig,ax = plt.subplots(3,2, figsize=(16,10), sharex='col',sharey='row')
nrows,ncols = 3,2
counter=3
for i in range(nrows):
    for j in range(ncols):
        ax[i][j].scatter(y=df['Happiness.Score'],x=df.iloc[:,counter])
        #sns.jointplot(data = df, x=df.columns[counter], y ='Happiness.Score',fig=)
        ax[i][j].set_xlabel(df.columns[counter])
        ax[i][j].set_ylabel('Happiness.Score')
        counter= counter + 1
#plt.scatter(x=df['Freedom'], y=df['Happiness.Score'])
fig,ax = plt.subplots(3,2, figsize=(16,10), sharex='col',sharey='row')
nrows,ncols = 3,2
counter=3
for i in range(nrows):
    for j in range(ncols):
        #ax[i][j].scatter(y=df['Happiness.Score'],x=df.iloc[:,counter])
        sns.distplot(df[df.columns[counter]],ax=ax[i][j])
        ax[i][j].set_xlabel(df.columns[counter])
        ax[i][j].set_ylabel('Happiness.Score')
        counter= counter + 1
#sns.distplot(df['Economy..GDP.per.Capita.'])
features = ['Economy..GDP.per.Capita.', 'Family', 'Health..Life.Expectancy.','Freedom', 'Generosity', 'Trust..Government.Corruption.','Dystopia.Residual']
target = 'Happiness.Score'
for i in features:
    print(i)
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
data = dict(type = 'choropleth', 
           locations = df['Country'],
           locationmode = 'country names',
           z = df['Happiness.Rank'], 
           text = df['Country'],
           colorbar = {'title':'Happiness'})
layout = dict(title = 'Global Happiness', 
             geo = dict(showframe = False, 
                       projection = {'type': 'Mercator'}))
choromap3 = go.Figure(data = [data], layout=layout)
iplot(choromap3)
df_shuffle = df.reindex(np.random.permutation(df.index))
df_shuffle.head()
X = df[features]
y = df[target]
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
lm = LinearRegression()
lm.fit(X_train,y_train)
y_pred = lm.predict(X_test)
print('MAE:', mean_absolute_error(y_test, y_pred))
print('MSE:', mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred)))
coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
coefficients