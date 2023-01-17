# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/abalone-dataset/abalone.csv'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/abalone-dataset/abalone.csv')

df.head()
print(df.describe())
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import pandas_profiling
df.profile_report()
df.corr().style.background_gradient(cmap='cool')
# dropping those that are highly correlated with one another 

df.head()
import plotly as ply

import plotly.graph_objs as go
trace1= go.Scatter(x = df.Diameter,

                    y = df.Height,

                    mode = "markers",

                    name = "Height",

                    marker = dict(color = 'blue'),

                    text= df.Sex)

layout = dict(title = 'Diameter v Height',

              xaxis= dict(title= 'Diameter ',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Height',ticklen= 5,zeroline= False)

             )

data= [trace1]

fig = dict(data = data, layout = layout)
from plotly.offline import *

ply.offline.iplot(fig)
#Dropping the outliers at the top 



df = df.drop(df[df.Height > 0.5].index)

df.shape



trace2= go.Scatter(x = df.Diameter,

                    y = df.Height,

                    mode = "markers",

                    name = "Height",

                    marker = dict(color = 'blue'),

                    text= df.Sex)

layout = dict(title = 'Diameter v Height',

              xaxis= dict(title= 'Diameter ',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'Height',ticklen= 5,zeroline= False)

             )

data= [trace2]

fig = dict(data = data, layout = layout)

ply.offline.iplot(fig)
df_ns = df.drop(columns = ['Sex'])

df_d = pd.get_dummies(df['Sex'],drop_first=True)

df_ns.reset_index(inplace=True, drop=True)

df_d.reset_index(inplace=True, drop=True)

df_combined= pd.concat([df_ns, df_d], axis=1)

df_combined.head()
from sklearn.model_selection import train_test_split

x = df_combined[['Diameter','Height','I','M','Shucked_weight']]

y = df_combined[['Rings']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score, mean_squared_error
lr = LinearRegression()

y_fit = lr.fit(x_train, y_train)

y_pred = y_fit.predict(x_test)

y_pred = pd.DataFrame(y_pred)

y_pred.columns= ['Predicted']



y_pred.reset_index(drop=True, inplace=True)

y_test.reset_index(drop=True, inplace=True)



print('the r2 is ', y_fit.score(x_train,y_train))

rmse= np.sqrt(mean_squared_error(y_pred,y_test))

print('the rmse is ', rmse)



comb = pd.concat([y_pred.round().astype(int), y_test], axis=1)

comb['error'] = (y_pred['Predicted']-y_test['Rings'])/y_test['Rings']

comb
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=2)

knn = knn.fit(x_train, y_train) 



y_pred = knn.predict(x_test)

y_pred1= pd.DataFrame(y_pred)



y_pred1.columns= ['Predicted']



print('the r2 is ', knn.score(x_train,y_train))

rmse= np.sqrt(mean_squared_error(y_pred1,y_test))

print('the rmse is ', rmse)



combined_df_knn= pd.concat([y_pred1, y_test], axis=1)

combined_df_knn['error for knn'] = (y_pred1['Predicted']-y_test['Rings'])/y_test['Rings']

combined_df_knn