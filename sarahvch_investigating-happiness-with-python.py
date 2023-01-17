import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.datasets import fetch_20newsgroups_vectorized

from sklearn.feature_selection import chi2

from sklearn.feature_selection import RFE

from sklearn.ensemble import ExtraTreesClassifier

from sklearn import datasets

from sklearn import metrics

import types

from sklearn.manifold import TSNE

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

%matplotlib inline





import seaborn as sns

sns.set(style="whitegrid", palette="muted")

current_palette = sns.color_palette()

df = pd.read_csv("../input/2015.csv")

df.head(2)
sns.distplot(df['Happiness Score'])
corrmat = df.corr()

sns.heatmap(corrmat, vmax=.8, square=True)
g = sns.stripplot(x="Region", y="Happiness Rank", data=df, jitter=True)

plt.xticks(rotation=90)
data = dict(type = 'choropleth', 

           locations = df['Country'],

           locationmode = 'country names',

           z = df['Happiness Rank'], 

           text = df['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Global Happiness', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [data], layout=layout)

iplot(choromap3)
y = df['Happiness Score']

X = df.drop(['Happiness Score', 'Happiness Rank', 'Country', 'Region'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
predictions = lm.predict( X_test)
plt.scatter(y_test,predictions)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
from sklearn import metrics



print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients