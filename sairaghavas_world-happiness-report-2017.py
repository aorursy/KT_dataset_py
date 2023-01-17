import numpy as np 

 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)



data_h = pd.read_csv("../input/2017.csv")

top_10 = data_h.loc[0:9,]

least_10 = data_h.loc[-9:]

top_10.head()
plt.figure(figsize=[13,5])

sns.barplot('Country','Happiness.Score', data = top_10)
df = dict(type = 'choropleth', 

           locations = data_h['Country'],

           locationmode = 'country names',

           z = data_h['Happiness.Rank'], 

           text = data_h['Country'],

           colorbar = {'title':'Happiness'})

layout = dict(title = 'Global.Happiness', 

             geo = dict(showframe = False, 

                       projection = {'type': 'Mercator'}))

choromap3 = go.Figure(data = [df], layout=layout)

iplot(choromap3)
# Plotting heatmap of pearson's correlation for 2016

fig, axes = plt.subplots(figsize=(10, 7))

corr = data_h.corr()

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True

with sns.axes_style("white"):

    ax = sns.heatmap(corr,linewidths=1,annot=True, mask=mask, vmax=.3, square=True)

axes.set_title("2017")
list(data_h)

data_h = data_h.rename(columns = {'Economy..GDP.per.Capita.': 'Economy/PCI'})

data_h = data_h.rename(columns = {'Health..Life.Expectancy.': 'Health'})

data_h = data_h.rename(columns = {'Trust..Government.Corruption.': 'Govt_corruption'})

sns.pairplot(data_h[['Happiness.Score','Economy/PCI','Family','Health', 'Dystopia.Residual']])
y = data_h['Happiness.Score']

X = data_h.drop(['Happiness.Score', 'Happiness.Rank', 'Country'], axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
print('Coefficients: \n', lm.coef_)
coeffecients = pd.DataFrame(lm.coef_,X.columns)

coeffecients.columns = ['Coeffecient']

coeffecients