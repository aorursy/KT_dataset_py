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
import matplotlib.pyplot as plt 

import seaborn as sns 

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler 

from scipy import stats

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

# read the data file 

df_train = pd.read_csv('../input/train.csv')
# sales price summary 

df_train['SalePrice'].describe()
# cheeck the column names

df_train.columns
df_train.info()
df_train.head(10)
import pandas_profiling

pandas_profiling.ProfileReport(df_train)


#histogram

sns.distplot(df_train['SalePrice'])
import plotly

import plotly.graph_objs as go 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

init_notebook_mode(connected=True)


trace1 = go.Histogram(

    x=df_train['SalePrice'],

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(171, 50, 96, 0.6)'))



data = [trace1]

layout = go.Layout(barmode='overlay',

                   title=' Housing Sales Price ',

                   xaxis=dict(title='Sales Price'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
print("Skewness: %f" % df_train['SalePrice'].skew()) 

print("Kurtosis: %f" % df_train['SalePrice'].kurt())
#GrLivArea = Above grade (ground) living area square feet

trace2 = go.Histogram(

    x=df_train['GrLivArea'],

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(271, 50, 96, 0.6)'))



data = [trace2]

layout = go.Layout(barmode='overlay',

                   title=' GrLivArea ',

                   xaxis=dict(title='GrLivArea'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace3 =go.Scatter(

                    x = df_train['GrLivArea'],

                    y = df_train['SalePrice'],

                    mode = "markers",

                    name = "GrlivArea",

                    marker = dict(color = 'rgba(255, 128, 255, 0.8)'))



data = [trace3]

layout = dict(title = 'SalePrice Vs GrlivArea',

              xaxis= dict(title= 'GrlivArea',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'SalePrice',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
#TotalBsmtSF= Total square feet of basement area

trace4 = go.Histogram(

    x=df_train['TotalBsmtSF'],

    opacity=0.75,

    name = "2011",

    marker=dict(color='rgba(571, 150, 6, 10.6)'))



data = [trace4]

layout = go.Layout(barmode='overlay',

                   title=' Total Basement Square Feet ',

                   xaxis=dict(title='Total Basement Square Feet'),

                   yaxis=dict( title='Count'),

)

fig = go.Figure(data=data, layout=layout)

iplot(fig)
trace5 =go.Scatter(

                    x = df_train['TotalBsmtSF'],

                    y = df_train['SalePrice'],

                    mode = "markers",

                    name = "GrlivArea",

                    marker = dict(color = 'rgba(25, 12, 255, 0.8)'))



data = [trace5]

layout = dict(title = 'SalePrice Vs TotalBsmtSF',

              xaxis= dict(title= 'TotalBsmtSF',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'SalePrice',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
#OverallQual: Overall material and finish quality

df_train['OverallQual'].describe()
data=[]

for i in range(0,11,1):

    x_i=df_train[df_train['OverallQual'] == i]

    data.append(x_i)

final=[]

for j in range(0,11,1):

    trace_j = go.Box(

    y=data[j]['SalePrice'],

    name = 'OverallQual'+str(j),

    marker = dict(

        color = 'rgb(12, 122, 140)'))

    final.append(trace_j)

iplot(final)   
# Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.

N=10

import random

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

data=[]

for i in range(0,11,1):

    x_i=df_train[df_train['OverallQual'] == i]

    data.append(x_i)

final_two=[]

for j in range(0,11,1):

    trace_j = go.Box(

    y=data[j]['SalePrice'],

    name = 'OverallQual'+str(j),

    marker = dict(color= c[j-5]))

    final_two.append(trace_j)

iplot(final_two)   
#earBuilt: Original construction date

df_train['YearBuilt'].describe()
# Plotly accepts any CSS color format, see e.g. http://www.w3schools.com/cssref/css_colors_legal.asp.

N=138

import random

c = ['hsl('+str(h)+',50%'+',50%)' for h in np.linspace(0, 360, N)]

data=[]

for i in range(1872,2011,1):

    x_i=df_train[df_train['YearBuilt'] == i]

    data.append(x_i)

final_three=[]

for j in range(0,139,1):

    trace_j = go.Box(

    y=data[j]['SalePrice'],

    name = 'YR'+str(j),

    marker = dict(color= c[j-1]))

    final_three.append(trace_j)

iplot(final_three)   


corrmat = df_train.corr()

data = [

    go.Heatmap(

        z=corrmat,

        x=corrmat.columns,

        y=corrmat.columns,

        colorscale='Viridis',

    )

]



iplot(data)

#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df_train[cols].values.T)

cols
data = [

    go.Heatmap(

        z=cm,

        x=cols,

        y=cols,

        colorscale='Viridis',

    )

]



iplot(data)


import plotly.figure_factory as ff

z_text = np.around(cm, decimals=2)

fig = ff.create_annotated_heatmap(z_text)

iplot(fig)
cols
x=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

fig = ff.create_annotated_heatmap(cm,y=x,x=x, annotation_text=z_text, colorscale='Greys', hoverinfo='z')



# Make text size smaller

for i in range(len(fig.layout.annotations)):

    fig.layout.annotations[i].font.size = 8

    

iplot(fig, filename='annotated_heatmap_numpy')
x=['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

fig = ff.create_annotated_heatmap(cm,y=x,x=x, annotation_text=z_text, colorscale='Viridis', hoverinfo='z')



# Make text size smaller

for i in range(len(fig.layout.annotations)):

    fig.layout.annotations[i].font.size = 8

    

iplot(fig, filename='annotated_heatmap_numpy')
trace6 = go.Splom(dimensions=[dict(label='SalePrice',

                                 values=df_train['SalePrice']),

                            dict(label='OverallQual',

                                 values=df_train['OverallQual']),

                            dict(label='GrLivArea',

                                 values=df_train['GrLivArea']),

                            dict(label='GarageCars',

                                 values=df_train['GarageCars']), 

                            dict(label='TotalBsmtSF',

                                 values=df_train['TotalBsmtSF']),

                             dict(label='FullBath',

                                 values=df_train['FullBath']),

                             dict(label='YearBuilt',

                                 values=df_train['YearBuilt'])])
data=[trace6]

iplot(data)
pl_colorscaled = [[0., '#119dff'],

                 [0.5, '#119dff'],

                 [0.5, '#ef553b'],

                 [1, '#ef553b']]



trace7 = go.Splom(dimensions=[dict(label='SalePrice',

                                 values=df_train['SalePrice']),

                            dict(label='OverallQual',

                                 values=df_train['OverallQual']),

                            dict(label='GrLivArea',

                                 values=df_train['GrLivArea']),

                            dict(label='GarageCars',

                                 values=df_train['GarageCars']), 

                            dict(label='TotalBsmtSF',

                                 values=df_train['TotalBsmtSF']),

                             dict(label='FullBath',

                                 values=df_train['FullBath']),

                             dict(label='YearBuilt',

                                 values=df_train['YearBuilt'])], 

                 marker=dict( colorscale=pl_colorscaled,

                              line=dict(width=0.5,

                                        color='rgb(230,230,230)') ))
axis = dict(showline=True,

          zeroline=False,

          gridcolor='#fff',

          ticklen=4)
layout = go.Layout(

    title='Iris Data set',

    dragmode='select',

    width=800,

    height=800,

    autosize=False,

    hovermode='closest',

    plot_bgcolor='rgba(240,240,240, 0.95)',

    xaxis1=dict(axis),

    xaxis2=dict(axis),

    xaxis3=dict(axis),

    xaxis4=dict(axis),

    xaxis5=dict(axis),

    xaxis6=dict(axis),

    xaxis7=dict(axis),

    yaxis1=dict(axis),

    yaxis2=dict(axis),

    yaxis3=dict(axis),

    yaxis4=dict(axis),

    yaxis5=dict(axis),

    yaxis6=dict(axis),

    yaxis7=dict(axis)

)



fig1 = dict(data=[trace7], layout=layout)

iplot(fig1, filename='splom-iris1')
#missing data

total = df_train.isnull().sum().sort_values(ascending=False)

percent = (df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
sns.heatmap(df_train.isnull(), cbar=False)
import missingno as mn

mn.heatmap(df_train)
mn.bar(df_train) 
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...
#standardizing data

saleprice_scaled = StandardScaler().fit_transform(df_train['SalePrice'][:,np.newaxis]);

low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]

high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]

print('outer range (low) of the distribution:')

print(low_range)

print('\nouter range (high) of the distribution:')

print(high_range)




data = [trace1]

layout = dict(title = 'SalePrice Vs GrlivArea',

              xaxis= dict(title= 'GrlivArea',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'SalePrice',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
#deleting points

df_train.sort_values(by = 'GrLivArea', ascending = False)[:2]

df_train = df_train.drop(df_train[df_train['Id'] == 1299].index)

df_train = df_train.drop(df_train[df_train['Id'] == 524].index)
trace9 =go.Scatter(

                    x = df_train['GrLivArea'],

                    y = df_train['SalePrice'],

                    mode = "markers",

                    name = "GrlivArea",

                    marker = dict(color = 'rgba(155, 128, 255, 0.8)'))



data = [trace9]

layout = dict(title = 'SalePrice Vs GrlivArea',

              xaxis= dict(title= 'GrlivArea',ticklen= 5,zeroline= False),

              yaxis= dict(title= 'SalePrice',ticklen= 5,zeroline= False)

             )

fig = dict(data = data, layout = layout)

iplot(fig)
from plotly import tools

fig = tools.make_subplots(rows=1, cols=2)

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace9, 1, 2)



fig['layout'].update(height=600, width=800, title='Before and After Deleting Outliers')

iplot(fig, filename='simple-subplot-with-annotations')
#histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)
import plotly.figure_factory as ff
#applying log transformation

df_train['SalePrice'] = np.log(df_train['SalePrice'])

#transformed histogram and normal probability plot

sns.distplot(df_train['SalePrice'], fit=norm);

fig = plt.figure()

res = stats.probplot(df_train['SalePrice'], plot=plt)


from statsmodels.graphics.gofplots import qqplot



qqplot_data = qqplot(df_train['SalePrice'], line='s').gca().lines



qqplot_data[1].get_xdata()
fig = go.Figure()



fig.add_trace({

    'type': 'scatter',

    'x': qqplot_data[0].get_xdata(),

    'y': qqplot_data[0].get_ydata(),

    'mode': 'markers',

    'marker': {

        'color': '#19d3f3'

    }

})



fig.add_trace({

    'type': 'scatter',

    'x': qqplot_data[1].get_xdata(),

    'y': qqplot_data[1].get_ydata(),

    'mode': 'lines',

    'line': {

        'color': '#636efa'

    }



})





fig['layout'].update({

    'title': 'Quantile-Quantile Plot',

    'xaxis': {

        'title': 'Theoritical Quantities',

        'zeroline': False

    },

    'yaxis': {

        'title': 'Sample Quantities'

    },

    'showlegend': False,

    'width': 800,

    'height': 700,

})





iplot(fig, filename='normality-QQ')
cat_cols = ['GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']

train_y = df_train['SalePrice']

train_df= pd.concat([df_train['GrLivArea'], df_train['GarageCars'], 

                 df_train['GarageArea'],

                 df_train['TotalBsmtSF'],

                 df_train['FullBath'],

                 df_train['TotRmsAbvGrd'],

                 df_train['YearBuilt']], axis=1)

from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, 

                                     max_features=0.3, n_jobs=-1, random_state=0)



model.fit(train_df, train_y)
feat_names = train_df.columns.values
## plot the importances ##

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
#dealing with missing data

df_train = df_train.drop((missing_data[missing_data['Total'] > 1]).index,1)

df_train = df_train.drop(df_train.loc[df_train['Electrical'].isnull()].index)

df_train.isnull().sum().max() #just checking that there's no missing data missing...



# select only numerical features 

df3 = df_train.select_dtypes(include = ['int64', 'float64'])

print(df3)
y = df3['SalePrice']
x= df3.drop (['SalePrice'], axis=1)
from sklearn import ensemble

model = ensemble.ExtraTreesRegressor(n_estimators=25, max_depth=30, 

                                     max_features=0.3, n_jobs=-1, random_state=0)



model.fit(x,y)
feat_names = x.columns.values

## plot the importances ##

importances = model.feature_importances_

std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

indices = np.argsort(importances)[::-1][:20]



plt.figure(figsize=(12,12))

plt.title("Feature importances")

plt.bar(range(len(indices)), importances[indices], color="r", yerr=std[indices], align="center")

plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')

plt.xlim([-1, len(indices)])

plt.show()
print(importances)