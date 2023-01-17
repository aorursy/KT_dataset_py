import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls
data = pd.read_csv('../input/kc_house_data.csv')
# Lets check it out 

data.head()
data.isnull().sum()
print(data.info())

print("**"*40)

print(data.describe())
data['date'] = pd.to_datetime(data['date'])

# while im at it, let me create a year and month column too

data['year'], data['month'] = data['date'].dt.year, data['date'].dt.month
# as we have everything from the date column, lets simply remove it 

del data['date']
plt.figure(figsize=(12,12))

sns.jointplot( 'long','lat',data = data, size=9 , kind = "hex")

plt.xlabel('Longitude', fontsize=10)

plt.ylabel('Latitude', fontsize=10)

plt.show()
dataa = [

    go.Heatmap(

        z= data.corr().values,

        x= data.columns.values,

        y= data.columns.values,

        colorscale='Viridis',

        text = True ,

        opacity = 1.0

        

    )

]



layout = go.Layout(

    title='Pearson Correlation',

    xaxis = dict(ticks='', nticks=30),

    yaxis = dict(ticks='' ),

    width = 800, height = 600,

    

)



fig = go.Figure(data=dataa, layout=layout)

py.iplot(fig, filename='Housedatacorr')
# the models we will run

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.ensemble import ExtraTreesRegressor



# some metrics to help us out

from sklearn.cross_validation import cross_val_score

from sklearn.cross_validation import train_test_split

from sklearn.metrics import mean_squared_error as mse
target = data['price']

# we dont need the price column in data anymore

del data['price']
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
dr = DecisionTreeRegressor()

dr.fit(X_train,y_train)

drimp = dr.feature_importances_
rfr = RandomForestRegressor(n_estimators=100)

rfr.fit(X_train,y_train)

rfrimp = rfr.feature_importances_
gbr =  GradientBoostingRegressor(n_estimators=100)

gbr.fit(X_train,y_train)

gbrimp = gbr.feature_importances_
abr =  AdaBoostRegressor(n_estimators=100)

abr.fit(X_train,y_train)

abrimp = abr.feature_importances_
etr =  ExtraTreesRegressor(n_estimators=100)

etr.fit(X_train,y_train)

etrimp = etr.feature_importances_
d = {'Decision Tree':drimp, 'Random Forest':rfrimp, 'Gradient Boost':gbrimp,'Ada boost':abrimp, 'Extra Tree':etrimp}
features = pd.DataFrame(data = d)

# lets check out features

features.head()
features['mean'] = features.mean(axis= 1) 

# we forgot to add the names of the features

features['names'] = data.columns.values
#lets check it out now 

features.head()
y = features['mean'].values

x = features['names'].values

data = [go.Bar(

            x= x,

             y= y,

            width = 0.5,

            marker=dict(

               color = features['mean'].values,

            colorscale='Portland',

            showscale=True,

            reversescale = False

            ),

            opacity=0.6

        )]



layout= go.Layout(

    autosize= True,

    title= 'Mean Feature Importance',

    hovermode= 'closest',

    yaxis=dict(

        title= 'Feature Importance for Housing Price',

        ticklen= 5,

        gridwidth= 2

    ),

    showlegend= False

)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='barplothouse')