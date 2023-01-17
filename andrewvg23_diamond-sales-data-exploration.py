import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import hypertools as hyp

import plotly.offline as py



from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn import linear_model



py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from sklearn.feature_selection import mutual_info_classif
%matplotlib inline
from subprocess import check_output

print(check_output(["ls", "../input/diamonds.csv"]).decode("utf8"))
dmnd = pd.read_csv('../input/diamonds.csv', low_memory=False, index_col=0)
dmnd.shape
dmnd.head()
plt.figure(figsize=(10,3))

plt.scatter(range(dmnd.shape[0]), np.sort(dmnd['carat'].values), alpha=.4)

plt.title('Sorted Carat Scatterplot', fontsize=15)

plt.xlabel('index', fontsize=13)

plt.ylabel('carats', fontsize=13)

plt.show()
plt.figure(figsize=(6,3))

sns.distplot(dmnd['carat'], bins=25, kde=False)

plt.title('Carat Distribution', fontsize=15)

plt.xlabel('carat', fontsize=13)

plt.ylabel('count', fontsize=13)

plt.show()
val_count = dmnd['color'].value_counts()



plt.figure(figsize=(6,3))

sns.barplot(['D','E','F','G','H','I','J'],

            [val_count.values[4], val_count.values[1], val_count.values[2],

             val_count.values[0], val_count.values[3], val_count.values[5], val_count.values[6]], alpha=0.8)

plt.title('Color Distribution', fontsize=15)

plt.xticks(rotation='vertical')

plt.xlabel('Color', fontsize=13)

plt.ylabel('Number of Occurrences', fontsize=13)

plt.show()
val_count = dmnd['cut'].value_counts()



plt.figure(figsize=(6,3))

sns.barplot(val_count.index, val_count.values, alpha=0.8)

plt.title('Distribution of Cut Quality', fontsize=15)

plt.xticks(rotation='vertical')

plt.xlabel('Cut', fontsize=13)

plt.ylabel('Number of Occurrences', fontsize=13)

plt.show()
val_count = dmnd['clarity'].value_counts()

val_count.index
val_count = dmnd['clarity'].value_counts()



plt.figure(figsize=(6,3))

sns.barplot(val_count.index, val_count.values, alpha=0.8, order=['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1'])

plt.title('Clarity Distribution', fontsize=15)

plt.xticks(rotation='vertical')

plt.xlabel('Clarity', fontsize=13)

plt.ylabel('Number of Occurrences', fontsize=13)

plt.show()
plt.figure(figsize=(6,3))

sns.distplot(dmnd['price'], bins=30, kde=False)

plt.title('Price Distribution', fontsize=15)

plt.xlabel('Price($)', fontsize=13)

plt.ylabel('count', fontsize=13)

plt.show()
plt.figure(figsize=(6,3))

sns.distplot(dmnd['depth'], bins=25, kde=False)

plt.title('Depth Distribution', fontsize=15)

plt.xlabel('Depth', fontsize=13)

plt.ylabel('count', fontsize=13)

plt.show()
color = []

cut = []

clarity = []



color = dmnd['color'].map({'D':1, 'E':2, 'F':3, 'G':4, 'H':5, 'I':6, 'J':7})

cut = dmnd['cut'].map({'Ideal':1, 'Premium':2, 'Very Good':3, 'Good':4, 'Fair':5})

clarity = dmnd['clarity'].map({'IF':1,'VVS1':2,'VVS2':3,'VS1':4,'VS2':5,'SI1':6,'SI2':7,'I1':8})
dmnd['color']=color

dmnd['cut']=cut

dmnd['clarity']=clarity
colormap = plt.cm.pink

plt.figure(figsize=(16,12))

plt.title('Correlations of the Features of a Diamond', y=1.05, size=15)

sns.heatmap(dmnd.corr(),linewidths=0.1,vmax=1.0, square=True, 

            cmap=colormap, linecolor='black', annot=True)

plt.show()
rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rf.fit(dmnd.drop(['price'],axis=1), dmnd['price'])

features = dmnd.drop(['price'],axis=1).columns.values
graph = go.Scatter(y = rf.feature_importances_,x = features,mode='markers',

    

    marker=dict(

        sizemode = 'diameter',

        size = 15,

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    )

)



data = [graph]



layout = dict(autosize=False, title='Random Forest Feature Importance', hovermode='closest')



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)

plt.show()
rf = RandomForestRegressor(n_estimators=150, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rf.fit(dmnd.drop(['price','carat','x','y','z'],axis=1), dmnd['price'])

features = dmnd.drop(['price','carat','x','y','z'],axis=1).columns.values
graph = go.Scatter(y = rf.feature_importances_,x = features,mode='markers',

    

    marker=dict(

        sizemode = 'diameter',

        size = 15,

        color = rf.feature_importances_,

        colorscale='Portland',

        showscale=True

    )

)



data = [graph]



layout = dict(autosize=False, title='Random Forest Feature Importance', hovermode='closest')



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)

plt.show()
sns.kdeplot(dmnd['price'],dmnd['carat'], shade=True, clip=(0,6000), shade_lowest=False).set(ylim=(0, 1.25))

plt.show()
hyp.plot(dmnd[['color','cut','clarity']],'.',n_clusters=8,animate="spin")

plt.show()
X_train, X_test, y_train, y_test = train_test_split(dmnd.drop(['price'],axis=1), dmnd['price'], test_size=0.2, random_state=42)
X_train.iloc[0]
rf = RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_leaf=4, max_features=0.2, n_jobs=-1, random_state=0)

rf.fit(X_train, y_train)

rf.score(X_test,y_test)



X_test.head()
#X = [1,1,1,1,1]

#rf.predict([[, 2,0,0, 0], [0,7,7, 7, 3]])
ridge = linear_model.Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

ridge.score(X_test, y_test)
linear = linear_model.LinearRegression()

linear.fit(X_train, y_train)

linear.score(X_test, y_test)
#lg = linear_model.LogisticRegression()

#lg.fit(X_train, y_train)

#lg.score(X_test, y_test)