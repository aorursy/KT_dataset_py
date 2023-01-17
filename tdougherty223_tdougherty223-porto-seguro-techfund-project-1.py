# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import missingno as msno



train=pd.read_csv('../input/pstrain.csv')

train.head()

rows=train.shape[0]

columns=train.shape[1]

print("The dataset contains {0} rows and {1} columns".format(rows, columns))



# We can use Gradient Boosting for feature inspection and generate a feature importance scatter plot



from sklearn.feature_selection import mutual_info_classif

from sklearn.ensemble import GradientBoostingClassifier

import warnings

warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

import plotly.graph_objs as go

import plotly.tools as tls

import warnings

py.init_notebook_mode(connected=True)





gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, min_samples_leaf=4, max_features=0.2, random_state=0)

gb.fit(train.drop(['id', 'target'],axis=1), train.target)

features = train.drop(['id', 'target'],axis=1).columns.values





trace = go.Scatter(y = gb.feature_importances_, x = features,mode='markers',

    marker=dict(sizemode = 'diameter', sizeref = 1, size = 13, color = gb.feature_importances_, colorscale='Portland', showscale=True),

    text = features)

data = [trace]



layout= go.Layout(autosize= True, title= 'Gradient Boosting Machine Feature Importance',hovermode= 'closest',

     xaxis= dict(ticklen= 5, showgrid=False, zeroline=False,showline=False),

     yaxis=dict(title= 'Feature Importance', showgrid=False, zeroline=False, ticklen= 5, gridwidth= 2),

     showlegend= False)

fig = go.Figure(data=data, layout=layout)

py.iplot(fig, filename='GB Feature Importance Plot')







train_copy=train

train_copy = train_copy.replace(-1, np.NaN)

# Checking null values by column using "Missingno" package.

msno.matrix(df=train_copy.iloc[:,2:39], figsize=(20, 14), color=(0.42, 0.1, 0.05))



# Need to replace missing values.

train.drop(["id", "target"], axis = 1, inplace = True)

na_count = train.isnull().sum()

na_columns = list(na_count[na_count>0].index.values)

train_no_missing = train.drop(na_columns, axis = 1)

cat_columns_no_missing = list(filter(lambda x: x.endswith("cat"), train_no_missing.columns.values))

train_no_missing_oh = pd.get_dummies(train_no_missing, columns = cat_columns_no_missing)



# Cluster remaining columns using KMeans

from sklearn.cluster import MiniBatchKMeans

kmeans = MiniBatchKMeans(n_clusters = 15, random_state = 0, batch_size = 2000)

kmeans.fit(train_no_missing_oh)

print("Clustersize: \n")

print(pd.Series(kmeans.labels_).value_counts())

train["cluster"] = kmeans.labels_



# Replace missing values from each cluster with median or most common value.

replace_missing_values=pd.DataFrame()

for i in na_columns:

    clean_df=train[["cluster",i]].dropna()

    if i.endswith("cat"):

        replace_missing_values[i]=clean_df.groupby(["cluster"]).agg(lambda x:x.value_counts().index.values[0])

    else:

        replace_missing_values[i]=clean_df.groupby(["cluster"]).median()

        

for cl, cat in ((x, y) for x in range(15) for y in na_columns):

    train.loc[(train["cluster"] == cl) & pd.isnull(train[cat]), cat] = reaplace_missing_values.loc[cl, cat]

    

print("\n remaining missing values: " + str(train.isnull().sum().sum()))





# Checking to see which columns only contain binary data.

binarycol=[col for col in train.columns if '_bin' in col]

zeroslist=[]

oneslist=[]

for col in binarycol:

    zeroslist.append((train[col]==0).sum())

    oneslist.append((train[col]==1).sum())



zerotrace=go.Bar(x=binarycol, y=zeroslist, name='Number of Zeros')

onestrace=go.Bar(x=binarycol, y=oneslist, name='Number of Ones')

data=[zerotrace, onestrace]

layout=go.Layout(barmode='group', title='Count of 0 and 1 in binary variables')

fig=go.Figure(data=data, layout=layout)

py.iplot(fig, filename='binary-grouped-bar')














