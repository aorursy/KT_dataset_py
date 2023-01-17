import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

%matplotlib inline
## Read file





data = pd.read_csv('../input/googleplaystore.csv')

x=data

data.head(3)
data.info()
data.shape
import pandas_profiling

pandas_profiling.ProfileReport(data)
data.isnull().sum()
data[data['Rating'] == 19]
data.iloc[10472,1:] = data.iloc[10472,1:].shift(1)

data[10471:10473]
data["Last Updated"] = pd.to_datetime(data['Last Updated'])

data['year_added']=data['Last Updated'].dt.year

data['month_added']=data['Last Updated'].dt.month
data.head(2)
data.columns
import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

col = "Type"

grouped = data[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])

layout = {'title': 'Target(0 = No, 1 = Yes)'}

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
d1=x[x['Type']=='Free']

d2=x[x['Type']=='Paid']
col='year_added'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"app udated or added over the years",'xaxis':{'title':"years"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='month_added'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name="Free", marker=dict())

layout={'title':"Free App added over the month",'xaxis':{'title':"months"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
col='month_added'

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v2[col], y=v2["count"], name="aid", marker=dict())

layout={'title':"Paid App added over the month",'xaxis':{'title':"months"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)


col='Content Rating'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.astype(str).sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Ratings of the free vs paid app",'xaxis':{'title':"Ratings"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Content Rating'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name="Free", marker=dict())

layout={'title':"Free App Content Rating ",'xaxis':{'title':"Contents"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
col='Content Rating'

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v2[col], y=v2["count"], name="aid",  marker=dict(color="#6ad49b"))

layout={'title':"Paid App Content Rating",'xaxis':{'title':"contents"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)


col='Rating'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.astype(str).sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Ratings of the free vs paid app",'xaxis':{'title':"Ratings"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Rating'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

trace1 = go.Bar(x=v1[col], y=v1["count"], name="Free", marker=dict())

layout={'title':"Free App Rating",'xaxis':{'title':"Ratings"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)
col='Rating'

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v2[col], y=v2["count"], name="Paid",  marker=dict(color="#6ad49b"))

layout={'title':"Paid App Rating",'xaxis':{'title':"Ratingss"}}

fig = go.Figure(data=[trace1], layout=layout)

iplot(fig)


col='Category'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"App Category"}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Android Ver'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Android Versions"}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Installs'

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="Free", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="Paid", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Installed App ",'xaxis':{'title':"Installs"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
d3=x[x['Rating']==4.5]

d4=x[x['Rating']==4]
col='Content Rating'

v1=d3[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d4[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name="rating = 4.5", marker=dict(color="#6ad49b"))

trace2 = go.Bar(x=v2[col], y=v2["count"], name="rating = 4", marker=dict())

y = [trace1, trace2]

layout={'title':"Rating over the contents",'xaxis':{'title':"Content Rating"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Android Ver'

v1=d3[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d4[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="rating = 4.5", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="rating = 4", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Rating over the Android Version "}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Category'

v1=d3[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d4[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name="rating = 4.5", marker=dict(color="#a678de"))

trace2 = go.Bar(x=v2[col], y=v2["count"], name="rating = 4", marker=dict())

y = [trace1, trace2]

layout={'title':"Category wise Rating"}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
col='Installs'

v1=d3[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d4[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name="rating = 4.5", marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name="rating = 4", marker=dict(color="#6ad49b"))

y = [trace1, trace2]

layout={'title':"Rating over total Installs ",'xaxis':{'title':"Installs"}}

fig = go.Figure(data=y, layout=layout)

iplot(fig)
data.isnull().sum().sum()
total=data.isnull().sum()

percent = (data.isnull().sum()/data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(13)
data.dropna(inplace=True)
data.shape
data.head(3)
catgry=pd.get_dummies(data['Category'],prefix='catg',drop_first=True)

typ=pd.get_dummies(data['Type'],prefix='typ',drop_first=True)

cr=pd.get_dummies(data['Content Rating'],prefix='cr',drop_first=True)

frames=[data,catgry,typ,cr]

data=pd.concat(frames,axis=1)

data.drop(['Category','Installs','Type','Content Rating'],axis=1,inplace=True)
data.drop(['App','Size','Price','Genres','Last Updated','Current Ver','Android Ver'],axis=1,inplace=True)
data.head(3)
X=data.drop('Rating',axis=1)

y=data['Rating'].values

y=y.astype('int')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
from sklearn.preprocessing import StandardScaler

sc_X=StandardScaler()

X_train=sc_X.fit_transform(X_train)

X_test=sc_X.transform(X_test)
#LogisticRegression

lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

lr_ac=accuracy_score(y_test, lr_pred)

print('LogisticRegression_accuracy:',lr_ac)
plt.figure(figsize=(10,5))

plt.title("lr_cm")

sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()
# DecisionTree Classifier

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(X_train,y_train)

dtree_pred=dtree_c.predict(X_test)

dtree_cm=confusion_matrix(y_test,dtree_pred)

dtree_ac=accuracy_score(dtree_pred,y_test)
plt.figure(figsize=(10,5))

plt.title("dtree_cm")

sns.heatmap(dtree_cm,annot=True,fmt="d",cbar=False)

print('DecisionTree_Classifier_accuracy:',dtree_ac)
#SVM regressor

svc_r=SVC(kernel='rbf')

svc_r.fit(X_train,y_train)

svr_pred=svc_r.predict(X_test)

svr_cm=confusion_matrix(y_test,svr_pred)

svr_ac=accuracy_score(y_test, svr_pred)
plt.figure(figsize=(10,5))

plt.title("svm_cm")

sns.heatmap(svr_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)

print('SVM_regressor_accuracy:',svr_ac)
#RandomForest

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

rdf_ac=accuracy_score(rdf_pred,y_test)
plt.figure(figsize=(10,5))

plt.title("rdf_cm")

sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

print('RandomForest_accuracy:',rdf_ac)
%matplotlib inline

model_accuracy = pd.Series(data=[lr_ac,dtree_ac,svr_ac,rdf_ac], 

        index=['Logistic_Regression','DecisionTree_Classifier','SVM_regressor_accuracy','RandomForest'])

fig= plt.figure(figsize=(8,8))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')