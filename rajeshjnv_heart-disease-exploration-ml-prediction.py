import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph.

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score

from sklearn.model_selection import cross_val_score

from sklearn.metrics import precision_recall_curve

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import roc_curve

from sklearn.metrics import f1_score

from sklearn.metrics import auc

from sklearn.svm import SVC

%matplotlib inline
df=pd.read_csv('../input/heart.csv')

df.head(5)
df.describe()
df.shape
import pandas_profiling

x=pandas_profiling.ProfileReport(df)

x
df.isnull().sum()
%matplotlib inline

plt.figure(figsize=(10,10))

sns.heatmap(df.corr(),annot=True,fmt='.1f')

plt.show()
df_corr=df.corr()['target'][:-1]

feature_list=df_corr[abs(df_corr)>0.1].sort_values(ascending=False)

feature_list
sns.distplot(df['target'],rug=True)

plt.show()
import plotly

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

col = "target"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])

layout = {'title': 'Target(0 = No, 1 = Yes)'}

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
sns.distplot(df['sex'],rug=True)

plt.show()
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

col = "sex"

grouped = df[col].value_counts().reset_index()

grouped = grouped.rename(columns = {col : "count", "index" : col})



## plot

trace = go.Pie(labels=grouped[col], values=grouped['count'], pull=[0.05, 0])

layout = {'title': 'Male(1), Female(0)'}

fig = go.Figure(data = [trace], layout = layout)

iplot(fig)
col='sex'

d1=df[df['target']==0]

d2=df[df['target']==1]

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name=0, marker=dict(color="#a678de"))

trace2 = go.Bar(x=v2[col], y=v2["count"], name=1, marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout={'title':"target over the sex(male or female)",'xaxis':{'title':"target"}}

#layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

iplot(fig)

sns.distplot(df['cp'],rug=True)

plt.show()
col='cp'

d1=df[df['target']==0]

d2=df[df['target']==1]

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name=0)

trace2 = go.Bar(x=v2[col], y=v2["count"], name=1)

data = [trace1, trace2]

layout={'title':"target over the chaist pain",'xaxis':{'title':"Chaist pain type"}}

#layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

iplot(fig)

sns.distplot(df['thalach'],rug=True)

plt.show()
col='thalach'

d1=df[df['target']==0]

d2=df[df['target']==1]

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name=0, marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name=1, marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout={'title':"target over the person's maximum heart rate achieved"}

fig = go.Figure(data, layout=layout)

iplot(fig)
sns.distplot(df['fbs'],rug=True)

plt.show()
col='fbs'

d1=df[df['target']==0]

d2=df[df['target']==1]

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Bar(x=v1[col], y=v1["count"], name=0, marker=dict(color="#a678de"))

trace2 = go.Bar(x=v2[col], y=v2["count"], name=1, marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout={'title':"target over the person's fasting blood sugar ",'xaxis':{'title':"fbs(> 120 mg/dl, 1 = true; 0 = false)"}}

#layout = go.Layout(title="Content added over the years", legend=dict(x=0.1, y=1.1, orientation="h"))

fig = go.Figure(data, layout=layout)

iplot(fig)
sns.distplot(df['age'],rug=True)

plt.show()
col='age'

d1=df[df['target']==0]

d2=df[df['target']==1]

v1=d1[col].value_counts().reset_index()

v1=v1.rename(columns={col:'count','index':col})

v1['percent']=v1['count'].apply(lambda x : 100*x/sum(v1['count']))

v1=v1.sort_values(col)

v2=d2[col].value_counts().reset_index()

v2=v2.rename(columns={col:'count','index':col})

v2['percent']=v2['count'].apply(lambda x : 100*x/sum(v2['count']))

v2=v2.sort_values(col)

trace1 = go.Scatter(x=v1[col], y=v1["count"], name=0, marker=dict(color="#a678de"))

trace2 = go.Scatter(x=v2[col], y=v2["count"], name=1, marker=dict(color="#6ad49b"))

data = [trace1, trace2]

layout={'title':"target over the age"}

fig = go.Figure(data, layout=layout)

iplot(fig)
sns.lmplot(x="trestbps", y="chol",data=df,hue="cp")

plt.show()
chest_pain=pd.get_dummies(df['cp'],prefix='cp',drop_first=True)

df=pd.concat([df,chest_pain],axis=1)

df.drop(['cp'],axis=1,inplace=True)

sp=pd.get_dummies(df['slope'],prefix='slope')

th=pd.get_dummies(df['thal'],prefix='thal')

frames=[df,sp,th]

df=pd.concat(frames,axis=1)

df.drop(['slope','thal'],axis=1,inplace=True)
df.head(5)
X = df.drop(['target'], axis = 1)

y = df.target.values
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



#SVM classifier

svc_c=SVC(kernel='linear',random_state=0)

svc_c.fit(X_train,y_train)

svc_pred=svc_c.predict(X_test)

sv_cm=confusion_matrix(y_test,svc_pred)

sv_ac=accuracy_score(y_test, svc_pred)



#Bayes

gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

bayes_pred=gaussian.predict(X_test)

bayes_cm=confusion_matrix(y_test,bayes_pred)

bayes_ac=accuracy_score(bayes_pred,y_test)



#SVM regressor

svc_r=SVC(kernel='rbf')

svc_r.fit(X_train,y_train)

svr_pred=svc_r.predict(X_test)

svr_cm=confusion_matrix(y_test,svr_pred)

svr_ac=accuracy_score(y_test, svr_pred)



#RandomForest

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

rdf_ac=accuracy_score(rdf_pred,y_test)



# DecisionTree Classifier

dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(X_train,y_train)

dtree_pred=dtree_c.predict(X_test)

dtree_cm=confusion_matrix(y_test,dtree_pred)

dtree_ac=accuracy_score(dtree_pred,y_test)



#KNN

knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)

knn_cm=confusion_matrix(y_test,knn_pred)

knn_ac=accuracy_score(knn_pred,y_test)
plt.figure(figsize=(20,10))

plt.subplot(2,4,1)

plt.title("LogisticRegression_cm")

sns.heatmap(lr_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,2)

plt.title("SVM_regressor_cm")

sns.heatmap(sv_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,3)

plt.title("bayes_cm")

sns.heatmap(bayes_cm,annot=True,cmap="Oranges",fmt="d",cbar=False)

plt.subplot(2,4,4)

plt.title("RandomForest")

sns.heatmap(rdf_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,5)

plt.title("SVM_classifier_cm")

sns.heatmap(svr_cm,annot=True,cmap="Reds",fmt="d",cbar=False)

plt.subplot(2,4,6)

plt.title("DecisionTree_cm")

sns.heatmap(dtree_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.subplot(2,4,7)

plt.title("kNN_cm")

sns.heatmap(knn_cm,annot=True,cmap="Blues",fmt="d",cbar=False)

plt.show()
print('LogisticRegression_accuracy:\t',lr_ac)

print('SVM_regressor_accuracy:\t\t',svr_ac)

print('RandomForest_accuracy:\t\t',rdf_ac)

print('DecisionTree_accuracy:\t\t',dtree_ac)

print('KNN_accuracy:\t\t\t',knn_ac)

print('SVM_classifier_accuracy:\t',sv_ac)

print('Bayes_accuracy:\t\t\t',bayes_ac)
def plotting(true,pred):

    fig,ax=plt.subplots(1,2,figsize=(10,5))

    precision,recall,threshold = precision_recall_curve(true,pred[:,1])

    ax[0].plot(recall,precision,'g--')

    ax[0].set_xlabel('Recall')

    ax[0].set_ylabel('Precision')

    ax[0].set_title("Average Precision Score : {}".format(average_precision_score(true,pred[:,1])))

    fpr,tpr,threshold = roc_curve(true,pred[:,1])

    ax[1].plot(fpr,tpr)

    ax[1].set_title("AUC Score is: {}".format(auc(fpr,tpr)))

    ax[1].plot([0,1],[0,1],'k--')

    ax[1].set_xlabel('False Positive Rate')

    ax[1].set_ylabel('True Positive Rate')
plt.figure()

plotting(y_test,gaussian.predict_proba(X_test))
model_accuracy = pd.Series(data=[lr_ac,sv_ac,bayes_ac,svr_ac,rdf_ac,dtree_ac,knn_ac], 

                index=['LogisticRegression','SVM_classifier','Bayes','SVM_regressor',

                                      'RandomForest','DecisionTree_Classifier','KNN'])

fig= plt.figure(figsize=(10,7))

model_accuracy.sort_values().plot.barh()

plt.title('Model Accracy')