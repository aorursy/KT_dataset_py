import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.graph_objects as go

import plotly.express as px

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

plotly.offline.init_notebook_mode (connected = True)
pd. set_option('display.max_columns', None)

df=pd.read_csv('../input/breast-cancer-wisconsin-data/data.csv')

df.head()
df.isna().any()
df=df.iloc[:,1:-1]
df['Count']=1

df_diag=df.groupby('diagnosis')['Count'].sum().reset_index()
fig1=px.pie(df_diag,values='Count',names='diagnosis',hole=0.4)

fig1.update_layout(title='Diagnosis distribution',title_x=.5,

                   annotations=[dict(text='Diagnosis',font_size=20, 

                   showarrow=False,height=800,width=700)])







fig1.show()
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

df_scaled=pd.DataFrame(scaler.fit_transform(df.iloc[:,1:-1]),columns=df.iloc[:,1:-1].columns)
df_scaled=pd.merge(df['diagnosis'],df_scaled,on=df_scaled.index)
df_scaled.drop('key_0',axis=1,inplace=True)
data = pd.melt(df_scaled.iloc[:,:-1],id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
fig2=px.violin(data,x='features',y='value',box=True,color='diagnosis',violinmode='overlay')

fig2.update_layout(violingap=0)

fig2.show()
df_mean=df_scaled.iloc[:,:11]

data = pd.melt(df_mean,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
fig3=px.violin(data,x='features',y='value',box=True,color='diagnosis',violinmode='overlay',labels={'value':'Scaled values'},color_discrete_sequence =['red','blue'])

fig3.update_layout(violingap=0,template='plotly_dark',title='Mean parameter distribution',title_x=0.5)

fig3.show()
df_se=df_scaled.iloc[:,11:21]



df_se=pd.merge(df['diagnosis'],df_se,on=df_se.index)

df_se.drop('key_0',axis=1,inplace=True)

data = pd.melt(df_se,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
fig4=px.violin(data,x='features',y='value',box=True,color='diagnosis',violinmode='overlay',color_discrete_sequence =['red','blue'],labels={'value':'Scaled values'})

fig4.update_layout(violingap=0,template='plotly_dark',title='SE parameter distribution',title_x=0.5)

fig4.show()
df_worst=df_scaled.iloc[:,21:]

df_worst=pd.merge(df['diagnosis'],df_worst,on=df_worst.index)

df_worst.drop('key_0',axis=1,inplace=True)
data = pd.melt(df_worst,id_vars="diagnosis",

                    var_name="features",

                    value_name='value')
fig5=px.violin(data,x='features',y='value',box=True,color='diagnosis',violinmode='overlay',color_discrete_sequence =['red','blue'],labels={'value':'Scaled values'})

fig5.update_layout(violingap=0,template='plotly_dark',title='Worst parameter distribution',title_x=0.5)

fig5.show()
corrs=df_scaled.corr()

plt.figure(figsize=(20,20))

sns.heatmap(corrs,annot=True)
df_sim=df_scaled.loc[:,['radius_worst','perimeter_worst','area_worst']]

g = sns.PairGrid(df_sim, diag_sharey=False)

g.map_lower(sns.kdeplot, cmap="viridis")

g.map_upper(plt.scatter)

g.map_diag(sns.kdeplot, lw=3)
imp_feat=['diagnosis','radius_mean','compactness_mean','concavity_mean',

          'concave points_mean','radius_worst','texture_worst','concave points_worst']



df_scaled[imp_feat].head()
df_scaled['diagnosis']=df_scaled['diagnosis'].replace({'M':1,'B':0})
target=df_scaled['diagnosis']

df_scaled.drop('diagnosis',axis=1,inplace=True)
X=df_scaled.values

Y=target
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,shuffle=True,random_state=0,test_size=0.25)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
reg_log=LogisticRegression()

reg_log.fit(X_train,y_train)
reg_log.score(X_train,y_train)
y_preds_log=reg_log.predict(X_test)

reg_log.score(X_test,y_test)
conf_mat_log=confusion_matrix(y_preds_log,y_test)



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_log,ax=ax,annot=True,cmap='summer')

ax.xaxis.set_ticklabels(['Benign', 'Malignant'])

ax.yaxis.set_ticklabels(['Benign','Malignant'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix Logsitic Regression',size=15)

plt.figure(figsize=(10,8))
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(max_depth=10,random_state=5)
rfc.fit(X_train,y_train)
rfc.score(X_train,y_train)
y_preds_rfc=rfc.predict(X_test)
conf_mat_rfc=confusion_matrix(y_preds_rfc,y_test)



fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111)

cax = sns.heatmap(conf_mat_rfc,ax=ax,annot=True,cmap='gnuplot')

ax.xaxis.set_ticklabels(['Benign', 'Malignant'])

ax.yaxis.set_ticklabels(['Benign','Malignant'],rotation=0)

ax.set_xlabel('Predicted',size=15)

ax.set_ylabel('Actual',size=15)

plt.title('Confusion matrix RFC',size=15)

plt.figure(figsize=(10,8))