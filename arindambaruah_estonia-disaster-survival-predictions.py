import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

df=pd.read_csv('../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv')

df.head()
df.isna().any()
unn_cols=['PassengerId','Firstname','Lastname']



for cols in unn_cols:

    df.drop(cols,axis=1,inplace=True)
df.head()
df_temp=df.copy()

df_temp['Count']=1

df_country=df_temp.groupby('Country')['Count'].sum().reset_index().sort_values(by='Count',ascending=False)



fig1=go.Figure([go.Pie(labels=df_temp['Country'],values=df_temp['Count'])])



fig1.update_traces(textfont_size=15,textinfo='value+percent')

fig1.update_layout(title='Passenger nationalities',title_x=0.5,height=700,width=700)

fig1.show()
df_temp['Survived']=df_temp['Survived'].replace(0,'Not survived')

df_temp['Survived']=df_temp['Survived'].replace(1,'Survived')



sns.catplot('Sex',kind='count',hue='Survived',data=df_temp,height=8,aspect=2,palette='winter')

plt.xticks(size=15)

plt.xlabel('Sex',size=15)

plt.ylabel('Number of passengers',size=15)

plt.title('Survival of passengers based on sex',size=25)
df_male=df_temp[df_temp['Sex']=='M']

df_female=df_temp[df_temp['Sex']=='F']



colors=['green','orange']

df_survival=df_temp.groupby('Survived')['Count'].sum().reset_index().sort_values(by='Count')

fig2=go.Figure([go.Pie(labels=df_survival['Survived'],values=df_survival['Count'])])



fig2.update_traces(textfont_size=15,textinfo='value+percent+label',marker=dict(colors=colors))

fig2.update_layout(title='\n  \n Male fatality rate: {0:.2f} % \n Female fatality rate: {1:.2f} %'.format(100*df_male['Survived'].value_counts()[0]/ df_male.shape[0],100*df_female['Survived'].value_counts()[0]/ df_female.shape[0]),title_x=0.5,height=700,width=700)

fig2.show()
plt.figure(figsize=(10,8))

sns.distplot(df_temp['Age'])

plt.title('Passenger age distribution',size=20)

plt.axvline(df_temp['Age'].median(),color='red',label='Median age')

plt.legend()
fig3=plt.figure(figsize=(10,8))

ax1=fig3.add_subplot(111)

plt.title('Surival with respect to age',size=20)



sns.regplot(df['Age'],df['Survived'],ax=ax1)

ax1.set_xlabel('Age',size=15)

ax1.set_ylabel('Survived',size=15)
df_temp['Category']=df_temp['Category'].replace('C','Crew member')

df_temp['Category']=df_temp['Category'].replace('P','Passenger')
df_cats=df_temp.groupby('Category')['Count'].sum().reset_index()



fig3=go.Figure([go.Pie(labels=df_cats['Category'],values=df_cats['Count'])])



fig3.update_traces(textfont_size=15,textinfo='value+percent')

fig3.update_layout(title='Passenger categories',title_x=0.5,height=700,width=700)

fig3.show()
df_crew=df_temp[df_temp['Category']=='Crew member']

df_pass=df_temp[df_temp['Category']=='Passenger']

sns.catplot('Category',kind='count',data=df_temp,hue='Survived',palette='viridis',aspect=2,height=8)

plt.xticks(size=15)

plt.xlabel('Category',size=15)

plt.ylabel('Number of passengers',size=15)

plt.title('Category wise fatalities \n \n Crew fatality rate:{0:.2f}% \n \n Passenger fatality rate:{1:.2f}%'.format(100*df_crew['Survived'].value_counts()[0]/df_crew.shape[0],

                                                                                                                     100*df_pass['Survived'].value_counts()[0]/df_pass['Survived'].shape[0]),size=20)
df.loc[df['Age']<=10,'Age band']=1

df.loc[(df['Age']>10) & (df['Age']<21),'Age band']=2

df.loc[(df['Age']>20) & (df['Age']<41),'Age band']=3

df.loc[(df['Age']>40) & (df['Age']<61),'Age band']=4

df.loc[(df['Age']>60),'Age band']=5
df.drop('Age',axis=1,inplace=True)
temp=pd.get_dummies(df['Category'])

df=df.merge(temp,on=df.index)
temp_sex=pd.get_dummies(df['Sex'])
df.drop('key_0',axis=1,inplace=True)

df=df.merge(temp_sex,on=df.index)
df.drop(['key_0','Country','Sex','Category'],axis=1,inplace=True)

df.head()
target=df['Survived']

df.drop('Survived',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X=df

y=target

X_train,X_test,y_train,y_test=train_test_split(X,y,shuffle=True,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
reg_log=LogisticRegression()

reg_log.fit(X_train,y_train)
y_pred=reg_log.predict(X_test)
reg_log.score(X_train,y_train)
from sklearn.metrics import confusion_matrix

fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(111)

conf_mat_log=confusion_matrix(y_pred,y_test)

sns.heatmap(conf_mat_log,annot=True,fmt='g',ax=ax)

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

ax.xaxis.set_ticklabels(['Not survived', 'Survived'])

ax.yaxis.set_ticklabels(['Not survived', 'Survived'],rotation=0)
from sklearn.tree import DecisionTreeClassifier



dtc=DecisionTreeClassifier(max_depth=4)

dtc.fit(X_train,y_train)

y_pred_dtc=dtc.predict(X_test)
fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(111)

conf_mat_dtc=confusion_matrix(y_pred_dtc,y_test)

sns.heatmap(conf_mat_dtc,annot=True,fmt='g',ax=ax,cmap='gnuplot')

ax.set_xlabel('Predicted')

ax.set_ylabel('Actual')

ax.xaxis.set_ticklabels(['Not survived', 'Survived'])

ax.yaxis.set_ticklabels(['Not survived', 'Survived'],rotation=0)
from xgboost import XGBClassifier
xgb=XGBClassifier()

xgb.fit(X_train,y_train)

xgb.score(X_train,y_train)
y_pred_xgb=xgb.predict(X_test)

fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(111)

conf_mat_xgb=confusion_matrix(y_pred_xgb,y_test)

sns.heatmap(conf_mat_xgb,annot=True,fmt='g',ax=ax,cmap='summer')

ax.set_ylabel('Predicted')

ax.set_xlabel('Actual')

ax.xaxis.set_ticklabels(['Not survived', 'Survived'])

ax.yaxis.set_ticklabels(['Not survived', 'Survived'],rotation=0)
from lightgbm import LGBMClassifier
lgb=LGBMClassifier()

lgb.fit(X_train,y_train)


fig=plt.figure(figsize=(10,8))

ax=fig.add_subplot(111)

y_pred_lgb=lgb.predict(X_test)

conf_mat_lgb=confusion_matrix(y_pred_lgb,y_test)

sns.heatmap(conf_mat_lgb,annot=True,fmt='g',ax=ax,cmap='coolwarm')

ax.set_xlabel('Actual')

ax.set_ylabel('Predicted')

ax.xaxis.set_ticklabels(['Not survived', 'Survived'])

ax.yaxis.set_ticklabels(['Not survived', 'Survived'],rotation=0)