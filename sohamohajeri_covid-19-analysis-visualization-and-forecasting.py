import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot

init_notebook_mode (connected=True)

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import plotly.io as pio

pio.renderers.default='notebook'
df=pd.read_csv('../input/covid19-dataset/Covid-19_Dataset.csv')
df.head(2)
df.shape
df.dtypes
df.info()
df.drop(['id','case_in_country','summary','symptom_onset', 'If_onset_approximated', 'hosp_visit_date', 'exposure_start',

'exposure_end', 'symptom', 'source', 'link'],axis=1,inplace=True)
100*df.isnull().sum()/df.shape[0]
df['age']= df['age']. fillna(df['age'].mean())
df_dum=pd.get_dummies(df['gender'].dropna(), drop_first=True)
df_dum['male'].median()
df['gender']= df['gender']. fillna('male')
df.dropna(inplace=True)
df.isnull().sum()
df.columns=df.columns.str.lower().str.replace(' ','_')
df['reporting_date']=pd.to_datetime(df['reporting_date'])
df['year']=df['reporting_date'].apply(lambda x:x.year)

df['month']=df['reporting_date'].apply(lambda x:x.month)
df['month'].unique()
df.drop(['reporting_date', 'year'], axis=1, inplace=True)
df.head(2)
plt.figure(figsize=(8,6))

df[df['death']==1]['age'].plot(kind='hist',bins=70,colormap='Accent')

plt.title('Number of Patients Died Based On Their Age',fontsize=15)

plt.xlabel('Age',fontsize=12)

plt.ylabel('Frequency',fontsize=12)

plt.show()
plt.figure(figsize=(8,6))

df[df['recovered']==1]['age'].plot(kind='hist',bins=70,colormap='rainbow')

plt.title('Number of Patients Recovered Based On Their Age',fontsize=15)

plt.xlabel('Age',fontsize=12)

plt.ylabel('Frequency',fontsize=12)

plt.show()
print('Current count of patients:',df['death'].count())

print('Number of Dead Patients:', df[df['death']==1]['death'].count())

print('Number of Recovered Patients:',df[df['recovered']==1]['death'].count())

print('Number of Patients Receiving Treatment:',df[(df['death']==0)&(df['recovered']==0)]['death'].count())
plt.figure(figsize=(8,6))

plt.bar(x=['Recovered','Dead'],height=[159,63], color='pink')

plt.title('Patients Status',fontsize=15)

plt.xlabel('Status', fontsize=12)

plt.ylabel('Number',fontsize=12)

plt.show()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.countplot(x='gender', data=df[df['death']==1], palette='viridis')

plt.xlabel('Gender', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Dead Patients',fontsize=15)

plt.subplot(1,2,2)

sns.countplot(x='gender', data=df[df['recovered']==1], palette='spring')

plt.xlabel('Gender', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Recoverred Patients',fontsize=15)

plt.show()
plt.figure(figsize=(15,6))

plt.subplot(1,2,1)

sns.countplot(x='from_wuhan', data=df[df['death']==1], palette='BuPu')

plt.xticks([0,1], ['Not from Wuhan','from Wuhan'])

plt.xlabel('Origin', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Dead Patients',fontsize=15)

plt.subplot(1,2,2)

sns.countplot(x='from_wuhan', data=df[df['recovered']==1], palette='hot')

plt.xticks([0,1], ['Not from Wuhan','from Wuhan'])

plt.xlabel('Origin', fontsize=12)

plt.ylabel('Count', fontsize=12)

plt.title('Recoverred Patients',fontsize=15)

plt.show()
country_order=list(df.groupby('country').count()['location'].sort_values(ascending=False).index)
plt.figure(figsize=(12,6))

sns.countplot(x='country',data=df,color='blue',order=country_order)

plt.xticks(rotation=90)

plt.ylabel('Number of Patients')

plt.xlabel('Country')

plt.title('Number of Covid Patients in Different Countries',fontsize=15)

plt.show()
groupby_df=df.groupby('country').sum()
data=dict (type='choropleth', locations=list(groupby_df.index), locationmode='country names', colorscale='viridis', reversescale=True, text= list(groupby_df.index),z=groupby_df['death'], colorbar={'title':'Number of Death Patients'})

layout=dict(title='Number of Death Patients in Each Country', geo=dict(showframe=False, projection={'type':'mercator'}))

choromap=go.Figure(data=[data], layout=layout)

iplot(choromap)
data=dict (type='choropleth', locations=list(groupby_df.index), locationmode='country names', colorscale='viridis', reversescale=True, text= list(groupby_df.index),z=groupby_df['recovered'], colorbar={'title':'Number of Recovered Patients'})

layout=dict(title='Number of Recovered Patients in Each Country', geo=dict(showframe=False, projection={'type':'mercator'}))

choromap=go.Figure(data=[data], layout=layout)

iplot(choromap)
le1=LabelEncoder()

le1.fit(df['location'])

df['location']=le1.transform(df['location'])
le2=LabelEncoder()

le2.fit(df['country'])

df['country']=le2.transform(df['country'])
le3=LabelEncoder()

le3.fit(df['gender'])

df['gender']=le3.transform(df['gender'])
df.head()
y=df['recovered']

X=df[['location','country','gender','age','visiting_wuhan','from_wuhan','month']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr1=LogisticRegression()

lr1.fit(X,y)

predictions_lr1=lr1.predict(X_test)
print(confusion_matrix(y_test,predictions_lr1))

print('\n')

print(classification_report(y_test,predictions_lr1))
dtc1=DecisionTreeClassifier()

dtc1.fit(X_train,y_train)

predictions_dtc1=dtc1.predict(X_test)
print(confusion_matrix(y_test,predictions_dtc1))

print("\n")

print(classification_report(y_test,predictions_dtc1))
rfc1=RandomForestClassifier(n_estimators=200)

rfc1.fit(X_train,y_train)

predictions_rfc1=rfc1.predict(X_test)
print(confusion_matrix(y_test,predictions_rfc1))

print('\n')

print(classification_report(y_test,predictions_rfc1))
svc1=SVC()

svc1.fit(X_train,y_train)

predictions_svc1=svc1.predict(X_test)
print(confusion_matrix(y_test,predictions_svc1))

print('\n')

print(classification_report(y_test,predictions_svc1))
param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}
gs1=GridSearchCV(SVC(),param_grid, verbose=3)

gs1.fit(X_train,y_train)

predictions_gs1=gs1.predict(X_test)
print(confusion_matrix(y_test,predictions_gs1))

print('\n')

print(classification_report(y_test,predictions_gs1))
xgbc1=xgb.XGBClassifier(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)

xgbc1.fit(X_train,y_train)
predictions_xgbc1=xgbc1.predict(X_test)
print(confusion_matrix(y_test,predictions_xgbc1))

print('\n')

print(classification_report(y_test,predictions_xgbc1))
print('Accuracy Score, Logistic Regression: ', round(accuracy_score(y_test,predictions_lr1),ndigits=3))

print('Accuracy Score, Decision Tree Classifier: ', round(accuracy_score(y_test,predictions_dtc1),ndigits=3))

print('Accuracy Score, Random Forest Classifier: ', round(accuracy_score(y_test,predictions_rfc1),ndigits=3))

print('Accuracy Score, Support Vector Classifier: ', round(accuracy_score(y_test,predictions_gs1),ndigits=3))

print('Accuracy Score, XGBoost Classifier: ', round(accuracy_score(y_test,predictions_xgbc1), ndigits=2))
y=df['death']

X=df[['location','country','gender','age','visiting_wuhan','from_wuhan','month']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr2=LogisticRegression()

lr2.fit(X,y)

predictions_lr2=lr2.predict(X_test)
print(confusion_matrix(y_test,predictions_lr2))

print('\n')

print(classification_report(y_test,predictions_lr2))
dtc2=DecisionTreeClassifier()

dtc2.fit(X_train,y_train)

predictions_dtc2=dtc2.predict(X_test)
print(confusion_matrix(y_test,predictions_dtc2))

print("\n")

print(classification_report(y_test,predictions_dtc2))
rfc2=RandomForestClassifier(n_estimators=200)

rfc2.fit(X_train,y_train)

predictions_rfc2=rfc2.predict(X_test)
print(confusion_matrix(y_test,predictions_rfc2))

print('\n')

print(classification_report(y_test,predictions_rfc2))
svc2=SVC()

svc2.fit(X_train,y_train)

predictions_svc2=svc2.predict(X_test)
print(confusion_matrix(y_test,predictions_svc2))

print('\n')

print(classification_report(y_test,predictions_svc2))
param_grid={'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf']}
gs2=GridSearchCV(SVC(),param_grid, verbose=3)

gs2.fit(X_train,y_train)

predictions_gs2=gs2.predict(X_test)
print(confusion_matrix(y_test,predictions_gs2))

print('\n')

print(classification_report(y_test,predictions_gs2))
xgbc2=xgb.XGBClassifier(n_estimators=200, learning_rate=0.08, gamma=0, subsample=0.5,colsample_bytree=1, max_depth=8)

xgbc2.fit(X_train,y_train)
predictions_xgbc2=xgbc2.predict(X_test)
print(confusion_matrix(y_test,predictions_xgbc2))

print('\n')

print(classification_report(y_test,predictions_xgbc2))
print('Accuracy Score, Logistic Regression: ', round(accuracy_score(y_test,predictions_lr2),ndigits=3))

print('Accuracy Score, Decision Tree Classifier: ', round(accuracy_score(y_test,predictions_dtc2),ndigits=3))

print('Accuracy Score, Random Forest Classifier: ', round(accuracy_score(y_test,predictions_rfc2),ndigits=3))

print('Accuracy Score, Support Vector Classifier: ', round(accuracy_score(y_test,predictions_gs2),ndigits=3))

print('Accuracy Score, XGBoost Classifier: ', round(accuracy_score(y_test,predictions_xgbc2), ndigits=3))