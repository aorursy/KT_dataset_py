import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import cufflinks as cf
import plotly.plotly as py
import plotly.graph_objs as go
init_notebook_mode(connected=True)
cf.go_offline()
%matplotlib inline
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
data=pd.read_csv('../input/ks-projects-201801.csv')
data.shape
data.head()
data.columns.values
def cat_feat(col):
    x=pd.DataFrame(data={'count':data[col].value_counts(),
                         'occurence rate':data[col].value_counts()*100/data.shape[0]},
                  index=data[col].unique())
    y=data[[col,'state']].groupby(col).mean()
    y['state']=y['state']*100
    y.rename(columns={'state':'success rate'},inplace=True)
    return pd.concat([x,y],axis=1)
col_details=pd.DataFrame(columns=['Null','Unique','Action'],index=data.columns.values)
for col in data:
    col_details.loc[col]['Null']=data[col].isnull().sum()
    col_details.loc[col]['Unique']=data[col].nunique()
col_details
data['state'].value_counts()
data[(data['state']=='successful')&(data['usd_pledged_real']<data['usd_goal_real'])]
data[(data['state']=='failed')&(data['usd_pledged_real']>=data['usd_goal_real'])]
data.at[data[(data['state']=='successful')|(data['usd_pledged_real']>=data['usd_goal_real'])].index,'state']=1
data.at[data[(data['state']!='live')&(data['usd_pledged_real']<data['usd_goal_real'])].index,'state']=0
test=data[data['state']=='live'].copy()
test.drop('state',axis=1,inplace=True)
data.drop(data[data['state']=='live'].index,inplace=True,axis=0)
data['state']=pd.to_numeric(data['state'])
data['len']=data['name'].str.len()
data[['len','state']].groupby('state').mean()
data['len'].iplot(kind='histogram',theme='polar',title='Distribution of length of the project name')
data.drop('len',axis=1,inplace=True)
col_details.loc['ID']['Action']='delete'
col_details.loc['name']['Action']='delete'
col_details
data['category'].nunique(),data['main_category'].nunique()
y=(data[['main_category','state']].groupby('main_category').mean().sort_values(by='state',ascending=False))*100
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='state',theme='polar',hline=y['state'].mean(),title='Success rate of various main categories')
y=(data[['category','state']].groupby('category').mean().sort_values(by='state',ascending=False))*100
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='state',theme='polar',hline=y['state'].mean(),title='Success rate of various categories')
y=cat_feat('main_category')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Main Category')
col_details.loc['category']['Action']='delete'
col_details.loc['main_category']['Action']='Done'
col_details
y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='occurence rate',theme='polar',title='Country Distribution')
y=cat_feat('country')
rm=y[y['occurence rate']<1].index
data['country']=data['country'].apply(lambda x:'Others' if x in rm else x)
test['country']=test['country'].apply(lambda x:'Others' if x in rm else x)
y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='occurence rate',theme='polar',title='Country Distribution')
y=cat_feat('country')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Country')
col_details.loc['country']['Action']='done'
col_details
y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='count',title='Currency Distribution')
y=cat_feat('currency')
rm=y[y['occurence rate']<1].index
data['currency']=data['currency'].apply(lambda x:'Others' if x in rm else x)
test['currency']=test['currency'].apply(lambda x:'Others' if x in rm else x)
y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='pie',labels='index',values='count',title='Currency Distribution')
data[(data['country']!='US')&(data['currency']=='USD')]['country'].value_counts()
data[(data['country']=='N,0"')]['currency'].value_counts()
(data[(data['country']=='N,0"')][['currency','state']].groupby('currency').mean().sort_values(by='state',ascending=False))*100
y=cat_feat('currency')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Currency')
col_details.loc['currency']['Action']='done'
col_details
type(data['deadline'][0]),type(data['launched'][0])
data['launched']=pd.to_datetime(data['launched'])
data['deadline']=pd.to_datetime(data['deadline'])
data['duration']=data[['launched','deadline']].apply(lambda x:(x[1]-x[0]).days,axis=1)
test['launched']=pd.to_datetime(test['launched'])
test['deadline']=pd.to_datetime(test['deadline'])
test['duration']=test[['launched','deadline']].apply(lambda x:(x[1]-x[0]).days,axis=1)
y=data[['main_category','duration']].groupby('main_category').mean().sort_values(by='duration',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='duration',theme='polar',hline=y['duration'].mean(),title='Average Duration of main categories')
y=data[['category','duration']].groupby('category').mean().sort_values(by='duration',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='duration',theme='polar',hline=y['duration'].mean(),title='Average duration of categories')
data['duration'].max(),data['duration'].min()
data.drop(data[(data['duration']>365)].index,axis=0,inplace=True)
test.drop(test[(test['duration']>365)].index,axis=0,inplace=True)
data['duration'].iplot(kind='histogram',theme='polar',title='Duration Distribution')
data['duration']=data['duration'].apply(lambda x:(int(x/10)+1)*10)
test['duration']=test['duration'].apply(lambda x:(int(x/10)+1)*10)
data['duration'].nunique()
y=cat_feat('duration')
y.reset_index(inplace=True)
y.iplot(kind='bar',x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Duration')
data['year']=data['launched'].apply(lambda x:x.year)
data['month']=data['launched'].apply(lambda x:x.month)
data['date']=data['launched'].apply(lambda x:x.day)
data['weekday']=data['launched'].apply(lambda x:x.weekday())
y=cat_feat('year')
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate & Success Rate year wise')
y=cat_feat('month')
y.rename(index={1:'January',2:'February',3:'March',4:'April',5:'May',6:'June',
               7:'July',8:'August',9:'September',10:'October',11:'November',12:'December'},
         inplace=True)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate & Success Rate month wise')
y=cat_feat('date')
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate vs. Success Rate date wise')
y=cat_feat('weekday')
y.rename(index={0:'Sunday',1:'Monday',2:'Tuesday',3:'Wednesday',4:'Thursday',5:'Friday',6:'Saturday'},inplace=True)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='index',y=['occurence rate','success rate'],color=['blue','green'],title='Occurence rate vs. Success Rate day wise')
y=data[['year','usd_goal_real']].groupby('year').mean()
y.reset_index(inplace=True)
y.iplot(kind='bar',x='year',y='usd_goal_real',theme='polar',title='Avg. goal Year wise')
y=data[data['year']==2017][['main_category','state']].groupby('main_category').mean()
y.rename(columns={'state':'2017'},inplace=True)
yy=data[data['year']<2017][['main_category','state']].groupby('main_category').mean()
yy.rename(columns={'state':'Prev'},inplace=True)
yyy=pd.concat([yy,y],axis=1)
yyy.reset_index(inplace=True)
yyy.iplot(kind='bar',x='main_category',theme='polar',title='Last year performance of various main categories')
y=data[data['state']==1][['year','usd_goal_real']].groupby('year').sum()
y.reset_index(inplace=True)
y.iplot(kind='bar',x='year',y='usd_goal_real',theme='polar',title='Money raised year wise')
data.drop(['year','month','date','weekday'],axis=1,inplace=True)
col_details.loc['launched']['Action']='delete'
col_details.loc['deadline']['Action']='delete'
col_details
data['usd_goal_real'].max(),data['usd_goal_real'].min()
data['usd_goal_real'].nunique()
data[data['state']==1].sort_values(by='usd_goal_real',ascending=False).head(5)
plt.figure(figsize=(20,5))
plt.scatter(data.index,data['usd_goal_real'],marker='.',s=10)
plt.title('Goal Distribution')
plt.show()
y=data[['main_category','usd_goal_real']].groupby('main_category').mean().sort_values(by='usd_goal_real',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='bar',x='main_category',y='usd_goal_real',hline=[data['usd_goal_real'].mean()],theme='polar',title='Average goal main category wise')
y=data[['category','usd_goal_real']].groupby('category').mean().sort_values(by='usd_goal_real',ascending=False)
y.reset_index(inplace=True)
y.iplot(kind='line',fill=True,x='category',y='usd_goal_real',hline=[data['usd_goal_real'].mean()],theme='polar',title='Average goal category wise')
y=data[data['state']==1][['main_category','usd_goal_real']].groupby('main_category').mean()
y.rename(columns={'usd_goal_real':'Successful'},inplace=True)
yy=data[data['state']==0][['main_category','usd_goal_real']].groupby('main_category').mean()
yy.rename(columns={'usd_goal_real':'Unsuccessful'},inplace=True)
y=pd.concat([y,yy],axis=1)
y.reset_index(inplace=True)
y.iplot(kind='bar',fill=True,x='main_category',barmode='overlay',y=['Unsuccessful','Successful'],color=['red','green'],title='Average goal of successful vs unsuccessful fundraiser')
x=data['usd_goal_real'].apply(lambda x:(int(x/10000))*10000)
x=pd.DataFrame(x.value_counts(),index=x.value_counts().index).sort_index()
x['usd_goal_real'][0:25].iplot(kind='bar',title='Goal Distribution',theme='polar')
data['range']=data['usd_goal_real'].apply(lambda x:(int(x/1000))*1000 if x/1000<=50 else 51000)
test['range']=test['usd_goal_real'].apply(lambda x:(int(x/1000))*1000 if x/1000<=50 else 51000)
y=cat_feat('range')
y.reset_index(inplace=True)
y.iplot(kind='line',x='index',y=['occurence rate','success rate'],title='Goal')
col_details.loc['goal']['Action']='delete'
col_details.loc['pledged']['Action']='delete'
col_details.loc['backers']['Action']='delete'
col_details.loc['usd pledged']['Action']='delete'
col_details.loc['usd_pledged_real']['Action']='delete'
col_details.loc['usd_goal_real']['Action']='delete'
col_details.loc['state']['Action']='done'
col_details
for col in col_details.index:
    if col_details.loc[col]['Action']=='delete':
        data.drop(col,axis=1,inplace=True)
        test.drop(col,axis=1,inplace=True)
data.head()
test.head()
for col in data:
    if (col!='state'):
        data[col]=data[col].apply(lambda x:col+'_'+str(x))
        test[col]=test[col].apply(lambda x:col+'_'+str(x))
        x=pd.get_dummies(data[col],drop_first=True)
        y=pd.get_dummies(test[col],drop_first=True)
        data=pd.concat([data,x],axis=1).drop(col,axis=1)
        test=pd.concat([test,y],axis=1).drop(col,axis=1)
data.head()
test.head()
X=data.drop('state',axis=1)
y=data['state']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
predictions=dtc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
rfc=RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
predictions=rfc.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
lr=LogisticRegression()
lr.fit(X_train,y_train)
predictions=lr.predict(X_test)
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))