# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import  matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold

from sklearn.preprocessing import MinMaxScaler,RobustScaler

df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
df.head()
df=df.rename(columns={'ssc_p':'sslc_p','ssc_b':'sslc_b','workex':'work_exp'})
df.drop('sl_no',inplace=True,axis=1)
df.info()
for col in df.select_dtypes(object).columns:
    print(f'{col} has {df[col].nunique()} values')
    print(f'{col} -----  {df[col].unique()}')
    print()
plt.hist(df['salary'])
#based on gender and status

gend_stat=df.groupby(['gender','status']).count()['sslc_p'].unstack(level=1)
gend_stat.plot(kind='bar')



gend_stat['Fem_placed']=(gend_stat['Placed'][0]/df['gender'].value_counts()[1])*100
gend_stat['Male_placed']=(gend_stat['Placed'][1]/df['gender'].value_counts()[0])*100
gend_stat
from plotly.subplots import make_subplots
val_f=[gend_stat.iloc[0][0],gend_stat.iloc[0][1]]
val_m=[gend_stat.iloc[1][0],gend_stat.iloc[1][1]]
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]],subplot_titles=['Female status %','Male Status %'])
fig_f=px.pie(values=val_f,names=['notplaced','placed'])
fig_m=px.pie(values=val_m,names=['notplaced','placed'])
fig.add_trace(fig_f['data'][0],row=1,col=1)
fig.add_trace(fig_m['data'][0],row=1,col=2)
df[df['salary'].isnull()]['status'].unique()
#Notplaced status  has nullvalues in salary.we impute it with 00000 ,but first let us see how salary varies wrt to gender, mba stream
df_placed=df[df['status']=='Placed']
df_placed['salary'].describe()
df_placed_f=df_placed[df_placed['gender']=='F']
df_placed_m=df_placed[df_placed['gender']=='M']
fig,ax=plt.subplots(1,2,figsize=(16,6))
sns.distplot(df_placed_f['salary'],ax=ax[0]).set_title('female salary dist')

sns.distplot(df_placed_m['salary'],ax=ax[1])
ax[1].set_title('male salary dist')
print(df_placed_f['salary'].describe())
print('------')
print(df_placed_m['salary'].describe())
#specilization wrt to mba streram vs salarydist

print(df_placed[df_placed['specialisation']=='Mkt&Fin']['salary'].describe())
print('__________')
print('HR VS salary')
print(df_placed[df_placed['specialisation']=='Mkt&HR']['salary'].describe())
fig,ax=plt.subplots(1,2,figsize=(16,4))
sns.distplot(df_placed[df_placed['specialisation']=='Mkt&HR']['salary'],ax=ax[0]).set_title('HR salary dist')
sns.distplot(df_placed[df_placed['specialisation']=='Mkt&Fin']['salary'],ax=ax[1]).set_title('Fin salary dist')

#Maxsalary for Hr is 450000andfor Fin is 940000
df_placed_sal=df_placed[df_placed['salary']<800000]
df_placed_sal.describe()
#so only one person had salry greater than 800000so for time we will remmove him

print(df_placed_sal[df_placed_sal['specialisation']=='Mkt&Fin']['salary'].describe())
print('__________')
print('HR VS salary')
print(df_placed_sal[df_placed_sal['specialisation']=='Mkt&HR']['salary'].describe())
val=[df_placed_sal[df_placed_sal['specialisation']=='Mkt&Fin']['salary'].mean(),df_placed_sal[df_placed_sal['specialisation']=='Mkt&HR']['salary'].mean()]
names=['Fin mean sal','HR mean sal']
sns.barplot(y=val,x=names)
#mean sal of mktfin is almost 20,000moretha mkthr
print('work_exp does not affect the chance of getting placed. More number of students  who  got placed have no work_exp ')
print('----------')
print(df_placed_sal.groupby(['specialisation','work_exp'])['work_exp'].count())

print()
print('-------------------')
print('having work_exp gets you lillte more salary')
print('-----')
print(df_placed_sal.groupby(['specialisation','work_exp'])['salary'].mean())
#having work_exp fetches more salary 

plt.figure(figsize=(13,6))
df['status_enc']=np.where(df['status']=='Placed',1,0)
df['work_exp_enc']=np.where(df['work_exp']=='No',0,1)
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu')
plt.yticks(rotation=0,fontsize=12,color='black')
plt.xticks(color='black')

#sslc_p,hsc_p,degree_p has more effect on statusthan others

#does MBA % affects placement
plt.style.use('ggplot')
plt.figure(figsize=(10,6))
sns.kdeplot(df_placed['mba_p'],label='placed',shade=True)
sns.kdeplot(df[df['status']=='Not Placed']['mba_p'],label='not_placed',shade=True)
plt.title('MBA Percentage')

#MBA percentage makes no effect on status
#sslc_p vs sta
plt.figure(figsize=(10,6))
sns.kdeplot(df_placed['sslc_p'],label='placed',shade=True,color='green')
sns.kdeplot(df[df['status']=='Not Placed']['sslc_p'],label='not_placed',shade=True,color='red')
plt.title('SSLC Percentage')
#hsc_p vs status
plt.figure(figsize=(10,6))
sns.kdeplot(df_placed['hsc_p'],label='placed',shade=True,color='green')
sns.kdeplot(df[df['status']=='Not Placed']['hsc_p'],label='not_placed',shade=True,color='red')
plt.title('12th Percentage')
#dgree vs status
plt.figure(figsize=(10,6))
sns.kdeplot(df_placed['degree_p'],label='placed',shade=True,color='green')
sns.kdeplot(df[df['status']=='Not Placed']['degree_p'],label='not_placed',shade=True,color='red')
plt.title('degree Percentage')
#doesetest have  impact on placement
#hsc_p vs status
plt.figure(figsize=(10,6))
sns.kdeplot(df_placed['etest_p'],label='placed',shade=True,color='green')
sns.kdeplot(df[df['status']=='Not Placed']['etest_p'],label='not_placed',shade=True,color='red')
plt.title('12th Percentage')

#we couldsee that etest_p has no impact
#does degree stream affect placement
fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'},{'type':'domain'}]],subplot_titles=['commerce placed %','others placed %','sci&tech placed %'])
df_degree=df.groupby(['degree_t','status'])['status'].count()
fig_c=px.pie(values=[df_degree[0],df_degree[1]],names=['notplaced','placed'])
fig_o=px.pie(values=[df_degree[2],df_degree[3]],names=['notplaced','placed'])
fig_sc=px.pie(values=[df_degree[4],df_degree[5]],names=['notplaced','placed'])
fig.add_trace(fig_c['data'][0],row=1,col=1)
fig.add_trace(fig_o['data'][0],row=1,col=2)
fig.add_trace(fig_sc['data'][0],row=1,col=3)
#now checking the salary for these streams
plt.figure(figsize=(10,4))
sns.boxplot(y=df_placed['degree_t'],x=df_placed['salary'])
df_p_c=df_placed.groupby(['degree_t'])['status'].count().reset_index()
fig=px.pie(values=df_p_c['status'],names=df_p_c['degree_t'])
fig.update_traces(textinfo='label+percent',marker=dict(line=dict(color='white', width=2)))
#again , commerce and science background students has more chance both accounting for53 and 42% respectively
#puc stream vs placement
df_puc_stream=df_placed.groupby(['hsc_s'])['status'].count().reset_index()
fig=px.pie(values=df_puc_stream['status'],names=df_puc_stream['hsc_s'])
fig.update_traces(textinfo='label+percent',marker=dict(line=dict(color='white', width=2)))
plt.figure(figsize=(25,5))
sns.countplot(df['salary'])
plt.xticks(rotation=90);
#23 students have salary 300000(majority) followed by 15 and 17 students with salary 240000 and 250000

fig,ax=plt.subplots(1,2,figsize=(20,8))
sns.boxplot(df['specialisation'],df['salary'],hue=df['gender'],ax=ax[0]).set_title('gender  wise specialisation wrt salary ')
sns.boxplot(df['gender'],df['salary'],ax=ax[1]).set_title('gender vs salary')


#as per above visuals,there is salary disparity ,unequal pay
#let us check percentage wrt to gender
fig,ax=plt.subplots(1,3,figsize=(15,8))
sns.boxplot(df['gender'],df['mba_p'],hue=df['gender'],ax=ax[0]).set_title('gender wrt mba_p  ')
sns.boxplot(df['gender'],df['hsc_p'],ax=ax[1]).set_title('gender wrt hsc_p')
sns.boxplot(df['gender'],df['degree_p'],ax=ax[2]).set_title('gender wrt degree_p')
#yup female mean score is more than male for mba,degree
df.groupby(['status','work_exp'])['work_exp'].count().unstack(level=1).plot(kind='bar')
plt.xticks(rotation=0)

#i dont think work_exp impacts placement ,bcz more number of people  are placed have no work_exp
#Imputing NUll values
#since null values in salary is bcz of students not being placed, so we will impute it 0000
#df_new['salary'].fillna(0,inplace=True)
#converting string type to int by one hot encoding.
df.head()
print(df.columns)
df['hsc_s1']=df['hsc_s'].map({'Commerce':2,'Science':1,'Arts':0})
df['degree_t1']=df['degree_t'].map({'Sci&Tech':1,'Comm&Mgmt':2,'Others':0})

df_new=df.copy()
df_new['gender']=df_new['gender'].map({'M':1,"F":0})
df_new['work_exp']=df_new['work_exp'].map({"No":0, "Yes":1})
df_new['specialisation']=df_new['specialisation'].map({"Mkt&HR":0, "Mkt&Fin":1})

df_new['status']=df_new['status'].map({'Placed':1,'Not Placed':0})    
df_new.head()
df_new=df.copy()
for col in ['gender','work_exp','specialisation']:
    dummy=pd.get_dummies(df_new[col],drop_first=True)
    df_new=df_new.join(dummy)
df_new.columns
X=df_new[['sslc_p',  'hsc_p', 'degree_p',
         'etest_p', 'mba_p', 
        'hsc_s1', 'degree_t1', 'M', 'Yes', 'Mkt&HR']]
y=df_new['status']

x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.75,random_state=42)
print(f'{x_train.shape} {x_test.shape} {y_train.shape} {y_test.shape}')
lr=LogisticRegression().fit(x_train,y_train)
pred=lr.predict(x_test)
print(confusion_matrix(pred,y_test))
print(classification_report(pred,y_test))
accuracy_score(y_test, pred)
#DecisionTree

from sklearn.model_selection import RandomizedSearchCV
dt=DecisionTreeClassifier()
param={'max_depth':[2,3,4,5,8]}
rs=RandomizedSearchCV(dt,param_distributions=param,n_iter=5,n_jobs=1,cv=3,random_state=22)
rs.fit(x_train,y_train)

rs.best_params_
def metrics(obj,x_test,y_test):
    pred=obj.predict(x_test)
    print(confusion_matrix(pred,y_test))
    print(classification_report(pred,y_test))
    print('acc_score',accuracy_score(pred,y_test))

dt=DecisionTreeClassifier(max_depth=5).fit(x_train,y_train)
metrics(dt,x_test,y_test)
feat_imp=pd.Series(dt.feature_importances_,index=X.columns).sort_values(ascending=False)
sns.barplot(feat_imp,feat_imp.index)
#80%accuracy  less  than log_reg
#dectree considering only percentage
#Random  Forest
rfc=RandomForestClassifier()
param={'n_estimators':[100,200,300],
      'max_depth':[2,4,6,8],
      'min_samples_leaf':[1,2,3,4,5]}
rs=RandomizedSearchCV(rfc,param_distributions=param,n_iter=5,n_jobs=1,cv=3,random_state=22)
rs.fit(x_train,y_train)


rs.best_params_
rfc=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_leaf=3).fit(x_train,y_train)
metrics(rfc,x_test,y_test)

feat_imp=pd.Series(rfc.feature_importances_,index=X.columns).sort_values(ascending=False)
sns.barplot(feat_imp,feat_imp.index)
#81% after hypertuning
# still percentages are playing key roles
from sklearn.neighbors import KNeighborsClassifier

error_rate = []

for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred = knn.predict(x_test)
    error_rate.append(np.mean(pred != y_test))
    
plt.plot(range(1,40),error_rate, marker='o', linestyle='dashed')    
#k value is 
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean').fit(x_train,y_train)
metrics(knn,x_test,y_test)

#80%% accuracy
#lets try ensemble method VotingClassifier
from sklearn.ensemble import VotingClassifier
estimators=[('log_reg',lr),('decision_tree',dt)]
vc=VotingClassifier(estimators).fit(x_train,y_train)
metrics(vc,x_test,y_test)
#still lr performs better
X
import pickle
pickle.dump(lr, open('model_upd.pkl','wb'))
model=pickle.load(open('model_upd.pkl','rb'))
model.predict([[67,91,58,55,58.8,2,1,1,0,1]])