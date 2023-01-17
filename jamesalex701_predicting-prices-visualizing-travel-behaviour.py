import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import cufflinks as cf
import matplotlib.pyplot as plt
%matplotlib inline
data=pd.read_csv("../input/flight-prices-india/Data_Train.csv")
data.head()
data.describe()
data.dtypes
sns.heatmap(data.isnull())
null_perc=lambda i:dict(i.isnull().sum()*100/len(i))
perc=null_perc(data)
perc
data.isnull().sum()[data.isnull().sum()>0]
data.dropna(axis=0,inplace=True)
data.isnull().sum()[data.isnull().sum()>0]
data.shape
#checking elemnts and pattern of stops.
data['Total_Stops'].value_counts()
# function to apply on stops column
import re
def stops(x):
    if bool(re.search(r'^(\d)',x)):
            return int(x[0])
    else:
            return 0
data['Duration'].head()
# function to apply on duration column
import re
def dura(k):
    m=re.findall(r'^(?<!h\s)([0-9]?[0-9]?)[mh]$',k)
    j=re.findall(r'(?P<h>[0-9]?[0-9]?)(?:h\s)?(?P<m>[0-9][0-9]?)m',k)
    
    if len(m)==0:
        a,b=j[0]
        return int(a)+(int(b)/60)
    else:
        return int(m[0])/60   
#############
####   CREATING NEW COLUMNS
#############

import re
from datetime import datetime
data['Date_of_Journey']=pd.to_datetime(data['Date_of_Journey'],format='%d/%m/%Y')
data['day']=data['Date_of_Journey'].apply(lambda x:x.strftime('%d'))
data['day']=data['day'].apply(int)
data['weekday']=data['Date_of_Journey'].apply(lambda x:x.weekday())
data['weekdayname']=data['Date_of_Journey'].apply(lambda x:x.strftime('%A'))
data['weekofyear']=data['Date_of_Journey'].apply(lambda x:x.strftime('%W'))
##
data['month']=data['Date_of_Journey'].apply(lambda x:x.strftime('%m'))
data['monthyname']=data['Date_of_Journey'].apply(lambda x:x.strftime('%B'))
##
data['duration']=data['Duration'].apply(dura)
data['Stops']=data['Total_Stops'].apply(stops)
## Since hours is a cyclic data i.e., 23rd hour and 1st hour are close to each other, we use sin and cos 
## to represent and capture this meaning

import datetime
import time
pt = data['Dep_Time'].apply(lambda x: datetime.datetime.strptime(x,'%H:%M'))
data['dep_total'] = pt.apply(lambda x : x.hour+(x.minute/60))
data['Dep_Hours_sin']=np.sin(data['dep_total']*np.pi*2/24)
data['Dep_Hours_cos']=np.cos(data['dep_total']*np.pi*2/24)
k=data['Arrival_Time'].str.split().str[0]
tt = k.apply(lambda x: datetime.datetime.strptime(x,'%H:%M'))
data['arriv_tot'] = tt.apply(lambda x : x.hour+(x.minute/60))
data['Arrival_Hours_sin']=np.sin(data['arriv_tot']*np.pi*2/24)
data['Arrival_Hours_cos']=np.cos(data['arriv_tot']*np.pi*2/24)
## Converting days in a week to cyclic data
data['dayw_sin']=np.sin(data['weekday']*np.pi*2/7)
data['dayw_cos']=np.cos(data['weekday']*np.pi*2/7)
## Converting days in a month to cyclic data
data['day_sin']=np.sin(data['day']*np.pi*2/31)
data['day_cos']=np.cos(data['day']*np.pi*2/31)
data['Route'].head()
def make_Elements(m):
    k=m.split(' ? ')
    kd=k.copy()
    i=1
    while len(k)!=0:
        k.pop()
        i=i+1
    for j in range(0,6):
        k.append(None)
    k[-1]=kd.pop()
    kd.reverse()
    k[0]=kd.pop()
    l=1
    while len(kd)!=0:
        k[l]=kd.pop()
        l=l+1
    return k


RRoute=data['Route'].apply(make_Elements)
data['Route']=RRoute
data_R = pd.DataFrame(RRoute.values.tolist(), columns=['s_1','r_2','r_3','r_4','r_5','d_6'] ,index= data.index)
data=pd.concat([data,data_R],axis=1)

## Since we already have source and destination columns
data.drop(['s_1','d_6'],axis=1,inplace=True)
    
data.columns
data.dtypes
k=pd.crosstab(columns=data['Source'],index=data['Airline'])
## Finding Perentages of Airline choices with respect to the 'Source'
a,b=k.shape
f,ax=plt.subplots(5,1,figsize=(25,30))
for i in range(b):
    k.plot.pie(ax=ax[i],y=list(k.columns)[i],autopct='%1.1f%%',fontsize=10).legend(loc=0,bbox_to_anchor=(2.0,1.0))
a=k.plot.bar(stacked=True,figsize=(10,8));
a.legend(loc=0,bbox_to_anchor=(1.0,1.0))
#for i in ax.patches:
    #ax.text(x=i.get_x()+0.2,y=i.get_y()+0.2,s=str(round(i.get_height(),2)),fontsize=22)
    #print(i.get_xy(),i.get_y(),i.get_width(),i.get_height(),sep='\t')
sns.distplot(data['dep_total'],bins=12,color='blue',hist_kws={'edgecolor':'black'});
#,y='Price');
fig,axes=plt.subplots(3,2,figsize=(25,8))
sns.kdeplot(ax=axes[0,0],data=data[data['Source']=='Chennai']['dep_total'],label='Chennai');
sns.kdeplot(ax=axes[1,0],data=data[data['Source']=='Kolkata']['dep_total'],label='Kolkata');
sns.kdeplot(ax=axes[2,0],data=data[data['Source']=='Banglore']['dep_total'],label='Banglore');
sns.kdeplot(ax=axes[1,1],data=data[data['Source']=='Mumbai']['dep_total'],label='Mumbai');
sns.kdeplot(ax=axes[0,1],data=data[data['Source']=='Delhi']['dep_total'],label='Delhi');
axes[0,0].xaxis.set_major_locator(plt.MaxNLocator(24))
axes[0,1].xaxis.set_major_locator(plt.MaxNLocator(24))
axes[1,0].xaxis.set_major_locator(plt.MaxNLocator(24))
axes[1,1].xaxis.set_major_locator(plt.MaxNLocator(24))
axes[2,0].xaxis.set_major_locator(plt.MaxNLocator(24))
axes[0,0].xaxis.set_major_locator(plt.MaxNLocator(24))
def cv(l):
    return np.std(l)*100/np.mean(l)
sns.barplot(y=data['Price'],x=data['day']);
plt.figure(figsize=(25,8))
plt.subplot(1,2,1)
sns.barplot(y=data['Price'],x=data['day'],hue=data['monthyname']);
plt.subplot(1,2,2)
sns.lineplot(x='day',y='Price',hue='monthyname',data=data);
sns.countplot(x=data['monthyname'],hue=data['weekdayname']).legend(loc=0,bbox_to_anchor=(1.0,1.0))
plt.subplots(figsize=(25,15))
sns.violinplot(data=data,x='weekofyear',y='Price');
f,axs=plt.subplots(1,2,figsize=(25,10))
sns.violinplot(ax=axs[0],data=data,y='Price',x='Source');
sns.barplot(ax=axs[1],data=data,y='Price',x='Source');
fig, ax=plt.subplots(figsize=(30,8))
#sns.barplot(data=data,x='Airline',y='Price')
ax.barh(data['Airline'],data['duration'])
#plt.xticks(rotation=90)
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
ax.xaxis.set_major_locator(plt.MaxNLocator(24))
plt.figure(figsize=(30,10))
sns.barplot(data=data,x='Airline',y='duration',estimator=cv);
plt.rc('xtick', labelsize=20)
plt.xticks(rotation=90)
plt.figure(figsize=(30,10))
sns.violinplot(data=data,x='Airline',y='duration');
plt.rc('xtick', labelsize=20)
plt.xticks(rotation=90)
ax.yaxis.set_major_locator(plt.MaxNLocator(24))
plt.figure(figsize=(30,10))
sns.boxplot(data=data,x='Airline',y='duration');
plt.rc('xtick', labelsize=20)
plt.xticks(rotation=90)
databd=data[((data['Source']=="Banglore") & (data['Destination']=="Chennai"))]
#| ((data['Source']=="Chennai") & (data['Destination']=="Banglore"))]
fig,ax=plt.subplots(figsize=(25,8))
sns.violinplot(x=data['Airline'],y=data['Price'],gridsize=200)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=12)
ax.yaxis.set_major_locator(plt.MaxNLocator(18))
plt.ylim(0,50000)
#Which day do people from bangalore travel
plt.figure(figsize=(25,25))
plt.subplot(2,2,1)
sns.countplot(data=data,x='weekdayname')
plt.subplot(2,2,2)
sns.countplot(data=data,x='Source',hue='weekdayname')
plt.subplot(2,2,3)
sns.countplot(data=data,x='weekdayname',hue='Source')
plt.subplot(2,2,4)
sns.countplot(data=data,x='Source')
plt.figure(figsize=(25,25))
plt.subplot(2,2,1)
sns.barplot(data=data,x='weekdayname',y='Price')
plt.subplot(2,2,2)
sns.barplot(data=data,x='Source',hue='weekdayname',y='Price')
plt.subplot(2,2,3)
sns.barplot(data=data,x='weekdayname',y='Price',hue='Source')
plt.subplot(2,2,4)
sns.barplot(data=data,x='Source',y='Price')
sns.jointplot(data=data,x='duration',y='Price',kind='kde');
data.columns
gdata=data.copy()
data.drop(['Date_of_Journey','Route','Dep_Time','Arrival_Time','Duration','Total_Stops','weekdayname','monthyname','weekofyear'],inplace=True,axis=1)
data.drop(['day','weekday','dep_total','arriv_tot'],axis=1,inplace=True)
#data.drop(['day_sin','day_cos','dayw_sin','dayw_cos','Dep_Hour_sin','Dep_Hours_cos','Arrival_Hours_sin','Arrival_Hours_cos'],axis=1,inplace=True)
#Set DataTypes
data.dtypes
plt.figure(figsize=(25,8))
sns.heatmap(data.corr(),cmap='coolwarm',annot=True);
s_1=pd.get_dummies(data['Source'])
r_2=pd.get_dummies(data['r_2'])
r_3=pd.get_dummies(data['r_3'])
r_4=pd.get_dummies(data['r_4'])
r_5=pd.get_dummies(data['r_5'])
d_6=pd.get_dummies(data['Destination'])

data.drop(['Source','r_2','r_3','r_4','r_5','Destination'],inplace=True,axis=1)

dr1=list(s_1.columns)
dr2=list(r_2.columns)
dr3=list(r_3.columns)
dr4=list(r_4.columns)
dr5=list(r_5.columns)
dr6=list(d_6.columns)
len(dr1),len(dr2),len(dr3),len(dr4),len(dr5),len(dr6)
## To avoid multicollinearity we remove one column while getting dummy columns.
s_1.drop(dr1[0],axis=1,inplace=True)
r_2.drop(dr2[0],axis=1,inplace=True)
r_3.drop(dr3[0],axis=1,inplace=True)
r_4.drop(dr4[0],axis=1,inplace=True)
#r_5.drop(dr5[0],axis=1,inplace=True) as only one element
d_6.drop(dr6[0],axis=1,inplace=True)

s_1=s_1.add_prefix('Source_')
r_2=r_2.add_prefix('r_2_')
r_3=r_3.add_prefix('r_3_')
r_4=r_4.add_prefix('r_4_')
r_5=r_5.add_prefix('r_5_')
d_6=d_6.add_prefix('Destination_')

Entire_Route=pd.concat([s_1,r_2,r_3,r_4,r_5,d_6],axis=1)
AirLine=pd.get_dummies(data['Airline'])
AirLine=AirLine.drop('IndiGo',axis=1)

data.drop(['Airline'],inplace=True,axis=1)
A=pd.get_dummies(data['Additional_Info'])
A=A.drop('No info',axis=1)

data.drop(['Additional_Info'],axis=True,inplace=True)
mont=pd.get_dummies(data['month'])
mont=mont.drop(['03'],axis=1)

data.drop(['month'],axis=1,inplace=True)
data.dtypes
data.columns
data.describe()
X=pd.concat([data,mont,A,AirLine,Entire_Route],axis=1)
X.dtypes
list(X.columns)
#We need to check outliers for the columns 'duration', 'Stops',
# for other float columns such as  'Dep_Hours_sin', 'Dep_Hours_cos', 'Arrival_Hours_sin', 'Arrival_Hours_cos',
# 'dayw_sin', 'dayw_cos', 'day_sin', 'day_cos'
#As they will only lie in range(-1,1) with no outliers
X[['duration']].boxplot()
q1, q3= np.percentile(X['duration'],[25,75])


(X['duration'][X['duration']>q3+(1.5*(q3-q1))].count()+X['duration'][X['duration']<q1-(1.5*(q3-q1))].count())*100/X['Price'].count()
sns.scatterplot(x=data['duration'],y=data['Price'])
## Capping Outliers.

q1, q3= np.percentile(X['duration'],[25,75])

X['duration'][X['duration']>q3+(1.5*(q3-q1))]=q3+(1.5*(q3-q1))
X[['duration']].boxplot()
X[['Stops']].boxplot()
q1, q3= np.percentile(X['Stops'],[25,75])


(X['Stops'][X['Stops']>q3+(1.5*(q3-q1))].count()+X['duration'][X['duration']<q1-(1.5*(q3-q1))].count())*100/X['Price'].count()
# Very few outliers and Discrete data, Hence we do not cap.
X.head()
X_data=X.drop(['Price'],axis=1)
y=X['Price']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data,y, test_size=0.33, random_state=102)



from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_trainp=sc.transform(X_train)
X_testp= sc.transform(X_test)

from sklearn import metrics
from sklearn.neighbors import KNeighborsRegressor

error_rate=[]
for i in range(1,40):
    knn=KNeighborsRegressor(n_neighbors=i)
    knn.fit(X_trainp,y_train)
    y_pred=knn.predict(X_testp)
    error_rate.append(np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
sns.jointplot(range(1,40),error_rate)
error_rate.index(min(error_rate))
#n=3 at index 2
knn=KNeighborsRegressor(n_neighbors=3)
knn.fit(X_trainp,y_train)
y_pred=knn.predict(X_testp)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
knn=1-rsmle
from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)

y_pred=lm.predict(X_test)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
lin_reg=1-rsmle
for i,v in zip(lm.coef_,X.columns):
    print('Feature:',v,'Score:',i)
import statsmodels.api as sm
X2 = sm.add_constant(X_train)
lop=sm.OLS(y_train,X2.astype(float))
p=lop.fit()
p.summary()

import statsmodels.api as sm
X3 = sm.add_constant(X_test)
lo=sm.OLS(y_test,X3.astype(float))
pp=lo.fit()
pp.summary()
from sklearn.linear_model import Ridge
lm=Ridge(alpha=0.9)
lm.fit(X_train,y_train)

y_pred=lm.predict(X_test)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
ridge_reg=1-rsmle
from sklearn.linear_model import Lasso
lm=Lasso(alpha=0.8)
lm.fit(X_train,y_train)

y_pred=lm.predict(X_test)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
lasso_reg=1-rsmle
from sklearn.svm import SVR
model = SVR(kernel='linear',C=0.1,gamma=0.01)
model.fit(X_trainp,y_train)
y_pred=model.predict(X_testp)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
svm=1-rsmle
from sklearn.tree import DecisionTreeRegressor
d=DecisionTreeRegressor()
d.fit(X_trainp,y_train)
y_pred=d.predict(X_testp)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
dtree=1-rsmle
importance = d.feature_importances_
# summarize feature importance
for i,v in zip(importance,list(X.columns)):
    print('Feature:',v,i)
#plt.bar([x for x in range(len(importance))], importance)
#plt.show()
from sklearn.ensemble import RandomForestRegressor
r=RandomForestRegressor()
r.fit(X_trainp,y_train)
y_pred=r.predict(X_testp)
sns.scatterplot(y_test,y_pred)
from sklearn import metrics
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
import sklearn.metrics as mm
mm.explained_variance_score(y_test,y_pred)
rsmle=mm.mean_squared_log_error((y_test+1),(y_pred+1))
rf=1-rsmle
importance = r.feature_importances_
# summarize feature importance
for i,v in zip(importance,list(X.columns)):
    print('Feature:',v,i)
#plt.bar([x for x in range(len(importance))], importance)
#plt.show()
k=[(a,b) for b,a in zip(importance,list(X.columns))]
M=sorted(k,key=lambda x:x[1],reverse=True)
#In order of importance
M
print('RandomForest_Score =',rf)
print('DecisionTree_Score =',dtree)
print('KNN_Score =',knn)
print('Linear_Regression_Score =',lin_reg)
print('Ridge_regression_Score =',ridge_reg)
print('Lasso_regression_Score =',lasso_reg)
print('Support_Vector_Score =',svm)