# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# load train_set&test_set
train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
test['casual']=0
test['registered']=0
test['count']=0
#remove Outlier piont
train = train[np.abs(train["count"]-train["count"].mean())<=(3*train["count"].std())] 
#create a union data
union_data=pd.concat([train,test],ignore_index=True)
#add date columns
union_data['day']=pd.to_datetime(union_data.datetime).dt.day
union_data['year']=pd.to_datetime(union_data.datetime).dt.year
union_data['month']=pd.to_datetime(union_data.datetime).dt.month
union_data['weekday']=pd.to_datetime(union_data.datetime).dt.weekday
union_data['date']=pd.to_datetime(union_data.datetime).dt.date
union_data['hour']=pd.to_datetime(union_data.datetime).dt.hour
union_data['year_season']=union_data.apply(lambda x:'{}_{}'.format(str(x['year']),str(x['season'])),axis=1)
union_data['year_month']=union_data.apply(lambda x:'{}_{}'.format(str(x['year']),str(x['month'])),axis=1)
#missing data fill
union_data['windspeed']=union_data[['year','month','hour','windspeed']].groupby(['year','month','hour']).transform(lambda x:x.replace(0,np.median([i for i in x if i>0])))
union_data['windspeed']=pd.cut(union_data['windspeed'],bins=[0,20,60],labels=['0','1'])

#union_data['hour_type']=0
#union_data['hour_type'][(union_data['hour']<=1)]='1'
#union_data['hour_type'][(union_data['hour']>=2)& (union_data['hour']<=4)]='2'
#union_data['hour_type'][(union_data['hour']>=4)& (union_data['hour']<=6)]='3'
#union_data['hour_type'][(union_data['hour']>=6)& (union_data['hour']<=8)]='4'
#union_data['hour_type'][(union_data['hour']>=9)& (union_data['hour']<=15)]='5'
#union_data['hour_type'][(union_data['hour']>=16)& (union_data['hour']<=18)]='6'
#union_data['hour_type'][(union_data['hour']>=19)& (union_data['hour']<=20)]='7'
#union_data['hour_type'][(union_data['hour']>=21)]='8'


#add day_type columns
union_data['day_type']=0
union_data['day_type'][(union_data['holiday']==0)& (union_data['workingday']==0)]='weekend'
union_data['day_type'][(union_data['holiday']==0)& (union_data['workingday']==1)]='workingday'
union_data['day_type'][(union_data['holiday']==1)]='holiday'

#create train set
train=union_data[:10739]
#windspeed counts
plt.figure(figsize=(100,5))
g=sns.factorplot(x='windspeed',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#season trend
g=sns.factorplot(x='season',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#month trend
g=sns.factorplot(x='month',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#day trend
g=sns.factorplot(x='day',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#weekday trend
g=sns.factorplot(x='weekday',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#hour trend
g=sns.factorplot(x='hour',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#weather analyse
g=sns.factorplot(x='weather',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#workingday analyse
g=sns.factorplot(x='workingday',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
#tempture analyse
g=sns.factorplot(x='temp',y='count',data=train,col='year',kind='bar',estimator=sum,ci=None,size=10,aspect=1)
from sklearn import tree
clf = tree.tree.DecisionTreeRegressor(max_depth=4,criterion='mse',min_samples_leaf=800)
clf = clf.fit(train['hour'].reshape(-1,1),np.ravel(train['count']))
import graphviz 
dot_data = tree.export_graphviz(clf, out_file=None,feature_names=['hour'],  
                                filled=True, rounded=True,  
                                special_characters=True,) 
graph = graphviz.Source(dot_data) 
graph
#dot_data = tree.export_graphviz(clf, out_file=None,feature_names=train[['hour']].columns.values,class_names=train[['count']].columns.values) 
#graph = graphviz.Source(dot_data)  
#graph 

from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
#X=train[['season', 'holiday', 'workingday', 'weather',
#      'atemp', 'humidity', 'windspeed','day', 'year', 'month', 'weekday','year_season']]

train_y=np.log1p(train[['count']]+1)
undumm=union_data[['datetime','year_month','atemp','temp', 'humidity','windspeed']]
get_dumm=union_data[['weather','workingday','hour','day_type','weekday']]
#enc = OneHotEncoder()
#enc.fit(train[['season', 'holiday', 'workingday', 'weather','day', 'year', 'month', 'weekday']])
#enc.transform(train[['season', 'holiday', 'workingday', 'weather','day', 'year', 'month', 'weekday']]).toarray().shape
dumm=pd.get_dummies(get_dumm,columns=get_dumm.columns)
train_X=pd.concat([undumm[:10739],dumm[:10739]],axis=1)
test_X=pd.concat([undumm[10739:],dumm[10739:]],axis=1)

train_X.columns
regr = RandomForestRegressor(n_estimators=300)
regr.fit(train_X.loc[:,'year_month':], np.ravel(train_y))
reg=GradientBoostingRegressor(n_estimators=2000, learning_rate=0.01,max_depth=4)
reg.fit(train_X.loc[:,'year_month':], np.ravel(train_y))

plt.figure(figsize=(100,5))
sns.barplot(x=train_X.loc[:,'year_month':].columns,y=regr.feature_importances_)
plt.figure(figsize=(100,5))
sns.barplot(x=train_X.loc[:,'year_month':].columns,y=reg.feature_importances_)
np.exp(regr.predict(test_X.loc[:,'year_month':]))-1

np.exp(reg.predict(test_X.loc[:,'year_month':]))-1
union_data['count'][10739:]=np.exp(reg.predict(test_X.loc[:,'year_month':]))-1
submission=pd.DataFrame({
        "datetime": union_data[10739:].datetime,
        "count": union_data[10739:]['count']
    })
submission.to_csv('bike_predictions_gbm_separate_without_fe.csv', index=False)