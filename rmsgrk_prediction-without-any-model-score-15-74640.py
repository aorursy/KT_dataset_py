import pandas as pd
import numpy as np
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train['date']=pd.to_datetime(train['date'])
test['date']=pd.to_datetime(test['date'])
print(train.dtypes,'\n',test.dtypes)
def features(df):
    df['year']=df['date'].dt.year
    df['month']=df['date'].dt.month
    df['week']=df['date'].dt.weekday
    df['weeknum']=df['date'].dt.week
    return df
features(train)
features(test)
print(train.head(10))
#Data visualization in tableau showed that weekday have more importance, and day have least importance so we are ignoring day !
#Weekday 6- is sunday so we take sales value on sunday for week number 1 (1 of 52) , item number 1 store 1 month 1 and for different year( 5 years from 2013-2017)
x=train[(train['item']==1) & (train['store']==1) & (train['week']==6) & (train['weeknum']==1) & (train['month']==1) ]
x
def fun(item,store,week,weeknum,month):
    x=train[(train['item']==item) & (train['store']==store) & (train['week']==week)& (train['weeknum']==weeknum) & (train['month']==month) ]
    return np.ceil(6*(x.sales.mean()/5)).astype(int)-1
#test function
y=[]
y.append(fun(1,1,6,1,1))
y.append(fun(1,1,1,1,1))
y.append(fun(1,1,1,1,1))
y
%%time
y=[]
for index, i in test.iterrows():
    y.append(fun(i['item'],i['store'],i['week'],i['weeknum'],i['month']))
id1=pd.read_csv("../input/test.csv",usecols=['id'])

sub=pd.DataFrame({'id':id1.id,'sales':y})
sub.head(5)
sub.to_csv("NO_MODEL.csv",index=False)
