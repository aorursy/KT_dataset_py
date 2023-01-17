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
    return df
features(train)
features(test)
print(train.head(10))
x=train[(train['item']==1) & (train['store']==1) & (train['week']==6) & (train['month']==1) ]
x
def fun(item,store,week,month):
    x=train[(train['item']==item) & (train['store']==store) & (train['week']==week) & (train['month']==month) ]
    return (np.round((1.1652*(x.sales.mean())))).astype(int)
#test function
y=[]
y.append(fun(1,1,0,1))
y.append(fun(1,1,1,1))
y.append(fun(1,1,2,1))
y.append(fun(1,1,3,1))
y.append(fun(1,1,4,1))
y.append(fun(1,1,5,1))
y.append(fun(1,1,6,1))
y
%%time
y=[]
for index, i in test.iterrows():
    y.append(fun(i['item'],i['store'],i['week'],i['month']))
id1=pd.read_csv("../input/test.csv",usecols=['id'])
sub=pd.DataFrame({'id':id1.id,'sales':y})
sub.head(5)
sub.to_csv("NO_MODEL.csv",index=False)
