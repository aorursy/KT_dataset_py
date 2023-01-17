# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

print(os.listdir('../input/ecommercetraindata'))
headerstr=['user_id', 'startTime', 'endTime', 'ProductList']
dftrain = pd.read_csv('../input/ecommercetraindata/trainingData.csv', header=None, names=headerstr)
dflabel = pd.read_csv('../input/ecommercetraindata/trainingLabel.csv', header=None, names=['gender'])
dftrain.head()
dflabel.head()
from collections import Counter
def get_products(prod):
    if ';' in prod:
        prodlist=prod.split(';')
        countitem=len(prodlist)
        temp=prodlist[0].split('/')
        firstlv1 = temp[0]
        firstlv2 = temp[1]
        lv1list=[]
        lv2list=[]
        for item in prodlist:
            a=item.split('/')
            lv1list.append(a[0])
            lv2list.append(a[1])
        totallv1=len(set(lv1list))
        totallv2=len(set(lv2list))
        mostfreqlv1=max(lv1list, key=Counter(lv1list).get)
    else:
        lvlist=prod.split('/')
        firstlv1=lvlist[0]
        firstlv2=lvlist[1]
        countitem=1
        totallv1=1
        totallv2=1
        mostfreqlv1=firstlv1
    return (countitem, firstlv1, firstlv2, totallv1, totallv2, mostfreqlv1)

newcol= ['NumProduct', 'FirstA', 'FirstB', 'TotalA', 'TotalB', 'MostA']
newcollst= dftrain['ProductList'].apply(lambda x: get_products(x))
newcoldf = pd.DataFrame(newcollst.tolist(), columns=newcol)
newcoldf.head()
data=pd.concat([dftrain, newcoldf], axis=1)
data.head()
data['startTime']=pd.to_datetime(data['startTime'])
data['endTime']=pd.to_datetime(data['endTime'])
data['duration']=data['endTime']-data['startTime']
data['duration']=data['duration'].astype('timedelta64[m]')
data['weekday']=data['startTime'].dt.dayofweek
data['hour_24h']=data['startTime'].dt.hour

droplist=['user_id', 'startTime', 'endTime', 'ProductList']
data=data.drop(droplist, axis=1)
data.head()
data=pd.get_dummies(data)
data.head()
x=data
y=dflabel['gender'].map({'female':1, 'male':0})
xtrain, xcv, ytrain, ycv = train_test_split(x,y,test_size=0.33)
print(y.value_counts())
print(x.shape)
print(y.shape)
print(xtrain.shape, ycv.shape)
print(ytrain.shape, ycv.shape)
clf=RandomForestClassifier(class_weight={1:0.1, 0:0.35})
clf.fit(xtrain, ytrain)
ycvpred=predictions=clf.predict(xcv)
print(ycvpred[:10])
print(ycv[:10])
print('f1 score:')
print(f1_score(ycv, clf.predict(xcv)))
print('-----')
print('accuracy score:')
print(accuracy_score(ycv, clf.predict(xcv)))
print('-----')
print('recall score micro:')
print(recall_score(ycv, clf.predict(xcv), average='micro'))
print('-----')
print('recall score macro:')
print(recall_score(ycv, clf.predict(xcv), average='macro'))