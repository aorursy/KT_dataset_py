# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
weather=pd.read_csv('/kaggle/input/weather-dataset-rattle-package/weatherAUS.csv')
weather.head()
x=weather.iloc[:,:-1]

y=weather.iloc[:,-1]
x.shape
x.info()
x.isnull().mean() # == to isnull().sum()/x.shape[0]
y.shape
np.unique(y)
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
# reset index 

for i in [xtrain,xtest,ytrain,ytest]:

    i.index=range(i.shape[0])
# check the target balance

ytrain.value_counts()
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder().fit(ytrain)
# to avoid unknow labels in the test

ytrain=pd.DataFrame(encoder.transform(ytrain))

ytest=pd.DataFrame(encoder.transform(ytest))
xtrain.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
xtest.describe([0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.99]).T
# since the max cloud is 8 and I find in test and train dataset, the max cloud9am and cloud3pm is 9

print(xtrain.loc[xtrain.loc[:,'Cloud9am']==9,'Cloud9am'])

print(xtest.loc[xtest.loc[:,'Cloud9am']==9,'Cloud9am'])

print(xtest.loc[xtest.loc[:,'Cloud3pm']==9,'Cloud3pm'])

xtrain=xtrain.drop(index=69085)

xtest=xtest.drop(index=37491)

xtest=xtest.drop(index=20157)

ytrain=ytrain.drop(index=69085)

ytest=ytest.drop(index=37491)

ytest=ytest.drop(index=20157)
xtrain['Rainfall'].isnull().sum()
xtrain.loc[xtrain.loc[:,'Rainfall']>=1,'RainToday']='YES'

xtrain.loc[xtrain.loc[:,'Rainfall']<1,'RainToday']='NO'

xtrain.loc[xtrain.loc[:,'Rainfall']==np.nan,'RainToday']=np.nan
xtrain['RainToday'].value_counts()
xtest.loc[xtest.loc[:,'Rainfall']>=1,'RainToday']='YES'

xtest.loc[xtest.loc[:,'Rainfall']<1,'RainToday']='NO'

xtest.loc[xtest.loc[:,'Rainfall']==np.nan,'RainToday']=np.nan
xtrain.head()
#split date to month

xtrain['Date']=xtrain['Date'].apply(lambda x: int(x.split('-')[1]))
xtrain=xtrain.rename(columns={'Date':'Month'})
xtest['Date']=xtest['Date'].apply(lambda x: int(x.split('-')[1]))

xtest=xtest.rename(columns={'Date':'Month'})
citylist=pd.read_csv('/kaggle/input/city-location-and-climate/cityll.csv',index_col=0)

citylist.head()
citycl=pd.read_csv('/kaggle/input/city-location-and-climate/Cityclimate.csv')

citycl.head()
citylist['Latitudenum']=citylist['Latitude'].apply(lambda x:float(x[:-1]))

citylist['Longitudenum']=citylist['Longitude'].apply(lambda x:float(x[:-1]))
cityld=citylist.iloc[:,[0,5,6]]
cityld['Climate']=citycl.iloc[:,1]
cityld.head()
cityld['Climate'].value_counts()
samplecity=pd.read_csv('/kaggle/input/samplecity/samplecity.csv',index_col=0)

samplecity.head()
samplecity['Latitudenum']=samplecity['Latitude'].apply(lambda x:float(x[:-1]))

samplecity['Longitudenum']=samplecity['Longitude'].apply(lambda x:float(x[:-1]))

samplecityd=samplecity.iloc[:,[0,5,6]]
samplecityd.head()
from math import radians,cos,sin,acos

cityld['slat']=cityld['Latitudenum'].apply(lambda x:radians(x))

cityld['slon']=cityld['Longitudenum'].apply(lambda x:radians(x))

samplecityd['elat']=samplecityd['Latitudenum'].apply(lambda x:radians(x))

samplecityd['elon']=samplecityd['Longitudenum'].apply(lambda x:radians(x))
import sys

for i in range(samplecityd.shape[0]):

    slat=cityld['slat']

    slon=cityld['slon']

    elat=samplecityd.loc[i,'elat']

    elon=samplecityd.loc[i,'elon']

    dist=6371.01*np.arccos(np.sin(slat)*np.sin(elat)+np.cos(slat)*np.cos(elat)*np.cos(slon.values-elon))

    city_index=np.argsort(dist)[0]

    samplecityd.loc[i,'closest_city']=cityld.loc[city_index,'City']

    samplecityd.loc[i,'Climate']=cityld.loc[city_index,'Climate']
samplecityd.head()
samplecityd['Climate'].value_counts()
localfinal=samplecityd.iloc[:,[0,-1]]
localfinal.columns=['Location','Climate']
localfinal=localfinal.set_index(keys='Location')
localfinal.head()
import re
xtrain['Location']=xtrain['Location'].map(localfinal.iloc[:,0])
xtrain.head()
xtrain['Location']=xtrain['Location'].apply(lambda x:re.sub(',','',x.strip()))
xtest['Location']=xtest['Location'].map(localfinal.iloc[:,0]).apply(lambda x:re.sub(',','',x.strip()))
xtrain=xtrain.rename(columns={'Location':'Climate'})

xtest=xtest.rename(columns={'Location':'Climate'})
xtrain.dtypes
cate=list(xtrain.columns[xtrain.dtypes=='object'])
cloud=['Cloud9am','Cloud3pm']

cate.extend(cloud)
from sklearn.impute import SimpleImputer

si=SimpleImputer(missing_values=np.nan,strategy='most_frequent')

si.fit(xtrain.loc[:,cate])
xtrain.loc[:,cate]=si.transform(xtrain.loc[:,cate])

xtest.loc[:,cate]=si.transform(xtest.loc[:,cate])
print(xtrain.loc[:,cate].isnull().mean())

print(xtest.loc[:,cate].isnull().mean())
from sklearn.preprocessing import OrdinalEncoder

oe=OrdinalEncoder()

oe=oe.fit(xtrain.loc[:,cate])
xtrain.loc[:,cate]=oe.transform(xtrain.loc[:,cate])

xtest.loc[:,cate]=oe.transform(xtest.loc[:,cate])
col=list(xtrain.columns)

for i in cate:

    col.remove(i)
im=SimpleImputer(missing_values=np.nan,strategy='mean')

im.fit(xtrain.loc[:,col])
xtrain.loc[:,col]=im.transform(xtrain.loc[:,col])

xtest.loc[:,col]=im.transform(xtest.loc[:,col])
xtrain.isnull().sum()
col.remove('Month')
from sklearn.preprocessing import StandardScaler

ss=StandardScaler()

ss=ss.fit(xtrain.loc[:,col])
xtrain.loc[:,col]=ss.transform(xtrain.loc[:,col])

xtest.loc[:,col]=ss.transform(xtest.loc[:,col])
xtrain.head()
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_auc_score,recall_score
for kernel in ['linear','poly','rbf','sigmoid']:

    clf=SVC(kernel=kernel

           ,gamma='auto'

           ,degree=1

           ,cache_size=5000 # the memory you want to use

           ).fit(xtrain,ytrain)

    result=clf.predict(xtest)

    score=clf.score(xtest,ytest)

    re_call=recall_score(ytest,result)

    auc=roc_auc_score(ytest,clf.decision_function(xtest))

    print('%s, testing accuracy %f, recall is %f, auc is %f'% (kernel,score,re_call,auc))
# to improve the recall

for kernel in ['linear','poly']:

    clf=SVC(kernel=kernel

           ,gamma='auto'

           ,degree=1

           ,cache_size=5000# the memory you want to use

           ,class_weight='balanced'

           ).fit(xtrain,ytrain)

    result=clf.predict(xtest)

    score=clf.score(xtest,ytest)

    re_call=recall_score(ytest,result)

    auc=roc_auc_score(ytest,clf.decision_function(xtest))

    print('%s, testing accuracy %f, recall is %f, auc is %f'% (kernel,score,re_call,auc))
valuec=ytest[0].value_counts()

valuec[0]/valuec.sum()
from sklearn.metrics import confusion_matrix as CM
clf=SVC(kernel='linear'

           ,gamma='auto'

           ,degree=1

           ,cache_size=5000 # the memory you want to use

           ).fit(xtrain,ytrain)

result=clf.predict(xtest)
cm=CM(ytest,result,labels=(1,0))

cm
#if we want more accuracy, we could adjust the class_weight more slightly
irange=np.linspace(0,0.01,10)
for i in irange:

    clf=SVC(kernel='linear'

           ,gamma='auto'

           ,degree=1

           ,cache_size=5000# the memory you want to use

           ,class_weight={1:1+i}

           ).fit(xtrain,ytrain)

    result=clf.predict(xtest)

    score=clf.score(xtest,ytest)

    re_call=recall_score(ytest,result)

    auc=roc_auc_score(ytest,clf.decision_function(xtest))

    print(' class_weight %f, testing accuracy %f, recall is %f, auc is %f'% (1+i,score,re_call,auc))
from sklearn.linear_model import LogisticRegression as LR

lr=LR(solver='liblinear').fit(xtrain,ytrain)

lr.score(xtest,ytest)
for i in np.linspace(3,5,10):

    lr=LR(solver='liblinear',C=i).fit(xtrain,ytrain)

    print(i,lr.score(xtest,ytest))