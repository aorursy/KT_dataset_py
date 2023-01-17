# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/lc_2016_2017.csv')
data.tail()
data.info()
data.shape[1]
data.loan_status.value_counts()
d = {'Late (31-120 days)':1,'Late (16-30 days)':1,'Default':1,'Current':0,'Fully Paid':0,'Charged Off':0,'In Grace Period':0}
d
data['target'] = data['loan_status'].map(d)
data.target.value_counts()
data.drop('member_id',axis=1,inplace=True)
miss =data.apply(lambda x: sum(x.isnull()))/data.shape[0]

miss =data.apply(lambda x: sum(x.isnull()))/data.shape[0]
miss_gt_60=list(miss[miss >.6].index)
miss_gt_60
def miss(var):
    if var == var:
        return(0)
    else:
        return(1)
miss(np.NaN)
data['desc_1']= data['desc'].apply(miss)
'ttt'+'_1'
for i in miss_gt_60:
    data[i+'_1']= data[i].apply(miss)
data[miss_gt_60].isnull().sum()
tr=[]
for i in miss_gt_60: 
    tr.append(i+'_1')
data[tr].sum()
data.drop(miss_gt_60,axis=1,inplace=True)
miss =data.apply(lambda x: sum(x.isnull()))/data.shape[0]
miss_vars=list(miss[miss >0].index)
miss_vars
cat_miss=[]
num_miss=[]
for i in miss_vars:
    if data[i].dtype == 'O':
        cat_miss.append(i)
    else:
        num_miss.append(i)
cat_miss
num_miss
for i in num_miss:
    data[i].fillna(0,inplace=True)
for i in cat_miss:
    data[i].fillna('Missing',inplace=True)
data[cat_miss].isnull().sum()
data.describe(include='all',percentiles=[.01,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]).transpose().to_csv('edd.csv')
cat_=[]
num_=[]
for i in data.columns:
    if data[i].dtype == 'O':
        cat_.append(i)
    else:
        num_.append(i)
cat_
import seaborn as sns
%matplotlib inline
sns.heatmap(data[num_].corr())
data[num_].corr().to_csv('corr.csv')
cat_lt10=[]
cat_gt10=[]
for i in cat_:
    print(i ,':',data[i].nunique())
    if data[i].nunique() <= 10:
        cat_lt10.append(i)
    else:
        cat_gt10.append(i)
cat_gt10
cat_lt10
data[cat_lt10].head()
grade = pd.get_dummies(data['grade'],drop_first=True)
data.shape
for i in cat_lt10:
    print('processed:',i)    
    data =pd.concat([data,pd.get_dummies(data[i],drop_first=True)],axis=1)
data.shape
data.drop(cat_lt10,axis=1,inplace=True)
data.shape
len(cat_lt10)
cat_gt10
data[cat_gt10].head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for i in cat_gt10:
    print('processing:',i)
    data[i] = le.fit_transform(data[i])

data[cat_gt10].head()
def fun(x):
    if x > 5000:
        return(5000)
    else:
        return(x)
data.info()
data['target'].value_counts()
data.drop('id',axis=1,inplace=True)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(data.drop('target',axis=1),data['target'],test_size=.3,random_state = 2018)
print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)
from sklearn.linear_model import LogisticRegression
lm = LogisticRegression()
lm.fit(X_train,y_train)
pred_train=lm.predict(X_train)
pred_test=lm.predict(X_test)

proba_train = lm.predict_proba(X_train)[:,1]
proba_test = lm.predict_proba(X_test)[:,1]

proba_test[:5]
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,auc,roc_curve
print('******************************')
print('TRAIN')
print('******************************')
print('Accuracy')
print(accuracy_score(pred_train,y_train))
print('P R F1')
print(classification_report(y_train,pred_train))
print('CM')
print(confusion_matrix(y_train,pred_train))
print('roc')
tpr,fpr,th=roc_curve(y_train,proba_train)
print(auc(tpr,fpr))
print('******************************')
print('test')
print('******************************')
print('Accuracy')
print(accuracy_score(pred_test,y_test))
print('P R F1')
print(classification_report(y_test,pred_test))
print('CM')
print(confusion_matrix(y_test,pred_test))
print('roc')
tpr,fpr,th=roc_curve(y_test,proba_test)
print(auc(tpr,fpr))
X_train['proba']= proba_train
for i in th[:100]:
    temp = X_train['proba'].apply(lambda x: 1 if x > i else 0)
    print(i,':',accuracy_score(temp,y_train))