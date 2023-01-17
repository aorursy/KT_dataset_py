# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
with warnings.catch_warnings():
	warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
data=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
data
data.info()
data['status'].value_counts().plot.bar()
import matplotlib.pyplot as plt
import seaborn as sn
sn.countplot('gender',hue='status', data=data )
sn.countplot('ssc_b',hue='status', data=data )
sn.countplot('hsc_b',hue='status', data=data )
sn.countplot('hsc_s',hue='status', data=data )
sn.countplot('degree_t',hue='status', data=data )
sn.countplot('workex',hue='status', data=data )
sn.countplot('specialisation',hue='status', data=data )
bins=[35,60,75,90,100] 
group=['35-60','60-75','75-90','90-100'] 
data['mba_p_bin']=pd.cut(data['mba_p'],bins,labels=group)
mba_p_bin=pd.crosstab(data['mba_p_bin'],data['status']) 
mba_p_bin.div(mba_p_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('mba percentage') 
P = plt.ylabel('status')
bins=[35,60,75,90,100] 
group=['35-60','60-75','75-90','90-100'] 
data['degree_p_bin']=pd.cut(data['degree_p'],bins,labels=group)
degree_p_bin=pd.crosstab(data['degree_p_bin'],data['status']) 
degree_p_bin.div(degree_p_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('degree percentage')
P = plt.ylabel('status')
plt.pie(data.pivot_table(data,index=['hsc_s'])['salary'],labels=['Arts', 'Science','Commerce'],explode =(0,0,0.1),colors = ['red', 'blue','yellow'], startangle=90, autopct='%.1f%%', shadow = True)
plt.title('salary grade based on Specialization in Higher Secondary Education',fontsize=20)
print(data.pivot_table(data,index=['hsc_s'])['salary'])
plt.pie(data.pivot_table(data,index=['degree_t'])['salary'],labels=['Comm&Mgmt', 'Others','Sci&Tech'],explode =(0,0,0.1),colors = ['red', 'blue','yellow'], startangle=90, autopct='%.1f%%', shadow = True)
plt.title('salary grade based on Field of degree education',fontsize=20)
print(data.pivot_table(data,index=['degree_t'])['salary'])
plt.pie(data.pivot_table(data,index=['workex'])['salary'],labels=['no','yes'],explode =(0,0.04),colors = ['lightskyblue','blue'], startangle=90, autopct='%.1f%%', shadow = True)
plt.title('salary grade based Work Experience',fontsize=20)
print(data.pivot_table(data,index=['workex'])['salary'])
plt.pie(data.pivot_table(data,index=['specialisation'])['salary'],labels=['Mkt&Fin','Mkt&HR'],explode =(0.04,0),colors = ['blue','lightskyblue'], startangle=90, autopct='%.1f%%', shadow = True)
plt.title('salary grade based Work Experience',fontsize=20)
print(data.pivot_table(data,index=['specialisation'])['salary'])
data['salary'].fillna(0,inplace=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['gender']=le.fit_transform(data['gender'])
data['ssc_b']=le.fit_transform(data['ssc_b'])
data['hsc_b']=le.fit_transform(data['hsc_b'])
data['hsc_s']=le.fit_transform(data['hsc_s'])
data['degree_t']=le.fit_transform(data['degree_t'])
data['workex']=le.fit_transform(data['workex'])
data['specialisation']=le.fit_transform(data['specialisation'])
data['status']=le.fit_transform(data['status'])
x=data.drop(['sl_no','salary','mba_p_bin','degree_p_bin'],axis=1)
y=data['salary']
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
i=1 
kf = StratifiedKFold(n_splits=5,random_state=7,shuffle=True) 
for train_index,test_index in kf.split(x,y):     
    print('\n{} of kfold {}'.format(i,kf.n_splits))     
    xtr,xvl = x.loc[train_index],x.loc[test_index]     
    ytr,yvl = y[train_index],y[test_index]         
    model = DecisionTreeClassifier(random_state=1)     
    model.fit(xtr, ytr)     
    pred_test = model.predict(xvl)     
    score = accuracy_score(yvl,pred_test)     
    print('accuracy_score',score*100)     
    i+=1 
from sklearn.preprocessing import StandardScaler
model=DecisionTreeClassifier()
rescaledx=StandardScaler().fit_transform(x)
model.fit(rescaledx,y)
predictions=model.predict(rescaledx)
print(accuracy_score(y, predictions))
