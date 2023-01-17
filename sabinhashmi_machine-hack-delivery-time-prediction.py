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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

import warnings

warnings.filterwarnings('ignore')
from scipy.stats import zscore

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score,confusion_matrix

from time import time

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier,ExtraTreesClassifier

lr=LogisticRegression()

dt=DecisionTreeClassifier()

knn=KNeighborsClassifier()

rf=RandomForestClassifier()

ada=AdaBoostClassifier()

bag=BaggingClassifier()

xtree=ExtraTreesClassifier()

from sklearn.model_selection import GridSearchCV

from sklearn.feature_extraction.text import CountVectorizer

count=CountVectorizer()

from sklearn.decomposition import PCA
train=pd.read_excel('/kaggle/input/predicting-food-delivery-time/Participants Data/Data_Train.xlsx')

test=pd.read_excel('/kaggle/input/predicting-food-delivery-time/Participants Data/Data_Test.xlsx')

sample=pd.read_excel('/kaggle/input/predicting-food-delivery-time/Participants Data/Sample_Submission.xlsx')
train['Restaurant']=le.fit_transform(train['Restaurant'])

train['Location']=le.fit_transform(train['Location'])

train['Minimum_Order']=pd.to_numeric(train['Minimum_Order'].str.replace('₹',' '))

train['Average_Cost']=pd.to_numeric(train['Average_Cost'].str.replace('[^0-9]',''))

train['Rating']=pd.to_numeric(train['Rating'].apply(lambda x : np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))

train['Votes']=pd.to_numeric(train['Votes'].apply(lambda x : np.nan if x=='-' else x))

train['Reviews']=pd.to_numeric(train['Reviews'].apply(lambda x : np.nan if x=='-' else x))

train['Delivery_Time']=pd.to_numeric(train['Delivery_Time'].str.replace('[^0-9]',''))
q1=train['Rating'].quantile(0.25)

q3=train['Rating'].quantile(0.75)

iqr=q3-q1

train['Rating']=train['Rating'].apply(lambda x: np.nan if x>q3+1.5*iqr or x<q1-1.5*iqr else x)

train['Rating']=train['Rating'].fillna(train['Rating'].median())





q1=train['Votes'].quantile(0.25)

q3=train['Votes'].quantile(0.75)

iqr=q3-q1

train['Votes']=train['Votes'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)

train['Votes']=train['Votes'].fillna(train['Votes'].mode()[0])





q1=train['Reviews'].quantile(0.25)

q3=train['Reviews'].quantile(0.75)

iqr=q3-q1

train['Reviews']=train['Reviews'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)

train['Reviews']=train['Reviews'].fillna(round(train['Reviews'].mean()))







q1=train['Average_Cost'].quantile(0.25)

q3=train['Average_Cost'].quantile(0.75)

iqr=q3-q1

train['Average_Cost']=train['Average_Cost'].apply(lambda x: np.nan if x>(q3+1.5*iqr) or x<(q1-1.5*iqr) else x)

train['Average_Cost']=train['Average_Cost'].fillna(round(train['Average_Cost'].mean()))
train.head()
train_01=train.copy()
train['Cuisines']=le.fit_transform(train['Cuisines'])
x=train.drop('Delivery_Time',axis=1)

y=train['Delivery_Time']
x=x.apply(zscore)
start_time=time()

model_list=[lr,dt,knn,rf,ada,bag,xtree]

Score=[]

for i in model_list:

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=50)

    i.fit(x_train,y_train)

    y_pred=i.predict(x_test)

    score=accuracy_score(y_test,y_pred)

    Score.append(score)

print(pd.DataFrame(zip(model_list,Score),columns=['Model Used','R2-Score']))

end_time=time()

print(round(end_time-start_time,2),'sec')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

accuracy_score(y_test,y_pred)
train_01.head()
train_01['Cuisines']=train_01['Cuisines'].str.lower()

train_01['Cuisines']=train_01['Cuisines'].str.replace('[^a-z]',' ')
count.fit(train_01['Cuisines'])
cols=['Restaurant','Location','Average_Cost','Minimum_Order','Rating','Votes','Reviews']
data=pd.concat([pd.DataFrame(zscore(train_01.drop(['Cuisines','Delivery_Time'],axis=1)),columns=cols),pd.DataFrame(count.transform(train_01['Cuisines']).todense())],axis=1)
x=data

y=train_01['Delivery_Time']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

accuracy_score(y_test,y_pred)
x=data

y=train_01['Delivery_Time']
x.shape
pca=PCA()

pca.fit(x)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
pca=PCA(n_components=22)
x=pca.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)

rf=RandomForestClassifier(n_estimators=100,criterion='entropy',random_state=42)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)

accuracy_score(y_test,y_pred)
grid=GridSearchCV(rf,param_grid={'n_estimators':range(1,10)},return_train_score=1).fit(x_train,y_train)
pd.DataFrame(grid.cv_results_).set_index('params')['mean_test_score'].plot.line()

pd.DataFrame(grid.cv_results_).set_index('params')['mean_train_score'].plot.line()

plt.xticks(rotation=45)
test['Restaurant']=le.fit_transform(test['Restaurant'])

test['Location']=le.fit_transform(test['Location'])

test['Minimum_Order']=pd.to_numeric(test['Minimum_Order'].str.replace('₹',' '))

test['Average_Cost']=pd.to_numeric(test['Average_Cost'].str.replace('[^0-9]',''))

test['Rating']=pd.to_numeric(test['Rating'].apply(lambda x : np.nan if x in ['Temporarily Closed','Opening Soon','-','NEW'] else x))

test['Votes']=pd.to_numeric(test['Votes'].apply(lambda x : np.nan if x=='-' else x))

test['Reviews']=pd.to_numeric(test['Reviews'].apply(lambda x : np.nan if x=='-' else x))

test['Rating']=test['Rating'].fillna(test['Rating'].median())

test['Votes']=test['Votes'].fillna(test['Votes'].mode()[0])

test['Reviews']=test['Reviews'].fillna(test['Reviews'].median())

test['Average_Cost']=test['Average_Cost'].fillna(test['Average_Cost'].mean())

sample['Delivery_Time']=le.fit_transform(pd.to_numeric(sample['Delivery_Time'].str.replace('[^0-9]','')))
test['Cuisines']=test['Cuisines'].str.lower()

test['Cuisines']=test['Cuisines'].str.replace('[^a-z]',' ')
count.fit(test['Cuisines'])
pca=PCA(n_components=22)
data=pd.concat([pd.DataFrame(zscore(test.drop(['Cuisines'],axis=1)),columns=cols),pd.DataFrame((count.transform(test['Cuisines']).todense()))],axis=1)
data.head()
x=pca.fit_transform(data)
data['Delivery_Time']=pd.DataFrame(rf.predict(x))
data['Delivery_Time'].value_counts()
sample_refined=[]

for i in data['Delivery_Time']:

    i=(str(i)+' minutes')

    sample_refined.append(i)
sample_refined=pd.DataFrame(sample_refined,columns=['Delivery_Time'])

sample_refined.to_excel('Machine_Hack_Submit.xlsx',index=False)
pd.read_excel('Machine_Hack_Submit.xlsx')['Delivery_Time'].value_counts()