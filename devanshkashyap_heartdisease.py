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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
df.head()
df.info()
df.describe()
df.shape
df.isna().sum()
df['target'].value_counts()
df['target'].plot.hist()
df['sex'].plot.hist()
sns.countplot(x="target",hue="sex",data=df)
sns.barplot(x="target",y="age",data=df)
sns.countplot(x="target",hue="cp",data=df)
sns.heatmap(df.corr(),annot=True,fmt='.1g')
sns.factorplot('age',kind='count',hue='target',data=df,height=10,aspect=0.9)

df.head()
df['sex']=df['sex'].map({1:'Male',0:'Female'})
df['cp']=df['cp'].map({0:'asymptomatic',1:'cp_atypical_angina',2:'cp_non_anginal_pain',3:'cp_typicalangina'}) 
df['restecg']=df['restecg'].map({0:'restecg_ventricular_hypertrophy',1:'restecg_normal',2:'restecg_ST_t_wave abnormality'})
df['slope']=df['slope'].map({0:'slope_downsloping',1:'slope_flat',2:'slope_upsloping'})
df['thal']=df['thal'].map({1:'thal_fixed_defect',2:'thal_normal',3:'thal_reversable_defect'})
df.head()
df.isna().sum()
dummy=pd.get_dummies(df[['cp', 'restecg','sex', 'slope', 'thal']])
dummy.head()
df=df.join(dummy,how='left')

df.head()
df=df.drop(columns=['cp', 'restecg','sex','slope', 'thal'],axis=1)
df.head()
df=df.drop_duplicates()
df.corr()
features=df.columns
x=df[features]
x=df.drop('target',axis=1)
y=df['target']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
knc=KNeighborsClassifier()
rfc=RandomForestClassifier(n_estimators=200)
lr=LogisticRegression()
dtc=DecisionTreeClassifier()
print("KNeighborsClassifier")
knc.fit(x_train,y_train)
y_pred_knc=knc.predict(x_test)
print(accuracy_score(y_test,y_pred_knc)*100)
print("RandomForestClassifier")
rfc.fit(x_train,y_train)
y_pred_rfc=rfc.predict(x_test)
print(accuracy_score(y_test,y_pred_rfc)*100)
print("DecisionTreeClassifier")
dtc.fit(x_train,y_train)
y_pred_dtc=dtc.predict(x_test)
print(accuracy_score(y_test,y_pred_dtc)*100)
print("LogisticRegression")
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)
print(accuracy_score(y_test,y_pred_lr)*100)
print("Final Algorithm considered is RandomForestClassifier with the accuracy of 84.615%")
