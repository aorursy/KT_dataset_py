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
df = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df
df.info()
lc = []

ln = []

for i in df.columns:

    if df[i].dtype == 'object':

        lc.append(i)

    else:

        ln.append(i)

        

print('The Categorical Values are',lc)

print('The no.of Categorical Values are',len(lc))

print('The Numerical Values are',ln)

print('The no.of Numerical Values are',len(ln))
df.isnull().sum()
print(df['gender'].value_counts())

print(df['ssc_b'].value_counts())

print(df['hsc_b'].value_counts())

print(df['hsc_s'].value_counts())

print(df['degree_t'].value_counts())

print(df['workex'].value_counts())

print(df['specialisation'].value_counts())

print(df['status'].value_counts())
import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(df['status'],hue=df['gender'])
sns.boxplot(df['ssc_b'],df['salary'])
for i in df.columns:

    if df[i].dtype != 'object':

        sns.boxplot(df[i])

        plt.show()
q3_hsc = df['hsc_p'].quantile(0.75)

q1_hsc = df['hsc_p'].quantile(0.25)
IQR_hsc = q3_hsc - q1_hsc
ul_hsc = q3_hsc+(1.5*IQR_hsc)

ll_hsc = q1_hsc-(1.5*IQR_hsc)
df = df[(df['hsc_p']>ll_hsc)&(df['hsc_p']<ul_hsc)]
q3_deg = df['degree_p'].quantile(0.75)

q1_deg = df['degree_p'].quantile(0.25)
IQR_deg = q3_deg - q1_deg
ul_deg = q3_deg+(1.5*IQR_deg)

ll_deg = q1_deg-(1.5*IQR_deg)
df = df[(df['degree_p']>ll_deg)&(df['degree_p']<ul_deg)]
df.shape
df1 = df.drop(columns = ['salary','sl_no','gender'],axis=1)

df1.shape
df1.head()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1['ssc_b'] = le.fit_transform(df1['ssc_b'])

df1['hsc_b'] = le.fit_transform(df1['hsc_b'])

df1['hsc_s'] = le.fit_transform(df1['hsc_s'])

df1['degree_t'] = le.fit_transform(df1['degree_t'])

df1['workex'] = le.fit_transform(df1['workex'])

df1['specialisation'] = le.fit_transform(df1['specialisation'])

df1['status'] = le.fit_transform(df1['status'])
df1
X = df1.drop(['status'],axis=1)

y = df1['status']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
y_pred = lr.predict(X_test)
from sklearn.metrics import roc_auc_score, classification_report
auc = roc_auc_score(y_test,y_pred)
print('Accuracy of Logistic Regression:',auc)
print('Classification Report:\n',classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
y_pred_rfc = rfc.predict(X_test)
auc_rfc = roc_auc_score(y_test,y_pred_rfc)
print('The Accuracy for Random Forest Classifier:',auc_rfc)
print('Classification Report:\n',classification_report(y_test,y_pred_rfc))
print('The Accuracy score of Logistic Regression:',auc)

print('The Accuracy score of Random Forest Classifier:',auc_rfc)