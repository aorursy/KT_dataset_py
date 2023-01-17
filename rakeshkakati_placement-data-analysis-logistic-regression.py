import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

init_notebook_mode(connected=True)

cf.go_offline()
file = pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

file.head()
file.isnull().sum()
file.dtypes
file.columns
file[['ssc_p','hsc_p','degree_p']].iplot(kind='spread')
file['ssc_p'].iplot(kind='hist',bins=25)
file['hsc_p'].iplot(kind='hist',bins=25)
file['degree_p'].iplot(kind='hist',bins=25)
#Correlation

sns.heatmap(file.corr(),annot=True)
#Distribution of ssc percentage

sns.barplot(x='sl_no',y='ssc_p',data=file)

#Distribution of hsc percentage

sns.barplot(x='sl_no',y='hsc_p',data=file)
sns.barplot(x='sl_no',y='degree_p',data=file)
sns.barplot(x='hsc_s',y='degree_p',data=file)

sns.barplot(x='degree_t',y='etest_p',data=file)
sns.barplot(x='degree_t',y='salary',data=file)
sns.countplot(x='gender',data=file)
sns.countplot(x='etest_p',data=file)
sns.countplot(x='salary',data=file)
sns.pairplot(file)
file1 = file.drop(['salary'],axis=1)

file1.head()
#Converting categorical to numerical

file2 = pd.get_dummies(file1,)

file2.drop(['status_Not Placed'],axis=1)
#Scaling the Values

from sklearn.preprocessing import StandardScaler

import numpy as np

ss = StandardScaler()

pd.DataFrame(ss.fit_transform(np.asarray(file2)),columns = file2.columns)
X = file2.drop(['status_Placed'],axis=1)

y = file2['status_Placed']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(solver = 'liblinear')

lr.fit(X_train,y_train)
pred = lr.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred))
confusion_matrix(y_test,pred)
#Checking the impact of percentages on placement
X = file2[['sl_no','ssc_p','hsc_p','degree_p','etest_p','mba_p']]

y = file2['status_Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
lr = LogisticRegression(solver = 'liblinear')

lr.fit(X_train,y_train)
pred = lr.predict(X_test)
print(classification_report(y_test,pred))
confusion_matrix(y_test,pred)