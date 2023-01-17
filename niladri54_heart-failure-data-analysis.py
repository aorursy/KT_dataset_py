
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
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams['figure.figsize']=12,10
df=pd.read_csv('/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.shape
df.info()
df.describe()
df['DEATH_EVENT'].value_counts()
df['anaemia'].value_counts()
df['anaemia'].plot(kind='box')
sns.distplot(df['age'],kde=False)
df['age'].plot(kind='box')
sns.set_style('whitegrid')
sns.countplot(x=df['DEATH_EVENT'],hue=df['diabetes'])
df['diabetes'].plot(kind='box')
df.groupby('diabetes')['DEATH_EVENT'].value_counts()
df['ejection_fraction'].plot(kind='hist')
df['ejection_fraction'].plot(kind='box')
df[df['ejection_fraction']>=70]
df=df[df['ejection_fraction']<70]
plt.figure(figsize=(10,8))
sns.set_style('whitegrid')
sns.countplot(x=df['DEATH_EVENT'],hue=df['high_blood_pressure'])
df['high_blood_pressure'].plot(kind='box')
df.groupby('high_blood_pressure')['DEATH_EVENT'].value_counts()
plt.figure(figsize=(10,8))
df['platelets'].plot(kind='hist',alpha=0.6,bins=20)
plt.figure(figsize=(10,8))
df['serum_creatinine'].plot(kind='hist',alpha=0.6,bins=20)
plt.figure(figsize=(10,8))
df['serum_sodium'].plot(kind='hist',alpha=0.6,bins=20)
sns.set_style('whitegrid')
sns.countplot(x=df['DEATH_EVENT'],hue=df['sex'])
df['sex'].plot(kind='box')
sns.set_style('whitegrid')
sns.countplot(x=df['DEATH_EVENT'],hue=df['smoking'])
df['smoking'].plot(kind='box')
sns.distplot(df['time'],kde=False)
df['time'].plot(kind='box')
plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),cmap='coolwarm',annot=True)
X=df.drop('DEATH_EVENT',axis=1)
y=df['DEATH_EVENT']
X.head()
from sklearn.preprocessing import StandardScaler
X_norm=StandardScaler().fit_transform(X)
X_norm[0:5]
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,log_loss
x_train,x_test,y_train,y_test=train_test_split(X_norm,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)
from sklearn.neighbors import KNeighborsClassifier
neighbors=range(1,10)
accuracy=[]
for i in neighbors:
    kn=KNeighborsClassifier(n_neighbors=i)
    kn.fit(x_train,y_train)
    yhat=kn.predict(x_test)
    accuracy.append(accuracy_score(y_test,yhat))
print(accuracy)
print('The maximum accuracy is {} and the neighbor number is {}'.format(max(accuracy),accuracy.index(max(accuracy))+1) )
plt.plot(neighbors,accuracy)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy score')
plt.text(7.2,0.783333333,'Maximum accuracy')
plt.show()
kn=KNeighborsClassifier(n_neighbors=7)
kn.fit(x_train,y_train)
yhat=kn.predict(x_test)
print('Accuracy for K Neighbors Algorithm is {}'.format(accuracy_score(y_test,yhat)))
plt.figure(figsize=(5,4))
matrix=confusion_matrix(y_test,yhat)
sns.heatmap(matrix,annot=True)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
from sklearn.tree import DecisionTreeClassifier
depth=range(2,10)
accuracy=[]
for i in depth:
    print(i)
    dst=DecisionTreeClassifier(criterion='entropy',max_depth=i)
    dst.fit(x_train,y_train)
    yhat=dst.predict(x_test)
    accuracy.append(accuracy_score(y_test,yhat))
accuracy
print('The maximum accuracy is {} and the neighbor number is {}'.format(max(accuracy),accuracy.index(max(accuracy))+2) )
plt.plot(depth,accuracy)
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.show()
dst=DecisionTreeClassifier(criterion='entropy',max_depth=3)
dst.fit(x_train,y_train)
yhat=dst.predict(x_test)
print('Accuracy for Decision Tree Algorithm is {}'.format(accuracy_score(y_test,yhat)))
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,yhat),annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
plt.show()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(C=0.01,solver='sag')
lr.fit(x_train,y_train)
yhat=lr.predict(x_test)
print('Accuracy for Logistic Regression Model {}'.format(accuracy_score(y_test,yhat)))
yhat_prob=lr.predict_proba(x_test)
print('Log loss for Logistic Regression {}'.format(log_loss(y_test,yhat)))
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,yhat),annot=True)
plt.xlabel('Predicted Value')
plt.ylabel('Actual Value')
from sklearn.svm import SVC
sv=SVC(kernel='rbf')
sv.fit(x_train,y_train)
yhat=sv.predict(x_test)
print('Accuracy for Support Vector Classifier {}'.format(accuracy_score(y_test,yhat)))
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test,yhat),annot=True)
plt.xlabel('Predict Value')
plt.ylabel('Actual Value')
