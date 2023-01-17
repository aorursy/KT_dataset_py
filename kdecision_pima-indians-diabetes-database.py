# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
plt.figure(figsize=(12,8))
sns.scatterplot(x='BMI',y='Glucose',data=df,palette='pastel')
plt.figure(figsize=(12,8))
sns.scatterplot(x='BMI',y='Age',data=df,palette='pastel')
plt.figure(figsize=(12,8))
sns.scatterplot(x='BMI',y='Glucose',data=df,palette='pastel')
plt.figure(figsize=(12,8))
sns.scatterplot(x='Insulin',y='Glucose',data=df,palette='pastel')
plt.figure(figsize=(12,8))
sns.scatterplot(x='BloodPressure',y='Glucose',data=df,palette='pastel')
from scipy.stats import norm
plt.figure(figsize=(12,7))
sns.distplot(df['Glucose'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['BloodPressure'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['Insulin'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['Age'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['BMI'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['DiabetesPedigreeFunction'],kde=True,fit=norm)
plt.figure(figsize=(12,7))
sns.distplot(df['SkinThickness'],kde=True,fit=norm)
df['Outcome'].value_counts()
plt.figure(figsize=(10,6))
labels=[0,1]
sns.barplot(x=labels,y=df['Outcome'].value_counts(),palette='pastel')
corr = df.corr()
plt.figure(figsize=(12,6))
sns.heatmap(corr,vmax=.3,annot=True,square=True)
sns.pairplot(data=df,hue='Outcome',diag_kind='kde')
plt.show()
y_train = df['Outcome'].values
df = df.drop(columns=['Outcome'],axis=True)
X_train,X_Test,y_train,y_test = train_test_split(df,y_train,test_size=0.2)
log  = LogisticRegression(random_state=42,max_iter=150,verbose=1)
res = log.fit(X_train,y_train)
res_p=log.predict(X_Test)

cm = confusion_matrix(y_test,res_p)
print(cm)
print(accuracy_score(y_test,res_p))
print(classification_report(y_test,res_p))
svc = SVC(kernel='linear',random_state=42)
fitt = svc.fit(X_train,y_train)

res_s=svc.predict(X_Test)
cm = confusion_matrix(y_test,res_s)
print(cm)
print(accuracy_score(y_test,res_s))
print(classification_report(y_test,res_s))
rf = RandomForestClassifier(max_depth=3,random_state=42,verbose=1)

res = rf.fit(X_train,y_train)
res_f = rf.predict(X_Test)
cm = confusion_matrix(y_test,res_f)
print(cm)
print(accuracy_score(y_test,res_f))
print(classification_report(y_test,res_f))

xg = XGBClassifier()
p =xg.fit(X_train,y_train)

res_x = xg.predict(X_Test)
cm = confusion_matrix(y_test,res_x)
print(cm)
print(accuracy_score(y_test,res_x))
print(classification_report(y_test,res_x))
