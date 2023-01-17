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
import numpy as np
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
train = pd.read_csv("../input/StudentsPerformance.csv")
print ("Number of entries in the given dataset are",train.shape[0])
print ("Number of columns in the given dataset are",train.shape[1])
train.head()
print ("Columns present in the given dataset are:-")
for i in train.columns:
    print(i,)

print ("Some basic evaluations :-")
print (train.describe())
train.info()
print()
print ("There are no null values in the dataset")
sns.countplot(train["gender"])
print (train['gender'].value_counts())
sns.countplot(train['race/ethnicity'],order=['group A','group B','group C','group D','group E'])
print (train['race/ethnicity'].value_counts())
sns.countplot(train['parental level of education'])
plt.xticks(rotation=30)
print (train['parental level of education'].value_counts())
sns.countplot(train['test preparation course'],palette='spring')
print (train['test preparation course'].value_counts())
# Let passing marks be 33 marks
train['Math_Pass_Status']=np.where(train['math score']>=33,'P','F')
print(train['Math_Pass_Status'].value_counts())
sns.countplot(train['Math_Pass_Status'])
# Let passing marks be 33 marks
train['Reading_Pass_Status']=np.where(train['reading score']>=33,'P','F')
print(train['Reading_Pass_Status'].value_counts())
sns.countplot(train['Reading_Pass_Status'])
# Let passing marks be 33 marks
train['Writing_Pass_Status']=np.where(train['writing score']>=33,'P','F')
print(train['Writing_Pass_Status'].value_counts())
sns.countplot(train['Writing_Pass_Status'])
f,axes = plt.subplots(figsize=(20,10),sharex=True)
sns.scatterplot(x='math score',y='reading score',data=train,hue=train['gender'])
f,axes = plt.subplots(figsize=(10,10),sharex=True)
sns.scatterplot(x='writing score',y='reading score',data=train,hue=train['gender'])
f,axes = plt.subplots(figsize=(10,10),sharex=True)
sns.scatterplot(x='writing score',y='math score',data=train,hue=train['gender'])
train.corr()
sns.heatmap(train.corr())
train.plot(subplots=True)
plt.show()
train['lunch'].value_counts()
fig,ax = plt.subplots(figsize=(5,5))
ax.pie(train["Math_Pass_Status"].value_counts(),labels=['P','F'],explode=(0.05,0.05),autopct='%1.1f%%',shadow=True,startangle=90)
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
fig,ax = plt.subplots(figsize=(5,5))
ax.pie(train["Reading_Pass_Status"].value_counts(),labels=['P','F'],explode=(0.05,0.05),autopct='%1.1f%%',shadow=True,startangle=90)
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
fig,ax = plt.subplots(figsize=(5,5))
ax.pie(train["Writing_Pass_Status"].value_counts(),labels=['P','F'],explode=(0.05,0.05),autopct='%1.1f%%',shadow=True,startangle=90)
ax.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()

sns.countplot(train['parental level of education'],hue=train['Math_Pass_Status'])
plt.xticks(rotation=90)
print("This shows that more children are failed in maths")
print ("whose parents are having high school level of education.")
sns.countplot(train['parental level of education'],hue=train['Reading_Pass_Status'])
plt.xticks(rotation=90)
print("This shows that more children are failed in reading")
print ("whose parents are having high school level of education.")
sns.countplot(train['parental level of education'],hue=train['Writing_Pass_Status'])
plt.xticks(rotation=90)
print("This shows that more children are failed in writing")
print ("whose parents are having high school level of education.")
def compute_percentage(x):
    pct = (x['math score']+x['reading score']+x['writing score'])/3.0
    return round(pct,2)
train['Percentage'] =(compute_percentage(train))
def compute_grade(train):
    if train['Percentage']<33.0:
        return 'F'
    elif train['Percentage']>=90.0:
        return 'A1'
    elif 80.0<=train['Percentage']<90.0:
        return 'A2'
    elif 70.0<=train['Percentage']<80.0:
        return 'B1'
    elif 60.0<=train['Percentage']<70.0:
        return 'B2'
    elif 50.0<=train['Percentage']<60.0:
        return 'C'
    elif 40.0<=train['Percentage']<50.0:
        return 'D'
    elif 33.0<=train['Percentage']<40.0:
        return 'E'
train['Grade'] = train.apply(compute_grade,axis=1)

sns.countplot(train['Grade'],order=['A1','A2','B1','B2','C','D','E','F'])
print(train['Grade'].value_counts())
sns.countplot(train['Grade'],hue=train['test preparation course'],order=['A1','A2','B1','B2','C','D','E','F'])
one_hot = pd.get_dummies(train['gender'],drop_first=True)
train = train.join(one_hot)
one_hot =  pd.get_dummies(train['race/ethnicity'],drop_first=True)
train = train.join(one_hot)
one_hot =  pd.get_dummies(train['parental level of education'],drop_first=True)
train = train.join(one_hot)
one_hot =  pd.get_dummies(train['test preparation course'],drop_first=True)
train = train.join(one_hot)
one_hot =  pd.get_dummies(train['Grade'],drop_first=True)
train = train.join(one_hot)

train = train.drop(['gender','race/ethnicity','parental level of education','test preparation course','Grade'],axis=1)
#train = train.drop(['Math_Pass_Status','Reading_Pass_Status','Writing_Pass_Status'],axis=1)
train = train.drop(['lunch'],axis=1)
X = train.drop(columns=['Math_Pass_Status','Reading_Pass_Status','Writing_Pass_Status'])
y = train['Math_Pass_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = DecisionTreeClassifier()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Model Accuracy (%):", accuracy_score(y_test,y_pred)*100)
X = train.drop(columns=['Math_Pass_Status','Reading_Pass_Status','Writing_Pass_Status'])
y = train['Reading_Pass_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = SVC()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
print("Model Accuracy (%):", accuracy_score(y_test,y_pred)*100)
X = train.drop(columns=['Math_Pass_Status','Reading_Pass_Status','Writing_Pass_Status'])
y = train['Writing_Pass_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print("Model Accuracy (%):", accuracy_score(y_test,y_pred)*100)
