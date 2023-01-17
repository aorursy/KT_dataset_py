import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
pd.set_option('display.max_columns', 999)#Showing All the columns
df.columns
df.head(30)
#Checking the unique value 

print(df['test preparation course'].nunique(),df['test preparation course'].unique())

print(df['parental level of education'].nunique(),df['parental level of education'].unique())

print(df['race/ethnicity'].nunique(),df['race/ethnicity'].unique())

print(df['lunch'].nunique(),df['lunch'].unique())
df[df['gender']=="male"]
df["total marks"] = df['math score'] + df['reading score'] + df['writing score']

df["total marks"]
print(df["math score"].max())

print(df["writing score"].max())

print(df["reading score"].max())
print(df["math score"].min())

print(df["writing score"].min())

print(df["reading score"].min())
df['percentage'] = (df['total marks']/300)*100

df['percentage']
grades = []

for i in df['percentage']:

    if i >= 85 and i <=100:

        grades.append('A')

    elif i >= 60 and i<85:

        grades.append('B')

    elif i >=50 and i<60:

        grades.append('C')

    elif i >=40 and i<50:

        grades.append('D')

    else:

        grades.append('F')

df['grades'] = grades

    
remarks = []

for i in df['percentage']:

    if i>40:

        remarks.append('Pass')

    else:

        remarks.append('Fail')

df['remarks'] = remarks
df.head()
sns.countplot(x = "gender",data=df,hue="grades")
sns.countplot(x = "gender",data=df,hue="parental level of education")
sns.countplot(y ="parental level of education",data=df,hue="test preparation course")
sns.countplot(y ="parental level of education",data=df,hue="remarks")
sns.countplot(x ="race/ethnicity",data=df,hue="grades")
for i in range(len(df)):

    if df.iloc[i,2] in ["bachelor's degree", "master's degree","associate's degree"]:

        df.iloc[i,2] = 'Degree'

    else:

        df.iloc[i,2] = 'No Degree'
sns.countplot(x ="race/ethnicity",data=df,hue="parental level of education")
df.head()
from sklearn.preprocessing import LabelEncoder

labelEncoder_X = LabelEncoder()

df['parental level of education'] = labelEncoder_X.fit_transform(df['parental level of education'])

df['gender'] = labelEncoder_X.fit_transform(df['gender'])

df['lunch'] = labelEncoder_X.fit_transform(df['lunch'])

df['race/ethnicity'] = labelEncoder_X.fit_transform(df['race/ethnicity'])

df['test preparation course'] = labelEncoder_X.fit_transform(df['test preparation course'])

df['grades'] = labelEncoder_X.fit_transform(df['grades'])

df['remarks'] = labelEncoder_X.fit_transform(df['remarks'])
df.head()
X=df.drop('remarks',axis=1)

y=df['remarks']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
#Logistic Regression

from sklearn.linear_model import LogisticRegression



lr = LogisticRegression()

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)



print(accuracy_score(y_test,y_pred))

print(confusion_matrix(y_test,y_pred))
#Decision Tree

from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)

predictions = dtree.predict(X_test)
print(accuracy_score(y_test,predictions ))

print(confusion_matrix(y_test,predictions ))
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=200)

rfc.fit(X_train,y_train)

rfc__pred = rfc.predict(X_test)
print(accuracy_score(y_test,rfc__pred))

print(confusion_matrix(y_test,rfc__pred))
rfc__pred[:50]
y_test[:50]