import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv")
pd.set_option('display.max_rows',5000, 'display.max_columns',100)
df.head()
df.isnull().sum()
df.info()
#Checking Maximum,Minimum,Average of GRE Score and TOEFL Score.

print(df['GRE Score'].max(),df['GRE Score'].min(),df['GRE Score'].mean())

print(df['TOEFL Score'].max(),df['TOEFL Score'].min(),df['TOEFL Score'].mean())
#People With Good Research write Good SOP 

sns.countplot(x='Research',hue='SOP',data=df)
grades = []

for i in df['CGPA']:

    if i >= 9.00 and i<10.00:

        grades.append('A')

    elif i >=8.00 and i<9.00:

        grades.append('B+')

    elif i >=7.00 and i<8.00:

        grades.append('B')

    elif i >=6.00 and i<7.00:

        grades.append('B-')

    elif i >=4.00 and i<6.00:

        grades.append('C')        

    else:

        grades.append('F')

df['grades'] = grades
df.head()
#Analysis of Reasearch wrt Grades.

sns.countplot(x='Research',hue='grades',data=df)
#Analysis of University Rating wrt Grades,So here we can see 

sns.countplot(x='University Rating',hue='grades',data=df)
#Analysis of GRE Score wrt Grades to check people in range in GRE SCORE wrt to grades

sns.boxplot(x='GRE Score',y='grades',data=df)
#Analysis of TOEFL Score wrt Grades to check people in range in TOEFL SCORE wrt to grades

sns.boxplot(x='TOEFL Score',y='grades',data=df)
#People having good grades have a good SOP

sns.boxplot(x='SOP',y='grades',data=df)
#People having good grades have a good LOR

sns.boxplot(x='LOR',y='grades',data=df)
Admission = []

for i in df['CGPA']:

    if i>=9.00:

        Admission.append('Very High')

    elif i>=8.00 and i<9.00:

        Admission.append('High')

    elif i>=7.00 and i<8.00:

        Admission.append('Medium')         

    else:

        Admission.append('Low')

df['Admission'] = Admission
df.head()
sns.set()

sns.relplot(x="University Rating", y="SOP", col="Admission",

            hue="grades", style="grades",

            data=df);
sns.set()



sns.relplot(x="University Rating", y="LOR", col="Admission",

            hue="grades", style="grades",

            data=df);
#Dropping Columns

df = df.drop(['Serial No.','grades'],axis=1)
from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

df['Admission'] = LE.fit_transform(df['Admission'])
df.head(30)
y=df['Admission']

X=df.drop('Admission',axis=1)
#Train and Test

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=101)
from sklearn.preprocessing import StandardScaler

Scaler_X = StandardScaler()

X_train = Scaler_X.fit_transform(X_train)

X_test = Scaler_X.transform(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
#Random Forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

rfpred = rf.predict(X_test)
print(accuracy_score(y_test,rfpred))

print(confusion_matrix(y_test,rfpred))