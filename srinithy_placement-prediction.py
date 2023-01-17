import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import matplotlib.pyplot as plt
import seaborn as sns
import sys

df = pd.read_csv("../input/graduate-admissions/Admission_Predict.csv",sep = ",")
df.columns
df.info()
df.describe()
df.head()
df.tail()
df.drop(['University Rating'],axis=1,inplace=True)
df.head()
df.columns=['NO','Coding','Analytic','SSLC','HSC','CGPA','Placed','Probability']
df.head()
fig,ax = plt.subplots(figsize=(6,5))
sns.heatmap(df.corr(), ax=ax, annot=True, linewidths=0.05, fmt= '.2f')
plt.show()
fig,ax=plt.subplots(figsize=(30,10))
sns.countplot(x='CGPA',data=df)
sns.countplot(x='Placed',data=df)

y = np.array([df["Coding"].min(),df["Coding"].mean(),df["Coding"].max()])
x = ["Worst","Average","Best"]
plt.bar(x,y)
plt.title("Coding Scores")
plt.xlabel("Level")
plt.ylabel("Coding Score")
plt.show()
plt.scatter(df["Placed"],df.CGPA,color='green')
plt.title("CGPA vs placed")
plt.xlabel("placed")
plt.ylabel("CGPA")
plt.show()
plt.scatter(df["Coding"],df.CGPA)
plt.title("CGPA vs coding scores")
plt.xlabel("coding Score")
plt.ylabel("CGPA")
plt.show()
plt.scatter(df["Analytic"],df.CGPA)
plt.title("CGPA vs analytic Scores")
plt.xlabel("analytic Score")
plt.ylabel("CGPA")
plt.show()
df[df.CGPA >= 8.5].plot(kind='scatter', x='Analytic', y='Coding',color="red")
plt.xlabel("Analytic ")
plt.ylabel("Coding")
plt.title("CGPA>=8.5")
plt.grid(True)
plt.show()
fig,ax=plt.subplots(figsize=(6,6))
z=sns.violinplot(data=df,x='SSLC',y='HSC')
z
q=sns.distplot(df["Coding"])
q
df.head()

serialNo = df["NO"].values
df.drop(["NO"],axis=1,inplace = True)

y = df["Placed"].values
x = df.drop(["Placed"],axis=1)


x.head()
y

from sklearn.model_selection import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.20,random_state = 42)


from sklearn.preprocessing import MinMaxScaler
scalerX = MinMaxScaler(feature_range=(0, 1))
x_train[x_train.columns] = scalerX.fit_transform(x_train[x_train.columns])
x_test[x_test.columns] = scalerX.transform(x_test[x_test.columns])

y_train_01 = [1 if each > 0.8 else 0 for each in y_train]
y_test_01  = [1 if each > 0.8 else 0 for each in y_test]
y_train_01 = np.array(y_train_01)
y_test_01 = np.array(y_test_01)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
y_pred
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test, y_pred))
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix

svc_model=SVC(gamma='auto')
svc_model.fit(x_train,y_train)
y_predict=svc_model.predict(x_test)
cm=confusion_matrix(y_test,y_predict)
sns.heatmap(cm,annot=True)
print(classification_report(y_test, y_predict))
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0,solver='lbfgs')
classifier.fit(x_train,y_train)
y_pred_test=classifier.predict(x_test)

cm=confusion_matrix(y_test,y_pred_test)
sns.heatmap(cm,annot=True,fmt='d')
print(classification_report(y_test, y_pred_test))
a=df[df.Placed ==1]
b=df[df.Placed ==0]
placed=len(a)
nplaced=len(b)
print("PERCENTAGE OF STUDENTS PLACED:",(placed/400)*100)
print("PERCENTAGE OF STUDENT NOT PLACED:",100-(placed/400)*100)
a.head()
a.describe()
p=a[a.Coding>329]
p=p[p.Analytic>114]
p=p[p.SSLC>3.5]
p=p[p.HSC>4]
p=p[p.CGPA>8]
print("THE NUMBER OF STUDENTS PLACED IN PRODUCT COMPANY IS",len(p))
print("PERCENTAGE OF STUDENTS PLACED IN PRODUCT COMPANY IS",(len(p)/placed)*100)
print("THE NUMBER OF STUDENTS PLACED IN SERVICE COMPANY IS:",400-(len(a)-len(p)))
print("PERCENTAGE OF STUDENTS PLACED IN SERVICE COMPANY IS",100-(len(p)/placed)*100)






