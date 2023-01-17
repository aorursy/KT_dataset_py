import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv(r"../input/pima-indians-diabetes-database/diabetes.csv")
df.head()
df.isnull().sum()
sns.scatterplot(x=df['Outcome'],y=df['Insulin'])
sns.barplot(x=df['Outcome'],y=df['Insulin'])
#the above plot indicates that higher the insulin level the higher the probability of diabetes
#we will now check for other features such as age etc
sns.barplot(x=df['Outcome'],y=df['Age'])
sns.barplot(x=df['Outcome'],y=df['Glucose'])
sns.barplot(x=df['Outcome'],y=df['DiabetesPedigreeFunction'])
#the above plot indicates that for DiabetesPedigreeFunction of >0.5 more people tend to get diabetes
sns.barplot(x=df['Outcome'],y=df['BMI'])
sns.barplot(x=df['Outcome'],y=df['Glucose'])
sns.barplot(x=df['Outcome'],y=df['BloodPressure'])
X=df.drop('Outcome',axis=1)
y=df['Outcome']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.svm import LinearSVC
svc=LinearSVC()

svc.fit(X_train,y_train)
predictions=svc.predict(X_test)
svc.score(X_test,y_test)
from sklearn.metrics import confusion_matrix,f1_score
confusion_matrix(predictions,y_test)

f1_score(predictions,y_test)
from sklearn.model_selection import cross_val_score
cross_val_score(svc,X_train,y_train,cv=5,scoring="accuracy")
from sklearn.ensemble import RandomForestClassifier
rdf=RandomForestClassifier()
rdf.fit(X_train,y_train)
predictions_1=rdf.predict(X_test)
rdf.score(X_test,y_test)
confusion_matrix(predictions_1,y_test)
f1_score(predictions_1,y_test)
from sklearn.ensemble import AdaBoostClassifier
adab=AdaBoostClassifier(learning_rate=0.01)
adab.fit(X_train,y_train)
predictions_2=adab.predict(X_test)
adab.score(X_test,y_test)
confusion_matrix(predictions_2,y_test)
f1_score(predictions_2,y_test)
cross_val_score(adab,X_train,y_train,cv=5,scoring='accuracy')
cross_val_score(rdf,X_train,y_train,cv=5,scoring='accuracy')

