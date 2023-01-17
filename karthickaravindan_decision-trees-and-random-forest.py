import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df=pd.read_csv("../input/loan_data.csv")
df.info()
df.describe()
df.head()
plt.figure(figsize=(10,6))
df[df['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
df[df['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.figure(figsize=(10,6))
df[df['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
df[df['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
sns.countplot(data=df,x="purpose",hue="not.fully.paid",palette='Set1')
sns.jointplot(x='fico',y='int.rate',data=df,color='purple')
sns.lmplot(y='int.rate',x='fico',data=df,hue='credit.policy',
           col='not.fully.paid',palette='Set1')
df.info()
cat_feats=['purpose']
final_data =pd.get_dummies(df,columns=cat_feats,drop_first=True)
final_data.info()
from sklearn.model_selection import train_test_split 
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.tree import DecisionTreeClassifier
dtree =DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predict = dtree.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report
print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))
from sklearn.ensemble import RandomForestClassifier
rfc= RandomForestClassifier(n_estimators=600)
rfc.fit(X_train,y_train)
predict_rfc= rfc.predict(X_test)
print(classification_report(y_test,predict_rfc))
print(confusion_matrix(y_test,predict_rfc))