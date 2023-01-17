import pandas as pd
data=pd.read_csv("../input/HR_comma_sep.csv")
print(data.head())
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data.info()
sns.countplot(x=data['left']) 
data.isnull().sum().sum()
plt.figure(figsize=(10,5))

data[data['left']==1]['satisfaction_level'].hist(alpha=0.5,color='blue',bins=20,label='left')

data[data['left']==0]['satisfaction_level'].hist(alpha=0.5,color='red',bins=20,label='not left')

plt.xlabel('satisfaction level')

plt.legend()
sns.countplot(x=data['left'],hue=data['salary'],palette='cool')
sns.regplot(x='satisfaction_level',y='last_evaluation',data=data,fit_reg=False)
c=data.corr()

sns.heatmap(c)
ax=sns.boxplot(x='left',y='time_spend_company',data=data)
sns.stripplot(y='average_montly_hours',x='left',data=data,jitter=True)
plt.figure(figsize=(12,6))

ax=sns.violinplot(x='sales',y='average_montly_hours',data=data,hue='left',split=True,inner='quartile')
df=pd.get_dummies(data,columns=['sales','salary']) #one hot encoding of the categorical variables

df.head()
from sklearn.model_selection import train_test_split
trainx,testx,trainy,testy=train_test_split(df.drop('left',1),df['left'],test_size=0.3)
from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
lr=LogisticRegression()

dt=DecisionTreeClassifier()

rf=RandomForestClassifier()
lr.fit(trainx,trainy)

dt.fit(trainx,trainy)

rf.fit(trainx,trainy)
lr_pred=lr.predict(testx)

dt_pred=dt.predict(testx)

rf_pred=rf.predict(testx)
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
lr_pred1=lr.predict_proba(testx)[:,1]

fpr_lr,tpr_lr,_=roc_curve(testy,lr_pred1)

dt_pred1=dt.predict_proba(testx)[:,1]

fpr_dt,tpr_dt,_=roc_curve(testy,dt_pred1)

rf_pred1=rf.predict_proba(testx)[:,1]

fpr_rf,tpr_rf,_=roc_curve(testy,rf_pred1)
print("Logistic Regression")

print(classification_report(testy,lr_pred))
print("Decision Tree Classifier")

print(classification_report(testy,dt_pred))
print("Random Forest Classifier")

print(classification_report(testy,rf_pred))
plt.figure(1)

plt.plot(fpr_lr,tpr_lr,label='LR')

plt.plot(fpr_dt,tpr_dt,label='DT')

plt.plot(fpr_rf,tpr_rf,label='RF')

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()