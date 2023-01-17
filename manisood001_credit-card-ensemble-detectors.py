# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df=pd.read_csv("../input/creditcard.csv")
df.head()
df.columns
df.info()
total_predictor_entries=df.shape[0]
non_fraud_class=sum(df['Class']==0)
fraud_class=sum(df['Class']==1)

print('the percentage of data of non fraud entries are - '+str(  ( non_fraud_class/total_predictor_entries)*100 )+'%' )
print('the percentage of data of fraud entries are - '+str(  ( fraud_class/total_predictor_entries)*100 )+'%' )

sns.countplot(x='Class',data=df)
data=df[['Time','Amount','Class']]
sns.kdeplot(data['Amount'])

plt.figure(figsize=(12,7))

sns.stripplot(x='Class',y='Amount',data=df,jitter=True,palette='viridis')
sns.violinplot(x='Class',y='Time',data=df)


df['hour']=(df['Time']//3600.).astype('int')
df['second']=(df['Time']%3600).astype('int')
df.drop('Time',axis=1,inplace=True)
plt.figure(figsize=(15,7))
sns.heatmap(df.corr(),cmap='coolwarm')
x=df.drop('Class',axis=1).values
y=df['Class'].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)

from sklearn.model_selection import  train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=0)
from imblearn.over_sampling import SMOTE
sm=SMOTE()
xtrain,ytrain=sm.fit_resample(xtrain,ytrain)
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
regressor=GaussianNB()
regressor2=LogisticRegression(solver='newton-cg')
regressor3=RandomForestClassifier(n_estimators=20)
regressor.fit(xtrain,ytrain)
regressor2.fit(xtrain,ytrain)
regressor3.fit(xtrain,ytrain)
print("*************")
from sklearn.metrics import classification_report
print('Gaussian Naive bayes Classifier accuracy')
print(classification_report(ytest,regressor.predict(xtest)))
print('--------------------')
print('Logistic Classifier accuracy ')
print(classification_report(ytest,regressor2.predict(xtest)))
print('--------------------')
print('Random Forest Classifier accuracy')
print(classification_report(ytest,regressor3.predict(xtest)))
print('Naive bayes')
print(pd.crosstab(ytest,regressor.predict(xtest)))
print('_____________________')
print('Logistic regression')
print(pd.crosstab(ytest,regressor2.predict(xtest)))
print('_____________________')
print("Random Forest")
print(pd.crosstab(ytest,regressor3.predict(xtest)))
ess=pd.DataFrame([regressor.predict(xtest),regressor2.predict(xtest),regressor3.predict(xtest)])
out=ess.apply(lambda x:x.mode())
es_test=out.transpose()[0].values
print(classification_report(ytest,es_test))
pd.crosstab(ytest,es_test)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
actual=ytest
predictions1=regressor2.predict_proba(xtest)[:,1] #logistic
predictions2=regressor.predict_proba(xtest)[:,1]#nb
predictions3=regressor3.predict_proba(xtest)[:,1]#rf
predictions4=es_test




fpr1,tpr1,t1=roc_curve(actual,predictions1)
fpr2,tpr2,t2=roc_curve(actual,predictions2)
fpr3,tpr3,t3=roc_curve(actual,predictions3)
fpr4,tpr4,t4=roc_curve(actual,predictions4)


auc1 = roc_auc_score(ytest,predictions1)
auc2 = roc_auc_score(ytest,predictions2)
auc3 = roc_auc_score(ytest,predictions3)
auc4 = roc_auc_score(ytest,predictions4)


fig=plt.figure(figsize=(15,10))
fig.legend()
plt.plot(fpr1,tpr1,label='logistic regression- auc-area'+str(round(auc1*100,3))+'%',color='red',lw=6)
plt.plot(fpr2,tpr2,label='naive bayes - auc-area'+str(round(auc2*100,3))+'%',color='blue',lw=6,ls=':')
plt.plot(fpr3,tpr3,label='random forest- auc-area'+str(round(auc3*100,3))+'%',color='green',lw=6,)
plt.plot(fpr4,tpr4,label='ensembled- auc-area'+str(round(auc4*100,3))+'%',color='brown',lw=6,ls='-')
plt.plot(np.arange(0,1,0.01),np.arange(0,1,0.01),label='random pick')
plt.legend()


from sklearn.model_selection import cross_val_score
cv=cross_val_score(regressor2,xtrain,ytrain,cv=10,scoring='accuracy')
cv.mean()
cv.std()
print('test set accuracy')
np.mean(regressor2.predict(xtest)==ytest)
