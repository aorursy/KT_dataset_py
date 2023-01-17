import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



df =  pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')

df.head()
df.describe()
df.columns
df.info()
#df[df.trestbps > 180]

#df[df.chol > 400]



df[(df.trestbps > 180) | (df.chol > 400)]
plt.figure(figsize=(6,4))

sns.countplot(x='target', hue = 'sex', data = df, palette='rainbow')
j =0

plt.figure(figsize=(16,14))



for i in ['age','trestbps','chol','thalach','oldpeak']:

    j=j+1

    plt.subplot(3,3,j)

    plt.title('variation of %s' %i)

    sns.boxplot(y = i,x ='exang',hue='target',data = df,palette='Set1')
sns.pairplot(hue ='target', vars = ['chol','trestbps','thalach','oldpeak'] , data = df,height =2.5,aspect=1.2)
j=0

plt.figure(figsize=(16,14))



for i in ['cp', 'fbs', 'restecg', 'exang', 'slope','ca', 'thal']:

    j=j+1

    plt.subplot(3,3,j)

    plt.title('variation of %s' %i)

    sns.countplot(i, hue ='target', data = df,palette='rainbow')
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import accuracy_score, confusion_matrix



X = df.drop('target', axis =1)

y = df.target



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.2, random_state =8)
from sklearn.naive_bayes import GaussianNB



nb = GaussianNB().fit(X_train, y_train)

res_nb = nb.predict(X_test)





print ('train score  : %.3f' %nb.score(X_train,y_train))

print ('test score   : {:.3f}'.format(nb.score(X_test,y_test)))

confusion_matrix(y_test, res_nb)
logr = LogisticRegression(C = 0.2,solver='liblinear', max_iter=10000).fit(X_train, y_train)

res_logr = logr.predict(X_test)



print ('train score  : %.3f' %logr.score(X_train,y_train))

print ('test score   : {:.3f}'.format(logr.score(X_test,y_test)))
confusion_matrix(y_test, res_logr)
scores = cross_val_score(logr, X_train, y_train, cv =10, scoring='accuracy')

print(scores)

print('\n',scores.mean())
from sklearn.svm import SVC



sv = SVC(C= 2, gamma= 0.1, kernel= 'linear').fit(X_train, y_train)

res_sv = sv.predict(X_test)
print ('train score  : %.3f' %sv.score(X_train,y_train))

print ('test score   : {:.3f}'.format(sv.score(X_test,y_test)))



confusion_matrix(y_test, res_sv)
scores = cross_val_score(sv, X_train, y_train, cv =8, scoring='accuracy')

print(scores)

print(scores.mean())
from sklearn.ensemble import RandomForestClassifier



rft = RandomForestClassifier(n_estimators=10, max_depth=3, max_features=12,random_state=2).fit(X_train, y_train)

res_rft = rft.predict(X_test)
print ('train score  : %.3f' %rft.score(X_train,y_train))

print ('test score   : {:.3f}'.format(rft.score(X_test,y_test)))

confusion_matrix(y_test, res_rft)