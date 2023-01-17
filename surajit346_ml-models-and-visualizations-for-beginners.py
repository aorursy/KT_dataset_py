# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#reading the file

df=pd.read_csv('../input/mushrooms.csv')

df.head()

df.info() #this method will show if there are any null character in our data 
#to check how many unique values are there in a column 

print('unique_elements: ',df['class'].unique())

#to check how many are there of each kind in a column

df['class'].value_counts()
df.describe()
sns.countplot(x='gill-color',hue='class',data=df)
sns.countplot(x='cap-surface',hue='class',data=df)
sns.countplot(x='odor',hue='class',data=df)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

for i in df.columns:

    df[i]=le.fit_transform(df[i])

df.head()

X=df.iloc[:,1:23]

y=df['class']

X.head()

y.head()
sns.set_style('whitegrid')

sns.boxplot( x=df['class'],y=df['cap-color'])
sns.boxplot( x='class',y='stalk-color-above-ring',data=df,palette="Set3")
sns.boxplot( x='class',y='stalk-color-below-ring',data=df,palette="Set2")

sns.stripplot(x='class',y='stalk-color-below-ring',data=df,jitter=True,color=".6")
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=4)

X_train.head()

y_train.head()
from sklearn.linear_model import LogisticRegression

logreg=LogisticRegression()

logreg.fit(X_train,y_train)

pred1=logreg.predict(X_test)

print('lr.score:',round(logreg.score(X_train,y_train)*100,3))

print('metrics.accuracy_score:',round(metrics.accuracy_score(y_test,pred1)*100,3))
from sklearn.model_selection import cross_val_score

cv_result=cross_val_score(logreg,X,y,cv=10)

print("cv_score:",cv_result)

print('average_score: ',np.sum(cv_result)/10)



from sklearn.metrics import classification_report,confusion_matrix

cm=confusion_matrix(y_test,pred1)

print('confusion_matrix: \n',cm)

print('classification_report:\n',classification_report(y_test,pred1))





#Heatmap visualization

sns.heatmap(cm,annot=True,fmt='d')

plt.show()
#plotting ROC

from sklearn.metrics import roc_curve,auc

y_pred_logreg=logreg.predict_proba(X_test)[:,1]

fpr,tpr,threshold=roc_curve(y_test,y_pred_logreg)

roc_auc=auc(fpr,tpr)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr,label='ROC curve (area = %0.2f)' % roc_auc)

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.title('ROC')

plt.show()

print(roc_auc)
from sklearn.tree import DecisionTreeClassifier

dr=DecisionTreeClassifier()

dr.fit(X_train,y_train)

pred=dr.predict(X_test)

#pred = np.where(prob > 0.5, 1, 0)[:,1]

dr.score(X_test,y_test)

from sklearn.metrics import classification_report,confusion_matrix

cm=confusion_matrix(y_test,pred)

print('confusion_matrix: \n',cm)

print('classification_report:\n',classification_report(y_test,pred))





#Heatmap visualization

sns.heatmap(cm,annot=True,fmt='d')

plt.show()

#plotting ROC



from sklearn.metrics import roc_curve,auc

y_pred=dr.predict_proba(X_test)[:,1]

print(y_pred)

fpr,tpr,threshold=roc_curve(y_test,y_pred)

plt.plot([0,1],[0,1],'k--')

plt.plot(fpr,tpr)

plt.xlabel('false positive rate')

plt.ylabel('true positive rate')

plt.title('ROC')

plt.show()
import seaborn as sns

corr=df.corr()

f,ax=plt.subplots(figsize=(12,9))

sns.heatmap(corr,xticklabels=corr.columns.values,

            yticklabels=corr.columns.values)

plt.show()
cols = corr.nlargest(15, 'class')['class'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

f,ax=plt.subplots(figsize=(10,9))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
from sklearn import svm

#as it is clear from the matrix that the feature veil-type hast no importace so lets drop the feature

x=X.drop(['veil-type'],axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=4)

clf=svm.SVC()

clf.fit(x_train,y_train)

clf.predict(x_test)

clf.score(x_test,y_test)