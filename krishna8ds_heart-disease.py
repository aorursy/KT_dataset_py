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

import pandas as pd 



import seaborn as sns 

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split



from sklearn.preprocessing import StandardScaler



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC



from sklearn.ensemble import RandomForestClassifier,VotingClassifier,AdaBoostClassifier,BaggingClassifier

import xgboost as xgb



from sklearn import metrics

from prettytable import PrettyTable
df = pd.read_csv('../input/H.D_1.csv')
df.replace('A',1,inplace=True)
df.head()
df.nunique()
df.info()
df.isnull().sum()
df['sex']=df.sex.astype('int64')
df.info()
plt.figure(figsize=(14,10))

corr = df.corr()

sns.heatmap(corr,annot=True,linewidths=3,linecolor='c',cbar=False)

plt.show()
sns.pairplot(df,diag_kind='kde')
plt.figure(figsize=(15,8))

plt.style.use('seaborn')

sns.lineplot(x=df.age,y=df.cp,hue=df.sex,ci=0)

plt.show()

# 0 -- female       1 -- male
plt.figure(figsize=(15,8))

plt.style.use('seaborn')

sns.lineplot(x=df.chol,y=df.restecg)

plt.xlabel('Cholesterol')

plt.ylabel('Resting ECG')

plt.show()
plt.figure(figsize=(15,8))

plt.style.use('seaborn')

sns.lineplot(x=df.trestbps,y=df.chol,hue=df.sex,ci=0)

plt.show()
plt.figure(figsize=(20,5))

plt.style.use('seaborn')

sns.countplot(x=df.chol,hue=df.cp)

plt.legend()

plt.show()

# 1 --Male    0 --Female
plt.figure(figsize=(10,5))

plt.style.use('seaborn')

sns.lineplot(x=df.exang,y=df.chol,hue=df.sex,ci=0)

plt.show()
plt.figure(figsize=(10,5))

plt.style.use('seaborn')

sns.lineplot(x=df.slope,y=df.chol,hue=df.sex,ci=0)

plt.show()
plt.figure(figsize=(10,5))

plt.style.use('seaborn')

sns.lineplot(x=df.ca,y=df.chol,hue=df.sex,ci=0)

plt.show()
plt.figure(figsize=(6,2))

plt.style.use('seaborn')

sns.boxplot(x=df.age)

plt.show()
plt.figure(figsize=(6,2))

plt.style.use('seaborn')

sns.boxplot(x=df.trestbps)

plt.show()
plt.figure(figsize=(6,2))

plt.style.use('seaborn')

sns.boxplot(x=df.chol)

plt.show()
plt.figure(figsize=(6,2))

plt.style.use('seaborn')

sns.boxplot(x=df.thalach)

plt.show()
plt.figure(figsize=(6,2))

plt.style.use('seaborn')

sns.boxplot(x=df.oldpeak)

plt.show()
pd.crosstab(df.target.count(),df.sex)
from scipy import stats

data_out = df[np.abs((stats.zscore(df)) < 3) .all(axis=1)].copy()
df1_y=df.iloc[:,-1]

df1_x=df.iloc[:,:-1]
x1train,x1test,y1train,y1test=train_test_split(df1_x,df1_y,random_state=43)
sc1=StandardScaler()

X1train=sc1.fit_transform(x1train)

X1test=sc1.transform(x1test)
LR1 = LogisticRegression(random_state=1)

LR1.fit(X1train,y1train)

LR_o_pre = LR1.predict(X1test)





print('accuracy ',metrics.accuracy_score(y1test,LR_o_pre))

print('recall',metrics.recall_score(y1test,LR_o_pre))

print('precision',metrics.precision_score(y1test,LR_o_pre))

print('F1',metrics.f1_score(y1test,LR_o_pre))

print('roc_auc_score',metrics.roc_auc_score(y1test,LR_o_pre))

print(metrics.classification_report(y1test,LR_o_pre))



print(metrics.confusion_matrix(y1test,LR_o_pre))



fpr1, tpr1, threshold = metrics.roc_curve(y1test,LR_o_pre)

roc_auc1 = metrics.auc(fpr1, tpr1)
dt1=DecisionTreeClassifier(criterion='gini',random_state=67)

dt1.fit(X1train,y1train)

dt_o_pre = dt1.predict(X1test)



print('accuracy ',metrics.accuracy_score(y1test,dt_o_pre))

print('precision',metrics.precision_score(y1test,dt_o_pre))

print('recall',metrics.recall_score(y1test,dt_o_pre))

print('F1',metrics.f1_score(y1test,dt_o_pre))

print('roc_auc_score',metrics.roc_auc_score(y1test,dt_o_pre))

print(metrics.confusion_matrix(y1test,dt_o_pre))

print(metrics.classification_report(y1test,dt_o_pre))

fpr2, tpr2, threshold = metrics.roc_curve(y1test,dt_o_pre)

roc_auc2 = metrics.auc(fpr2, tpr2)
rf_o = RandomForestClassifier(random_state=60,n_estimators=29)

rf_o.fit(X1train,y1train)

rf_o_pre=rf_o.predict(X1test)



print('accuracy ',metrics.accuracy_score(y1test,rf_o_pre))

print('precision',metrics.precision_score(y1test,rf_o_pre))

print('recall',metrics.recall_score(y1test,rf_o_pre))

print('F1',metrics.f1_score(y1test,rf_o_pre))

print('roc_auc_score',metrics.roc_auc_score(y1test,rf_o_pre))

print(metrics.confusion_matrix(y1test,rf_o_pre))

print(metrics.classification_report(y1test,rf_o_pre))



fpr6, tpr6, threshold = metrics.roc_curve(y1test,rf_o_pre)

roc_auc6 = metrics.auc(fpr6, tpr6)
bg_o = BaggingClassifier(DecisionTreeClassifier(),n_estimators=11,random_state=5)

bg_o.fit(X1train,y1train)

bg_o_pre = bg_o.predict(X1test)



print(metrics.accuracy_score(y1test,bg_o_pre))

print(metrics.precision_score(y1test,bg_o_pre))

print(metrics.recall_score(y1test,bg_o_pre))

print(metrics.f1_score(y1test,bg_o_pre))

print(metrics.roc_auc_score(y1test,bg_o_pre))

fpr3, tpr3, threshold = metrics.roc_curve(y1test,bg_o_pre)

roc_auc3 = metrics.auc(fpr3, tpr3)
ab_o = AdaBoostClassifier(DecisionTreeClassifier(),random_state=13,n_estimators=20,learning_rate=1)

ab_o.fit(X1train,y1train)

ab_0_pre = ab_o.predict(X1test)

    

print(metrics.accuracy_score(y1test,ab_0_pre))

print(metrics.precision_score(y1test,ab_0_pre))

print(metrics.recall_score(y1test,ab_0_pre))

print(metrics.f1_score(y1test,ab_0_pre))

print(metrics.roc_auc_score(y1test,ab_0_pre))

fpr4, tpr4, threshold = metrics.roc_curve(y1test,ab_0_pre)

roc_auc4 = metrics.auc(fpr4, tpr4)
lr=LogisticRegression(random_state=2)

rf =RandomForestClassifier(random_state=60)

svm = SVC(random_state=20,kernel='poly')

nb=GaussianNB()

vc_o = VotingClassifier(estimators=[('lr',lr),('rf',rf),('svm',svm),('nb',nb)],voting='hard')

vc_o=vc_o.fit(X1train,y1train)

vc_0_pre=vc_o.predict(X1test)



print(metrics.accuracy_score(y1test,vc_0_pre))

print(metrics.precision_score(y1test,vc_0_pre))

print(metrics.recall_score(y1test,vc_0_pre))

print(metrics.f1_score(y1test,vc_0_pre))

print(metrics.roc_auc_score(y1test,vc_0_pre))

print(metrics.confusion_matrix(y1test,vc_0_pre))



fpr5, tpr5, threshold = metrics.roc_curve(y1test,vc_0_pre)

roc_auc5 = metrics.auc(fpr5, tpr5)
plt.title('Receiver Operating Characteristic')

plt.plot(fpr1, tpr1, 'b', label = 'Logistic---AUC = %0.2f' % roc_auc1)

plt.plot(fpr2, tpr2, 'g', label = 'DecisionTree---AUC = %0.2f' % roc_auc2)

plt.plot(fpr6, tpr6, 'r', label = 'RandomForest---AUC = %0.2f' % roc_auc6)

plt.plot(fpr3, tpr3, 'r', label = 'Bagging---AUC = %0.2f' % roc_auc3)

plt.plot(fpr4, tpr4, 'c', label = 'Boosting---AUC = %0.2f' % roc_auc4)

plt.plot(fpr5, tpr5, 'k', label = 'Voting---AUC = %0.2f' % roc_auc5)

plt.legend(loc = 'lower right')

plt.plot([0, 1], [0, 1],'r--')

plt.xlim([0, 1])

plt.ylim([0, 1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
t = PrettyTable(['Model', 'Accuracy', 'Recall','Precision','F1','roc_auc'])

t.add_row(['Random Forest',82,78,90,84,81])

t.add_row(['Logistic Regression',90,100,83,91,89])

t.add_row(['DecisionTree',79,82,78,79,79])

t.add_row(['Voting Classifier',89,84,95,90,88])

t.add_row(['Bagging',82,77,93,84,81])

t.add_row(['AdaBooating',84,85,85,85,84])
print(t)