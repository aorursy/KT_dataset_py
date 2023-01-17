import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
plt.style.use('ggplot')
df = pd.read_csv("../input/HR_comma_sep.csv", delimiter=',')
df.info()
df.describe()
corr = df.corr()
corr
sns.heatmap(corr)
f, axes = plt.subplots(ncols=3,figsize=(17,6))

sns.countplot(x='salary',hue='left',ax=axes[0],data=df)
sns.countplot(x='promotion_last_5years',hue='left',ax=axes[1],data=df)
sns.countplot(x='sales',hue='left',ax=axes[2],data=df)
plt.xticks(rotation=90)
plt.xlabel('Department')
f, axes = plt.subplots(ncols=3,figsize=(17,6))

a0 = sns.countplot(x='number_project',hue='left',ax=axes[0],data=df)
a0.set_title('Number of Projects Completed')
a1 = sns.countplot(x='Work_accident',hue='left',ax=axes[1],data=df)
a1.set_title('Work Accident (Y/N)')
a2 = sns.countplot(x='time_spend_company',hue='left',ax=axes[2],data=df)
a2.set_title('Number of Years at the Company')
sns.barplot(x='sales',y='satisfaction_level',hue='left',data=df)
plt.xlabel('Department')
plt.xticks(rotation=90)
plt.title('Satisfaction Level in Each Department')
left_yes = df[df['left'] == 1]
left_no = df[df['left'] == 0]
sns.distplot(left_no.satisfaction_level,label='0')
sns.distplot(left_yes.satisfaction_level,label='1')
plt.legend(title='left',loc='best')
sns.barplot(x='sales',y='average_montly_hours',hue='left',data=df)
plt.xlabel('Department')
plt.xticks(rotation=90)
plt.title('Average hours clocked by employees in each department')
sns.distplot(left_no.average_montly_hours,label='0')
sns.distplot(left_yes.average_montly_hours,label='1')
plt.legend(title='left',loc='best')
sns.barplot(x='sales',y='last_evaluation',hue='left',data=df)
plt.xlabel('Department')
plt.xticks(rotation=90)
plt.title('Last evaluation')
lst = ['sales','technical','support','IT','product_mng','marketing','RandD','accounting','hr','management']

for i, pos in enumerate(lst):
    df.sales.replace(to_replace=pos,value=i,inplace=True)
df.salary.value_counts()
lst = ['low','medium','high']

for i, sal in enumerate(lst):
    df.salary.replace(to_replace=sal,value=i,inplace=True)
df.info()
x = df.drop('left',axis=1)
y = df.left
y.values.reshape(-1,1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
x_dev,x_test,y_dev,y_test = train_test_split(x_test,y_test,test_size=0.5)
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)
x_dev = scaler.transform(x_dev)
x_test = scaler.transform(x_test)
clf = DecisionTreeClassifier(min_samples_split=3,)
clf.fit(x_train,y_train)
pred = clf.predict(x_train)
print('Training Report\n {}'.format(classification_report(y_train,pred)))
print('Training accuracy: {:.3f}'.format(accuracy_score(y_train,pred)))
pred = clf.predict(x_dev)
print('Dev set Report\n {}'.format(classification_report(y_dev,pred)))
print('Dev set accuracy: {:.3f}'.format(accuracy_score(y_dev,pred)))
print('AUC: {:.3f}'.format(roc_auc_score(y_dev,pred)))
clf = RandomForestClassifier(n_estimators=300)
clf.fit(x_train,y_train)
pred = clf.predict(x_train)
print('Training Report\n {}'.format(classification_report(y_train,pred)))
print('Training accuracy: {:.3f}'.format(accuracy_score(y_train,pred)))
pred = clf.predict(x_dev)
print('Dev set Report\n {}'.format(classification_report(y_dev,pred)))
print('Dev set accuracy: {:.3f}'.format(accuracy_score(y_dev,pred)))
print('AUC: {:.3f}'.format(roc_auc_score(y_dev,pred)))
clf = SVC(C=500)
clf.fit(x_train,y_train)
pred = clf.predict(x_train)
print('Training Report\n {}'.format(classification_report(y_train,pred)))
print('Training accuracy: {:.3f}'.format(accuracy_score(y_train,pred)))
pred = clf.predict(x_dev)
print('Dev set Report\n {}'.format(classification_report(y_dev,pred)))
print('Dev set accuracy: {:.3f}'.format(accuracy_score(y_dev,pred)))
print('AUC: {:.3f}'.format(roc_auc_score(y_dev,pred)))
clf = RandomForestClassifier(n_estimators=300)
clf.fit(x_train,y_train)
pred = clf.predict(x_test)
print('Test set Report\n {}'.format(classification_report(y_test,pred)))
print('Test set accuracy: {:.3f}'.format(accuracy_score(y_test,pred)))
print('AUC: {:.3f}'.format(roc_auc_score(y_test,pred)))
