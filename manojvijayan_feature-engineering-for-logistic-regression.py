# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/turnover.csv')
df.info()
df.describe().T
df[['sales', 'salary']].describe().T
(df.salary.value_counts()/len(df.salary))*100
((df.sales.value_counts()/len(df.sales))*100).plot(kind='bar')
df.satisfaction_level.hist()
plt.figure(figsize=(35,7))
((df.satisfaction_level.value_counts().sort_index()/len(df.satisfaction_level))*100).plot(kind='bar')
df.satisfaction_level.plot(kind='box')
df.left.value_counts()/len(df.left)
df.satisfaction_level = df.satisfaction_level.astype('category')
ax = pd.Series((df[df.left==0].satisfaction_level.value_counts()/len(df.left))*100).sort_index().plot(kind='bar',color='g',figsize=(35,10))
pd.Series((df[df.left==1].satisfaction_level.value_counts()/len(df.left))*100).sort_index().plot(kind='bar',color='r',alpha= 0.7,figsize=(35,10), ax=ax)
ax.legend(["Stayed", "Left"])
plt.figure(figsize=(5,5))
ax = ((df[df.left==0].salary.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g')
((df[df.left==1].salary.value_counts().sort_index()/len(df.left)*100)).plot(kind='bar',color='r',alpha= 0.7, ax= ax)
ax.legend(["Stayed", "Left"])
plt.figure(figsize=(35,7))
((df.last_evaluation.value_counts().sort_index()/len(df.last_evaluation))*100).plot(kind='bar')
plt.figure(figsize=(35,5))
((df[df.left==1].last_evaluation.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].last_evaluation.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
plt.figure(figsize=(5,2))
((df[df.left==0].number_project.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g')
((df[df.left==1].number_project.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r',alpha=0.7)
plt.figure(figsize=(40,5))
((df[df.left==1].average_montly_hours.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].average_montly_hours.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
plt.figure(figsize=(10,5))
((df[df.left==1].time_spend_company.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].time_spend_company.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
plt.figure(figsize=(5,3))
((df[df.left==1].Work_accident.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].Work_accident.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
plt.figure(figsize=(35,7))
((df[df.Work_accident ==1].satisfaction_level.value_counts().sort_index()/len(df[df.Work_accident ==1]))*100).plot(kind='bar')
plt.figure(figsize=(5,3))
((df[df.left==1].promotion_last_5years.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].promotion_last_5years.value_counts().sort_index()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
plt.figure(figsize=(5,3))
((df[df.left==1].sales.value_counts()/len(df.left))*100).plot(kind='bar',color='r')
((df[df.left==0].sales.value_counts()/len(df.left))*100).plot(kind='bar',color='g',alpha=0.4)
df.salary.value_counts()
lkup = {"low": 0, "medium": 1, "high": 2}
df['sal_num'] = df['salary'].map(lkup)
df.drop('salary', inplace=True, axis=1)
df.sal_num.value_counts()
plt.figure(figsize=(10,5))
sns.heatmap(df.corr(),cbar=True,fmt =' .2f', annot=True, cmap='coolwarm')
df = pd.concat([df, pd.get_dummies(df['sales'],prefix='sl', prefix_sep='_')], axis=1)
df.drop('sales', inplace=True, axis=1)
y = df.left.values
df.drop('left', inplace=True, axis=1)
df_NK = df.copy()

bins = [0, 0.11, 0.35, 0.46, 0.71, 0.92,1.0]
df_NK['satisfaction_level_bin'] = pd.cut(df_NK.satisfaction_level,bins)

bins = [0, 0.47, 0.48, 0.65, 0.88, 0.89,1.0]
df_NK['last_evaluation_bin'] = pd.cut(df_NK.last_evaluation,bins)

lkup = { 3: "low", 4 : "medium", 5 : "medium",  2: "high", 6: "high", 7: "Very high"}
df_NK['number_project_cat'] = df_NK['number_project'].map(lkup)

bins = [96, 131, 165, 178, 179, 259, 287]
df_NK['average_montly_hours_bin'] = pd.cut(df_NK.average_montly_hours,bins)

lkup = { 2: "low", 3 : "medium", 4 : "medium", 6 : "medium", 5: "high", 7: "very low", 8: "very low", 10: "very low"}
df_NK['time_spend_company_cat'] = df_NK['time_spend_company'].map(lkup)

df_NK = pd.concat([df_NK, pd.get_dummies(df_NK['satisfaction_level_bin'],prefix='sts', prefix_sep='_')], axis=1)
df_NK.drop('satisfaction_level', inplace=True, axis=1)
df_NK.drop('satisfaction_level_bin', inplace=True, axis=1)

df_NK = pd.concat([df_NK, pd.get_dummies(df_NK['last_evaluation_bin'],prefix='le', prefix_sep='_')], axis=1)
df_NK.drop('last_evaluation_bin', inplace=True, axis=1)
df_NK.drop('last_evaluation', inplace=True, axis=1)

df_NK = pd.concat([df_NK, pd.get_dummies(df_NK['number_project_cat'],prefix='np', prefix_sep='_')], axis=1)
df_NK.drop('number_project_cat', inplace=True, axis=1)
df_NK.drop('number_project', inplace=True, axis=1)

df_NK = pd.concat([df_NK, pd.get_dummies(df_NK['average_montly_hours_bin'],prefix='am', prefix_sep='_')], axis=1)
df_NK.drop('average_montly_hours_bin', inplace=True, axis=1)
df_NK.drop('average_montly_hours', inplace=True, axis=1)

df_NK = pd.concat([df_NK, pd.get_dummies(df_NK['time_spend_company_cat'],prefix='tsc', prefix_sep='_')], axis=1)
df_NK.drop('time_spend_company_cat', inplace=True, axis=1)
df_NK.drop('time_spend_company', inplace=True, axis=1)

df_NK.info()
# Split into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=0.50)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
# Split into train and test dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_NK, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=0.50)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(2)
# Split into train and test dataset
X_train, X_test, y_train, y_test = train_test_split(pol.fit_transform(df_NK.as_matrix()), y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=0.5)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df_PCA= sc.fit_transform(df_NK)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_PCA, y, test_size=0.30, random_state=101)
from sklearn.decomposition import PCA
pca_comp = 2
pca = PCA(n_components=pca_comp)
#pca = PCA(n_components=df_NK.shape[1])
pca.fit(X_train)
#pca_mat = pd.DataFrame(pca.components_.T, columns=['PC' + str(i) for i in range(1,df_NK.shape[1]+1) ], index=df_NK.columns)
pca_mat = pd.DataFrame(pca.components_.T, columns=['PC' + str(i) for i in range(1,pca_comp+1) ], index=df_NK.columns)
#pca_mat = pca_mat*pca.explained_variance_ratio_
pca_mat= pca_mat.abs()
pca_mat['lin_influ'] = pca_mat.sum(axis=1)
from sklearn.preprocessing import PolynomialFeatures
pol = PolynomialFeatures(2)
X_train, X_test, y_train, y_test = train_test_split(pol.fit_transform(df_NK[pca_mat.lin_influ.nlargest(29).index].as_matrix()), y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',C=0.4)
lr.fit(X_train,y_train)
pred=lr.predict(X_test)
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
