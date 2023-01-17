# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
pd.set_option('display.max_rows',250)

df=pd.read_csv('/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.set_index('sl_no',inplace=True)

df['salary'].replace({np.nan:0},inplace=True)

df.head(10)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(8,8))





plt.subplot(2,2,1)

plt.pie(np.array(df['gender'].value_counts()),labels=['Male','Female'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('% of male and female in who sat in placement')



plt.subplot(2,2,2)

plt.pie(np.array(df.groupby('status').get_group('Placed')['gender'].value_counts()),labels=['Male','Female'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Placed students statistics by gender')



plt.subplot(2,2,3)

plt.pie(np.array(df.groupby('gender').get_group('M')['status'].value_counts()),labels=['Placed','Not Placed'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Male Students')





plt.subplot(2,2,4)

plt.pie(np.array(df.groupby('gender').get_group('F')['status'].value_counts()),labels=['Placed','Not Placed'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Female Students')



plt.tight_layout()



plt.figure(figsize=(8,8))



plt.subplot(4,1,1)

sns.barplot(y='status',x='ssc_p',hue='gender',data=df,orient='h')

plt.title('10th Percentage')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)





plt.subplot(4,1,2)

sns.barplot(y='status',x='hsc_p',hue='gender',data=df,orient='h')

plt.title('12th Percentage')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.subplot(4,1,3)

sns.barplot(y='status',x='degree_p',hue='gender',data=df,orient='h')

plt.title('Undergrad Percentage')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.subplot(4,1,4)

sns.barplot(y='status',x='mba_p',hue='gender',data=df,orient='h')

plt.title('MBA Percentage')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.tight_layout()


plt.figure(figsize=(8,8))



plt.subplot(2,2,1)

plt.pie(np.array(df.groupby('status').get_group('Placed')['workex'].value_counts()),labels=['No Work Exp','Work Exp'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Placed Students')



plt.subplot(2,2,2)

plt.pie(np.array(df.groupby('status').get_group('Not Placed')['workex'].value_counts()),labels=['No Work Exp','Work Exp'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Not Placed Students')



plt.subplot(2,2,3)

plt.pie(np.array(df.groupby('workex').get_group('Yes')['status'].value_counts()),labels=['Placed','Not Placed'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Work Exp')



plt.subplot(2,2,4)

plt.pie(np.array(df.groupby('workex').get_group('No')['status'].value_counts()),labels=['Placed','Not Placed'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('No Work Exp')



plt.tight_layout()
plt.pie(np.array(df['workex'].value_counts()),labels=['No','Yes'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Students having Work Exp')



plt.tight_layout()
plt.figure(figsize=(8,8),frameon=False)



plt.subplot(2,1,1)

sns.scatterplot(x=np.array(df['status']),y=np.array(df['etest_p']))

plt.ylabel('Employablility Test percentage')

plt.xlabel('Placement Status')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.subplot(2,1,2)

sns.barplot(x='status',y='etest_p',hue='gender',data=df,orient='v')

plt.title('Employability Test Percentage')

ax=plt.gca()

ax.spines['right'].set_visible(False)

ax.spines['top'].set_visible(False)



plt.tight_layout()
plt.figure(figsize=(8,8))



sns.countplot(x='specialisation',hue='status',data=df)

plt.legend(frameon=False)

plt.ylabel('Number of Candidates')

plt.yticks(range(0,120,10))

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize=(8,8))



sns.countplot(x='degree_t',hue='status',data=df)

plt.legend(frameon=False)

plt.ylabel('Number of Candidates')

plt.yticks(range(0,120,10))

ax=plt.gca()

ax.spines['top'].set_visible(False)

ax.spines['right'].set_visible(False)
plt.figure(figsize=(8,8))



plt.subplot(1,2,1)

plt.pie(np.array(df.groupby('degree_t').get_group('Comm&Mgmt')['status'].value_counts()),labels=['Placed','Unplalced'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Commerce and Finance')



plt.subplot(1,2,2)

plt.pie(np.array(df.groupby('degree_t').get_group('Sci&Tech')['status'].value_counts()),labels=['Placed','Unplalced'],autopct='%1.1f%%',wedgeprops={'edgecolor':'black'})

plt.title('Science and Engineering')



plt.tight_layout()
df_modified=df.drop(['salary'],axis=1)



df_modified['workex'].replace({'No':0,'Yes':1},inplace=True)

df_modified['status'].replace({'Placed':1,'Not Placed':0},inplace=True)

from sklearn.model_selection import train_test_split

df_modified=pd.get_dummies(df_modified)



xtrain,xtest,ytrain,ytest=train_test_split(df_modified.drop('status',axis=1),df_modified['status'],test_size=0.33,random_state=0)
from sklearn.ensemble import GradientBoostingClassifier



clf1=GradientBoostingClassifier(learning_rate=0.5,n_estimators=100,max_depth=3,random_state=0)

clf1.fit(xtrain,ytrain)



print('train score: '+str(clf1.score(xtrain,ytrain)))

print('test score: '+str(clf1.score(xtest,ytest)))

print()

print('feature importance:-')

print(list(zip(xtrain.columns,clf1.feature_importances_)))
from sklearn.ensemble import RandomForestClassifier



clf2=RandomForestClassifier(max_depth=20,n_estimators=100,random_state=0)

clf2.fit(xtrain,ytrain)



print('train score: '+str(clf2.score(xtrain,ytrain)))

print('test score: '+str(clf2.score(xtest,ytest)))
from sklearn.preprocessing import MinMaxScaler



scaler=MinMaxScaler()

xtrain2=scaler.fit_transform(xtrain)

xtest2=scaler.transform(xtest)
from sklearn.svm import SVC



clf3=SVC(C=10,kernel='rbf',gamma=0.01,random_state=0)

clf3.fit(xtrain2,ytrain)



print('train score: '+str(clf3.score(xtrain2,ytrain)))

print('test score: '+str(clf3.score(xtest2,ytest)))
from sklearn.linear_model import LogisticRegression



clf4=LogisticRegression(C=8.0,max_iter=1000,random_state=0)

clf4.fit(xtrain2,ytrain)



print('train score: '+str(clf4.score(xtrain2,ytrain)))

print('test score: '+str(clf4.score(xtest2,ytest)))