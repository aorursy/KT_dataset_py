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
df = pd.read_csv('/kaggle/input/hr-analytics/HR_comma_sep.csv')
import seaborn as sns

from matplotlib import pyplot as plt
df.left = df.left.map({

    0:'False',  #True (1) or False (0) in boolean logic. 

    1:'True'

})

df
df.dtypes
df.isnull().sum()
df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)
_=sns.heatmap(df.drop(['Department', 'salary'], axis=1).corr(), annot=True,cmap='YlGnBu')


plt.figure(figsize=(8,8))

ax=sns.countplot(data=df,x=df['left'],order=df['left'].value_counts().index)

plt.xlabel('Target Variable- Salary')

plt.ylabel('Distribution of target variable')

plt.title('Distribution of Salary')

total = len(df)

for p in ax.patches:

        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+100))


ax=pd.crosstab(df.salary,df.left).plot(kind='bar',figsize=(8,8))

for p in ax.patches:

        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+100))


ax=pd.crosstab(df.Department,df.left).plot(kind='bar',figsize=(8,8))

for p in ax.patches:

        ax.annotate('{:.1f}%'.format(100*p.get_height()/total), (p.get_x()+0.1, p.get_height()+60))
_=sns.boxplot(y=df.left,x=df.satisfaction_level,orient='horizontal',hue=df.left, palette="Set2")

_=sns.boxplot(y=df.left,x=df.last_evaluation,orient='horizontal',hue=df.left, palette="Set3")
_=sns.boxplot(y=df.left,x=df.number_project,orient='horizontal',hue=df.left, palette="Set1")
a=df[(df.left=='False') & (df.number_project==6)]

print(df.shape)

a
df=df[df.apply(lambda x: x.values.tolist() not in a.values.tolist(), axis=1)]

df.shape
sns.boxplot(y=df.left,x=df.number_project,orient='horizontal',hue=df.left, palette="Set1")
_=sns.boxplot(y=df.left,x=df.average_montly_hours,orient='horizontal',hue=df.left, palette="Set2")
_=sns.boxplot(y=df.left,x=df.time_spend_company,orient='horizontal',hue=df.left, palette="Set3")
df.shape
df = df[df.time_spend_company<8]

df.shape
sns.boxplot(y=df.left,x=df.promotion_last_5years,orient='horizontal',hue=df.left, palette="Set1")
df.drop('promotion_last_5years', inplace=True, axis=1)

df.head()
_=sns.distplot(df['satisfaction_level'],kde=False)
_=sns.distplot(df['last_evaluation'],kde=False)
_=sns.distplot(df['number_project'],kde=False)
_=sns.distplot(df['average_montly_hours'],kde=False)
_=sns.distplot(df['time_spend_company'],kde=False)
df.head()
dummies1 = pd.get_dummies(df.salary)

dummies1 = dummies1.drop('low', axis=1)

dummies1
df.drop('salary',axis=1,inplace=True)
dummies2 = pd.get_dummies(df.Department)

dummies2 = dummies2.drop('sales', axis=1)

dummies2
df.drop('Department',axis=1,inplace=True)
df = pd.concat([df,dummies1], axis='columns') 

df = pd.concat([df,dummies2], axis='columns') 

df
X = df.drop('left', axis=1).values

X
y = df.left

y
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit(X).transform(X.astype(float))

X
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=10)

print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import jaccard_score

from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.linear_model import LogisticRegression
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4) #max_depth maximum depth of tree

drugTree.fit(X_train,y_train) 
#itterating to find the best k value

Ks = 10

mean_acc = np.zeros((Ks-1))

std_acc = np.zeros((Ks-1))

ConfustionMx = [];

for n in range(1,Ks):

    

    #Train Model and Predict  

    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)

    yhat=neigh.predict(X_test)

    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)



    

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])



mean_acc
#Plot model accuracy for Different number of Neighbors



plt.plot(range(1,Ks),mean_acc,'g')

plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10) #showing 68% data

plt.legend(('Accuracy ', '+/- 1xstd'))

plt.ylabel('Accuracy ')

plt.xlabel('Number of Nabors (K)')

plt.tight_layout()

plt.show()

#Training



k = 1

#Train Model and Predict  

neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)

neigh

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)

LR

print("Decision Tree's Accuracy: ", metrics.accuracy_score(y_test, drugTree.predict(X_test)))

print("KNN's Accuracy: ", metrics.accuracy_score(y_test, neigh.predict(X_test)))

print("LR's Accuracy: ", metrics.accuracy_score(y_test, LR.predict(X_test)))

print("DT:", classification_report(y_test,  drugTree.predict(X_test)))

print("KNN:", classification_report(y_test,  neigh.predict(X_test)))

print("Logr:", classification_report(y_test, LR.predict(X_test)))
