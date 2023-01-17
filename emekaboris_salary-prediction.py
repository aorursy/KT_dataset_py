# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory







# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/Salary_Data.csv')
df.head()
sns.distplot(df['YearsExperience'], kde=False, bins=10)

#this plot is used to visualise distribution
sns.barplot(x='YearsExperience', y='Salary', data=df)
sns.distplot(df.Salary) #lets see the flow of salary 
#Now lets split our data set into training and testing set
x=df.iloc[:, :-1].values

y=df.iloc[:, 1].values



#spiliting to independent and dependent variables
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size=1/3, random_state=0)
#let's fit the training data

lr.fit(x_train, y_train)
#Lets create a Linear regresion model 

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train, y_train)

y_pred=lr.predict(x_test)

y_pred
plt.scatter(x_train,y_train,color='blue')

plt.plot(x_train, lr.predict(x_train),color='red')

plt.title('Salary vs Years of Experience(Training Data)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary of an employee')

plt.show()
plt.scatter(x_test,y_test,color='blue')

plt.plot(x_test, lr.predict(x_test),color='red')

plt.title('Salary vs Years of Experience(Test Data)')

plt.xlabel('Years of Experience')

plt.ylabel('Salary of an employee')

plt.show()
#Knn Algorithm

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=4)

knn.fit(x_train, y_train)

y_predict=knn.predict(x_test)

y_predict
#Svm Clasifier

from sklearn import svm

clf = svm.SVC(kernel='linear')

clf.fit(x_train, y_train)

y_prediction=clf.predict(x_test)

y_prediction
#Lets check our model accuracy using r2_score

from sklearn.metrics import r2_score

r2_score(y_test,y_prediction)#for svm classifier
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_jobs=2, random_state=0)



clf.fit(x_train, y_train)

RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',

            max_depth=None, max_features='auto', max_leaf_nodes=None,

            min_impurity_split=1e-07, min_samples_leaf=1,

            min_samples_split=2, min_weight_fraction_leaf=0.0,

            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,

            verbose=0, warm_start=False)
preds = clf.predict(x_test)

print(preds)
r2_score(y_test,preds) #for knn algorithm
r2_score(y_test,y_pred)#for linear regression
train_set=df['YearsExperience']

train_labels=df['Salary']
print("Coefficients: ", lr.coef_)

print("Intercept: ", lr.intercept_)