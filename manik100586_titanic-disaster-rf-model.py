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
import numpy as np # linear algebra

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib import style

%matplotlib inline



# data visualization

import seaborn as sns



# Algorithms

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB
df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_full_df = pd.read_csv('/kaggle/input/titanic/test.csv')

test_df= test_full_df

print("Training data: {}".format(df.shape))

print("Testing data: {}".format(test_df.shape))
df.info()
df.describe()
df.columns.values
df = df.drop(['Name', 'Ticket', 'Cabin','Parch'], axis=1)

test_df = test_df.drop(['Name', 'Ticket', 'Cabin','Parch'], axis=1)
sns.catplot(x ='Embarked', hue ='Survived',  

kind ='count', col ='Pclass', data = df) 
sns.catplot(x ="Sex", hue ="Survived",  

kind ="count", data = df) 
group = df.groupby(['Pclass', 'Survived']) 

pclass_survived = group.size().unstack() 

  

# Heatmap - Color encoded 2D representation of data. 

sns.heatmap(pclass_survived, annot = True, fmt ="d") 
sns.violinplot(x ="Sex", y ="Age", hue ="Survived",  

data = df, split = True) 
df = df.drop(['PassengerId'], axis=1)

test_df = test_df.drop(['PassengerId'], axis=1)
df['Age'] = df['Age'].fillna(df['Age'].median())

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())



sns.distplot(df['Age'])
df['Embarked'].describe()
df['Embarked'] = df['Embarked'].fillna("S")

test_df['Embarked'] = test_df['Embarked'].fillna("S")
def grouping_Age(x):

    if x in range(0, 21):

        return 1

    elif x in range(21, 41):

        return 2

    else:

        return 3

    

    

df['Age group']= df['Age'].apply(grouping_Age)

test_df['Age group']= test_df['Age'].apply(grouping_Age)
age_index = df['Age group'].value_counts().index

age_values = df['Age group'].value_counts()

age_values
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)

cols =['r','y','b']

df['Age group'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,colors=cols)
df = pd.get_dummies(df, columns = ["Embarked","Age group", "Pclass","Sex"],

                             prefix=["Em_type", "Age_group", "Pclass_","Sex_"])



df.drop(['Age','Fare'],axis=1,inplace=True)

#df = pd.concat([df,sex,embark],axis=1)

test_df = pd.get_dummies(test_df, columns = ["Embarked","Age group", "Pclass","Sex"],

                             prefix=["Em_type", "Age_group", "Pclass_","Sex_"])

test_df.drop(['Age','Fare'],axis=1,inplace=True)
y = df['Survived']

X = df.drop(['Survived'], axis=1)
X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3,random_state=101)

X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



accuracy_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

accuracy_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



accuracy_log = round(logreg.score(X_train, Y_train) * 100, 2)
knn = KNeighborsClassifier(n_neighbors = 3) 

knn.fit(X_train, Y_train)  

Y_pred = knn.predict(X_test)  

accuracy_knn = round(knn.score(X_train, Y_train) * 100, 2)
gaussian = GaussianNB() 

gaussian.fit(X_train, Y_train)  

Y_pred = gaussian.predict(X_test)  

accuracy_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



accuracy_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



accuracy_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
decision_tree = DecisionTreeClassifier() 

decision_tree.fit(X_train, Y_train)  

Y_pred = decision_tree.predict(X_test)  

accuracy_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [accuracy_linear_svc, accuracy_knn, accuracy_log, 

              accuracy_random_forest, accuracy_gaussian, accuracy_perceptron, 

              accuracy_sgd, accuracy_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(15)
from sklearn.model_selection import cross_val_score

rf = RandomForestClassifier(n_estimators=100)

scores = cross_val_score(rf, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
Y_pred = random_forest.predict(test_df)

Y_pred
submission = pd.DataFrame({

        "PassengerId": test_full_df["PassengerId"],

        "Survived": Y_pred})
submission
submission.to_csv("submission.csv",index=False)