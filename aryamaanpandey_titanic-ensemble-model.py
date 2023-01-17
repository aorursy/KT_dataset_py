# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler,Normalizer,RobustScaler

import re as re

import itertools

import seaborn as sns

import matplotlib.gridspec as gridspec

from sklearn import datasets

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB 

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from mlxtend.classifier import StackingClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, train_test_split

from mlxtend.plotting import plot_learning_curves

from mlxtend.plotting import plot_decision_regions

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.info() #We can see that Age,Cabin and Embarked values are missing

#test.info() #We can see that Age,Fare and Cabin values are missing

dataset = [train,test] # Created dataset

plt.hist(train['Age'],50)

print("Mean age = "  + str(train['Age'].mean()))

print("Median age = " + str(train['Age'].median()))


for data in dataset:

    age_median = data['Age'].median()

    age_deviation = data['Age'].std()

    age_missing_count = data['Age'].isnull().sum() #count of missing values

    age_values = np.random.normal(age_median,age_deviation,age_missing_count) #generating random values of age

    data['Age'][np.isnan(data['Age'])] = age_values

train['Age'] = train['Age'].astype(int) #Converting from float to int

test['Age'] = test['Age'].astype(int)

#train['Age'].count()
train = train.drop(columns = ['Cabin','Ticket','PassengerId','Pclass'])

#test = test.drop(columns = ['Cabin','Ticket','PassengerId','Pclass'])

#test.head()
train['Sex'] = train['Sex'].map( {'male': 1,'female': 0} )

test['Sex'] = test['Sex'].map( {'male': 1,'female': 0} )
val = ['Q','C','S']

count_missing_train = train['Embarked'].isnull().sum()

count_missing_test = test['Embarked'].isnull().sum()

for i in range(count_missing_train):

    train['Embarked'] = train['Embarked'].fillna(val[np.random.randint(891) % 3])

for i in range(count_missing_test):

    test['Embarked'] = test['Embarked'].fillna(val[np.random.randint(891) % 3])

def convert(val):

    if(val == 'Q'):

        return 0

    elif(val == 'S'):

        return 1

    else:

        return 2

train['Embarked'] = train['Embarked'].map(convert)

test['Embarked'] = test['Embarked'].map(convert)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#test['Fare'].isna().sum()



def name(val):

    get_title = re.search(' ([A-Za-z]+)\.', val)

    if get_title:

        return len(get_title.group(1))



train['Name'] = train['Name'].map(name)

test['Name'] = test['Name'].map(name)
train.info()
train_X = train[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']]

y = train['Survived']

test_X = test[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']]

scale = RobustScaler()

#train_X[['Age','Fare']] = scale.fit_transform(train_X[['Age','Fare']].as_matrix())

#test_X[['Age','Fare']] = scale.fit_transform(test_X[['Age','Fare']].as_matrix())

train_X[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']] = scale.fit_transform(train_X[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']].as_matrix())

test_X[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']] = scale.fit_transform(test_X[['Name','Sex','Age','SibSp','Parch','Fare','Embarked']].as_matrix())

train_X.head()
test_X.head()
clf1 = RandomForestClassifier(random_state=1)

clf2 = GaussianNB()

clf3 = SVC(C=1, kernel='rbf', degree=4)

#clf4 = DecisionTreeClassifier()

#clf5 = KNeighborsClassifier(n_neighbors=1)

clf6 = AdaBoostClassifier()

clf7 = GradientBoostingClassifier()

clf8 = LinearDiscriminantAnalysis()

clf9 = QuadraticDiscriminantAnalysis()

lr = LogisticRegression()

sclf = StackingClassifier(classifiers=[clf1,clf2,clf3,clf6,clf7,clf8,clf9], meta_classifier=lr)
label = ['RF','NB','SVC','ADA','GB','LD','QD','Stacking Classifier']

clf_list = [clf1,clf2,clf3,clf6,clf7,clf8,clf9,sclf]

    



clf_cv_mean = []

clf_cv_std = []

for clf, label in zip(clf_list, label):

        

    scores = cross_val_score(clf, train_X, y, cv=10, scoring='accuracy')

    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))

    clf_cv_mean.append(scores.mean())

    clf_cv_std.append(scores.std())

        

    clf.fit(train_X, y)

plt.figure()

(_, caps, _) = plt.errorbar(range(8), clf_cv_mean, yerr=clf_cv_std, c='blue', fmt='-o', capsize=5)

for cap in caps:

    cap.set_markeredgewidth(1)                                                                                                                                

plt.xticks(range(8), ['RF','NB','SVC','ADA','GB','LD','QD','Stacking Classifier'])        

plt.ylabel('Accuracy'); plt.xlabel('Classifier'); plt.title('Stacking Ensemble');

plt.show()
X_train, X_test, y_train, y_test = train_test_split(train_X, y, test_size=0.3, random_state=42)

    

plt.figure()

plot_learning_curves(X_train, y_train, X_test, y_test, sclf, print_model=False, style='ggplot')

plt.show()
ans = pd.DataFrame(columns = ['PassengerId','Survived'])

for i in range(test.shape[0]):

    id = int(test['PassengerId'][i])

    prediction = int(sclf.predict([[test_X['Name'][i],test_X['Sex'][i],test_X['Age'][i],test_X['SibSp'][i],test_X['Parch'][i],test_X['Fare'][i],test_X['Embarked'][i]]]))

    ans = ans.append({'PassengerId' : id,'Survived' : prediction}, ignore_index=True)

ans.head()                      
ans.to_csv('final.csv',index=False)