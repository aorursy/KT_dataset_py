# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd

import math



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier



#(yes i know i probably dont need most of this...)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.





train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')

combine = [train_df, test_df]



combine

# preview the data

train_df[:10]



def isNumber(s):

    try:

        s=float(s)

        #print("s=",s)

        return s

    except ValueError:

        return -1





def cellsNULL(data):

    i=0

    #x=0

    n=len(data)

    while(i<n):    

        data[i]=isNumber(data[i])

        #print every 100th line to show that the code is still running and show how much longer its taking

        if(i%100==0):

            print(i)

        i=i+1

    return data
for dataset in combine:

    dataset['Sex'] = dataset['Sex'].replace('male', 0)

    dataset['Sex'] = dataset['Sex'].replace('female', 1)

    

    

#train_df[:100000]

#print(combine[0]['Embarked'][829])

for dataset in combine:

    dataset['Embarked'] = dataset['Embarked'].replace('C', 0)

    dataset['Embarked'] = dataset['Embarked'].replace('S', 1)

    dataset['Embarked'] = dataset['Embarked'].replace('Q', 2)

    #dataset['Embarked']=cellsNULL(dataset['Embarked'])

    dataset['Embarked'] = dataset['Embarked'].fillna(value=-1, method=None, axis=None, inplace=False, limit=None, downcast=None)

    
#train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

#train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)







for dataset in combine:    

    dataset.loc[ dataset['Age'] <= 7, 'Age'] = 0

    dataset.loc[(dataset['Age'] > 7) & (dataset['Age'] <= 17), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 17) & (dataset['Age'] <= 22), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 30), 'Age'] = 3

    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 40), 'Age'] = 4

    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 50), 'Age'] = 5

    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 60), 'Age'] = 6

    dataset.loc[ dataset['Age'] > 60, 'Age'] = 7

    dataset['Age'] = dataset['Age'].fillna(value=-1, method=None, axis=None, inplace=False, limit=None, downcast=None)



train_df.head()





#train_df = train_df.drop(['AgeBand'], axis=1)

combine = [train_df, test_df]

train_df.head()
#combine age classes with Pclasses

#for dataset in combine: 

#    dataset.assign(ClassAge=np.nan)

#    dataset['ClassAge'] = int(str(dataset['Pclass'])+str(dataset['Age']))

#    print(dataset['ClassAge'])

    #dataset.loc[(dataset['Age'] > 7) & (dataset['Age'] <= 17), 'Age'] = 1

    

#drop colloums i dont want

for dataset in combine: 

    dataset=dataset.drop(['Name','Ticket'], axis=1)

    for c in dataset.columns:

        #dataset[c] = dataset[c].drop('')

        dataset[c] = dataset[c].fillna(value=-1, method=None, axis=None, inplace=False, limit=None, downcast=None)

print(train_df.columns.values)

train_df=train_df.drop(['Name','Ticket', 'Cabin'], axis=1)

test_df=test_df.drop(['Name','Ticket', 'Cabin'], axis=1)



print(train_df.dtypes)

for c in train_df.columns:

    train_df[c] = train_df[c].fillna(value=-1, method=None, axis=None, inplace=False, limit=None, downcast=None)

for c in test_df.columns:

    test_df[c] = test_df[c].fillna(value=-1, method=None, axis=None, inplace=False, limit=None, downcast=None)
#train_df = combine['train_df']

#test_df = combine['test_df']
X_train = train_df.drop("Survived", axis=1)

X_train = X_train.drop("PassengerId", axis=1)

Y_train = train_df["Survived"]

X_test  = test_df.drop("PassengerId", axis=1).copy()

#X_train  = train_df.drop("PassengerId", axis=1).copy()



X_train.shape, Y_train.shape, X_test.shape
# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
coeff_df = pd.DataFrame(train_df.columns.delete(0))

coeff_df.columns = ['Feature']

coeff_df["Correlation"] = pd.Series(logreg.coef_[0])



coeff_df.sort_values(by='Correlation', ascending=False)
# Support Vector Machines



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc
knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

acc_random_forest
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)