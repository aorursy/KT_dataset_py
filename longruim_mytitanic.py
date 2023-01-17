# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd



train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')



print(train_df.shape, test_df.shape)
train_df.head(2)
train_df.describe()
test_df.head(2)
test_df.describe()
# 取出标签值

train_labels = train_df.get("Survived")



train_df = train_df.drop("Survived", axis=1)



print(train_labels.shape, train_df.shape)
# 决定将训练与测试样本放在一起进行预处理

total_df = train_df.copy()

total_df = total_df.append(test_df)

total_df.shape
# 去除 PassengerId、Name、Ticket、Cabin

drop_list = ["PassengerId","Name","Ticket","Cabin"]

for title in drop_list:

    total_df = total_df.drop(title, axis=1)



total_df.shape
total_df.head()
# 查看哪些列里面有空值

have_null = total_df.isnull().any().values

for i in range(len(have_null)):

    if have_null[i]:

        print(total_df[total_df.columns[i]].isnull().value_counts())

        print("_"*40)

    
# 先填空缺值,将年龄为空的填为-1,将登陆口填为最多的那个口，将 Fare 填为均值

total_df["Age"] = total_df["Age"].fillna(value = -1)

total_df["Embarked"] = total_df["Embarked"].fillna(value = total_df["Embarked"].value_counts().index[0])

total_df["Fare"] = total_df["Fare"].fillna(value = total_df["Fare"].mean())



# 将字符型数值进行映射

total_df["Sex"] = total_df["Sex"].map({"male": 0, "female": 1}).astype(int)

total_df["Embarked"] = total_df["Embarked"].map({"S": 0, "C": 1, "Q": 2}).astype(int)



# 将 Fare 进行归一化

total_df["Fare"] = (total_df["Fare"] - total_df["Fare"].min()) / (total_df["Fare"].max() - total_df["Fare"].min())



total_df.head()

# 分回 train 和 test

train_df = total_df[:train_df.shape[0]]

test_df = total_df[train_df.shape[0]:]



X_train = train_df

Y_train = train_labels

X_test = test_df



print(X_train.shape, Y_train.shape, X_test.shape)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print("LogisticRegression =", acc_log)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

print("SVC =", acc_svc)
# KNN

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print("knn =", acc_knn)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print("GaussianNB =", acc_gaussian)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print("Perceptron =", acc_perceptron)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print("LinearSVC =", acc_linear_svc)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

print("SGDClassifier =", acc_sgd)
# Decision Tree

from sklearn.tree import DecisionTreeClassifier



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print("DecisionTreeClassifier =", acc_decision_tree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, Y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print("RandomForestClassifier =", acc_random_forest)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

models.sort_values(by='Score', ascending=False)
Y_pred = random_forest.predict(X_test)

test_df = pd.read_csv('../input/test.csv')



submission = pd.DataFrame({

        "PassengerId": test_df["PassengerId"],

        "Survived": Y_pred

    })

## submission.to_csv('../output/submission.csv', index=False)