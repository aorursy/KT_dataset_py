# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pickle

import tensorflow as tf

import math

import os

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/titanic/train.csv")

test = pd.read_csv("../input/titanic/test.csv")



train.describe(include='all')
print(train.columns)
print(pd.isnull(train).sum())
train.sample(5)
sns.barplot(x="Sex", y = "Survived", data=train)

print("Percentage of females who survived:", train["Survived"][train["Sex"] == 'female'].value_counts(normalize = True)[1]*100)



print("Percentage of males who survived:", train["Survived"][train["Sex"] == 'male'].value_counts(normalize = True)[1]*100)
sns.barplot(x="Pclass", y="Survived", data=train)



#print percentage of people by Pclass that survived

print("Percentage of Pclass = 1 who survived:", train["Survived"][train["Pclass"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 2 who survived:", train["Survived"][train["Pclass"] == 2].value_counts(normalize = True)[1]*100)



print("Percentage of Pclass = 3 who survived:", train["Survived"][train["Pclass"] == 3].value_counts(normalize = True)[1]*100)
sns.barplot(x="SibSp", y="Survived", data=train)



#I won't be printing individual percent values for all of these.

print("Percentage of SibSp = 0 who survived:", train["Survived"][train["SibSp"] == 0].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 1 who survived:", train["Survived"][train["SibSp"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of SibSp = 2 who survived:", train["Survived"][train["SibSp"] == 2].value_counts(normalize = True)[1]*100)
sns.barplot(x="Parch", y="Survived", data=train)

plt.show()
#sort the ages into logical categories

train["Age"] = train["Age"].fillna(-0.5)

test["Age"] = test["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

train['AgeGroup'] = pd.cut(train["Age"], bins, labels = labels)

test['AgeGroup'] = pd.cut(test["Age"], bins, labels = labels)



#draw a bar plot of Age vs. survival

sns.barplot(x="AgeGroup", y="Survived", data=train)

plt.show()
train["CabinBool"] = (train["Cabin"].notnull().astype('int'))

test["CabinBool"] = (test["Cabin"].notnull().astype('int'))



#calculate percentages of CabinBool vs. survived

print("Percentage of CabinBool = 1 who survived:", train["Survived"][train["CabinBool"] == 1].value_counts(normalize = True)[1]*100)



print("Percentage of CabinBool = 0 who survived:", train["Survived"][train["CabinBool"] == 0].value_counts(normalize = True)[1]*100)

#draw a bar plot of CabinBool vs. survival

sns.barplot(x="CabinBool", y="Survived", data=train)

plt.show()
train = train.drop(['Cabin'], axis = 1)

test = test.drop(['Cabin'], axis = 1)

train = train.drop(['Ticket'], axis = 1)

test = test.drop(['Ticket'], axis = 1)
#now we need to fill in the missing values in the Embarked feature

print("Number of people embarking in Southampton (S):")

southampton = train[train["Embarked"] == "S"].shape[0]

print(southampton)



print("Number of people embarking in Cherbourg (C):")

cherbourg = train[train["Embarked"] == "C"].shape[0]

print(cherbourg)



print("Number of people embarking in Queenstown (Q):")

queenstown = train[train["Embarked"] == "Q"].shape[0]

print(queenstown)
train = train.fillna({"Embarked": "S"})
#create a combined group of both datasets

combine = [train, test]



#extract a title for each Name in the train and test datasets

for dataset in combine:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(train['Title'], train['Sex'])
for dataset in combine:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',

    'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')

    

    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



train[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#map each of the title groups to a numerical value

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6}

for dataset in combine:

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)



train.head()
# fill missing age with mode age group for each title

mr_age = train[train["Title"] == 1]["AgeGroup"].mode() #Young Adult

miss_age = train[train["Title"] == 2]["AgeGroup"].mode() #Student

mrs_age = train[train["Title"] == 3]["AgeGroup"].mode() #Adult

master_age = train[train["Title"] == 4]["AgeGroup"].mode() #Baby

royal_age = train[train["Title"] == 5]["AgeGroup"].mode() #Adult

rare_age = train[train["Title"] == 6]["AgeGroup"].mode() #Adult
age_title_mapping = {1: "Young Adult", 2: "Student", 3: "Adult", 4: "Baby", 5: "Adult", 6: "Adult"}



#I tried to get this code to work with using .map(), but couldn't.

#I've put down a less elegant, temporary solution for now.

#train = train.fillna({"Age": train["Title"].map(age_title_mapping)})

#test = test.fillna({"Age": test["Title"].map(age_title_mapping)})



for x in range(len(train["AgeGroup"])):

    if train["AgeGroup"][x] == "Unknown":

        train["AgeGroup"][x] = age_title_mapping[train["Title"][x]]

        print(train["AgeGroup"][x])

        

for x in range(len(test["AgeGroup"])):

    if test["AgeGroup"][x] == "Unknown":

        test["AgeGroup"][x] = age_title_mapping[test["Title"][x]]
age_mapping = {'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7}

train['AgeGroup'] = train['AgeGroup'].map(age_mapping)

test['AgeGroup'] = test['AgeGroup'].map(age_mapping)



train.head()



#dropping the Age feature for now, might change

train = train.drop(['Age'], axis = 1)

test = test.drop(['Age'], axis = 1)
#drop the name feature since it contains no more useful information.

train = train.drop(['Name'], axis = 1)

test = test.drop(['Name'], axis = 1)
#map each Sex value to a numerical value

sex_mapping = {"male": 0, "female": 1}

train['Sex'] = train['Sex'].map(sex_mapping)

test['Sex'] = test['Sex'].map(sex_mapping)



train.head()
embarked_mapping = {"S": 1, "C": 2, "Q": 3}

train['Embarked'] = train['Embarked'].map(embarked_mapping)

test['Embarked'] = test['Embarked'].map(embarked_mapping)



train.head()
#fill in missing Fare value in test set based on mean fare for that Pclass 

for x in range(len(test["Fare"])):

    if pd.isnull(test["Fare"][x]):

        pclass = test["Pclass"][x] #Pclass = 3

        test["Fare"][x] = round(train[train["Pclass"] == pclass]["Fare"].mean(), 4)

        

#map Fare values into groups of numerical values

train['FareBand'] = pd.qcut(train['Fare'], 4, labels = [1, 2, 3, 4])

test['FareBand'] = pd.qcut(test['Fare'], 4, labels = [1, 2, 3, 4])



#drop Fare values

train = train.drop(['Fare'], axis = 1)

test = test.drop(['Fare'], axis = 1)
train.head()
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA



predictors = train.drop(['Survived', 'PassengerId'], axis=1)

pca = PCA(n_components=8)

predictors_pca = pca.fit_transform(predictors)

target = train["Survived"]

x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
# Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



gaussian = GaussianNB()

gaussian.fit(x_train, y_train)

y_pred = gaussian.predict(x_val)

acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gaussian)
# Logistic Regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression(penalty='l2')

logreg.fit(x_train, y_train)

y_pred = logreg.predict(x_val)

acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_logreg)
# Support Vector Machines

from sklearn.svm import SVC



svc = SVC(kernel='rbf', C=100, random_state=0)

svc.fit(x_train, y_train)

y_pred = svc.predict(x_val)

acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_svc)
# Linear SVC

from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train, y_train)

y_pred = linear_svc.predict(x_val)

acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_linear_svc)
# Perceptron

from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train, y_train)

y_pred = perceptron.predict(x_val)

acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_perceptron)
#Decision Tree

from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train, y_train)

y_pred = decisiontree.predict(x_val)

acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_decisiontree)
# Random Forest

from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train, y_train)

y_pred = randomforest.predict(x_val)

acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_randomforest)
# KNN or k-Nearest Neighbors

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

y_pred = knn.predict(x_val)

acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_knn)
# Stochastic Gradient Descent

from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train, y_train)

y_pred = sgd.predict(x_val)

acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_sgd)
# Gradient Boosting Classifier

from sklearn.ensemble import GradientBoostingClassifier



gbk = GradientBoostingClassifier()

gbk.fit(x_train, y_train)

y_pred = gbk.predict(x_val)

acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)

print(acc_gbk)
test_x = test.drop('PassengerId', axis=1)
batch_size = 512

l_rate = 0.001

init = tf.initializers.truncated_normal(seed=123, stddev=0.1)

input_data = tf.placeholder(tf.float32, [None, 9], name="input_data")

labels = tf.placeholder(tf.float32, shape=[None, 2], name="labels")

drop = tf.placeholder(tf.float32, shape=None, name = "drop_out")



def make_mlp(data):

    net = tf.layers.dense(data, 64, kernel_initializer=init, activation=None)

    net = tf.nn.relu(net)

    net = tf.nn.dropout(net, drop, seed=123)

#     net = tf.layers.dense(data, 16, kernel_initializer=init, activation=None)

#     net = tf.nn.relu(net)

#     net = tf.nn.dropout(net, drop, seed=123)

#     net = tf.layers.dense(data, 16, kernel_initializer=init, activation=None)

#     net = tf.nn.relu(net)

#     net = tf.nn.dropout(net, drop, seed=123)

    return tf.layers.dense(net, 2, kernel_initializer=init, activation=None)



lambda_l2_reg = 0.000001

output_net = make_mlp(input_data)

l2 = lambda_l2_reg * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables() if not ("noreg" in tf_var.name or "Bias" in tf_var.name))

hypothesis = tf.nn.softmax(output_net)

cost = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(hypothesis), reduction_indices=1)) + l2

optimizer = tf.train.AdamOptimizer(l_rate)

train_op = optimizer.minimize(cost)

correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(labels, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction = tf.argmax(hypothesis, 1)
def batch(input_data, labels, batch_size):

    n_batch = int(math.ceil(len(input_data) / batch_size))

    index = 0

    for _ in range(n_batch):

        batch_input_data = np.array(input_data[index: index + batch_size])

        batch_labels = np.array(labels[index: index + batch_size])



        index += batch_size

        yield batch_input_data, batch_labels
# x_train, x_val, y_train, y_val

config = tf.ConfigProto(allow_soft_placement=True)

config.gpu_options.allow_growth = True

temp = [0, 10, 0]

final_test = []

with tf.Session(config=config) as session:

    session.run(tf.global_variables_initializer())

    y_train = tf.one_hot(y_train, depth=2).eval(session=session)

    y_val = tf.one_hot(y_val, depth=2).eval(session=session)

    for i in range(1000):

        for batch_input_data, batch_labels in batch(x_train, y_train, batch_size):

            train_loss, _ = session.run([cost, train_op],

                                feed_dict={input_data: batch_input_data,

                                            labels: batch_labels,

                                          drop: 0.75})

        val_loss, val_accuracy = session.run([cost, accuracy],

                            feed_dict={input_data: x_val,

                                       labels: y_val,

                                      drop: 1.0})

        test_predictions = session.run(prediction,

                            feed_dict={input_data: test_x,

                                      drop: 1.0})

        if val_loss < temp[1]:

            temp = [i, val_loss, val_accuracy]

            final_test = test_predictions

            print("max_acc: ", temp)
acc_mlp = round(temp[-1] * 100, 2)

print(acc_mlp)
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 

              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier', 'Multi Layer Perceptron'],

    'Score': [acc_svc, acc_knn, acc_logreg, 

              acc_randomforest, acc_gaussian, acc_perceptron,acc_linear_svc, acc_decisiontree,

              acc_sgd, acc_gbk, acc_mlp]})

models.sort_values(by='Score', ascending=False)
# #set ids as PassengerId and predict survival 

# ids = test['PassengerId']

# predictions = gbk.predict(test.drop('PassengerId', axis=1))



# #set the output as a dataframe and convert to csv file named submission.csv

# output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

# output.to_csv('submission.csv', index=False)
ids = test['PassengerId']



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': final_test })

output.to_csv('submission.csv', index=False)