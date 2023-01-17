import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap



from sklearn.metrics import confusion_matrix,accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import normalize

from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score, log_loss

from sklearn import tree

from sklearn import linear_model

from sklearn import svm

from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import AdaBoostClassifier

import sys,os





import seaborn as sns

%matplotlib inline



training_data = pd.read_csv('../input/train.csv')

training_data.sample(5)



training_data.describe()
def simplify_ages(df):

    df.Age = df.Age.fillna(-0.5)

    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)

    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

    categories = pd.cut(df.Age, bins, labels=group_names)

    df.Age = categories

    return df



def simplify_cabins(df):

    df.Cabin = df.Cabin.fillna('N')

    df.Cabin = df.Cabin.apply(lambda x: x[0])

    return df



def simplify_fares(df):

    df.Fare = df.Fare.fillna(-0.5)

    bins = (-1, 0, 8, 15, 31, 1000)

    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']

    categories = pd.cut(df.Fare, bins, labels=group_names)

    df.Fare = categories

    return df



def format_name(df):

    df['Lname'] = df.Name.apply(lambda x: x.split(' ')[0])

    df['NamePrefix'] = df.Name.apply(lambda x: x.split(' ')[1])

    return df    

    

def drop_features(df):

    return df.drop(['Ticket', 'Name', 'Embarked'], axis=1)



def transform_features(df):

    df = simplify_ages(df)

    df = simplify_cabins(df)

    df = simplify_fares(df)

    df = format_name(df)

    df = drop_features(df)

    return df



transformed_train = transform_features(training_data)

transformed_train.head()
test_data = pd.read_csv('../input/test.csv')

test_data.sample(5)
transformed_test = transform_features(test_data)

transformed_test.head()
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Fare', 'Cabin', 'Age', 'Sex', 'Lname', 'NamePrefix']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(transformed_train, transformed_test)

data_train.head()
data_test.head()
sns.pairplot(data_train)
sns.pairplot(data_train,hue='Survived')
X = data_train.drop(['Survived'], axis=1)

Y = data_train.Survived

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)

print(X_train.shape)

print(X_test.shape)

print(Y_train.shape)

print(Y_test.shape)
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train, Y_train)

Y_pred =  dtree.predict(X_test)

print(confusion_matrix(Y_test,Y_pred))

print(accuracy_score(Y_test,Y_pred))
training_data.head()
t1 = sns.heatmap(data_train[["Survived","Sex","Pclass","SibSp","Parch","Age","Fare","NamePrefix"]].corr(),annot=True, fmt = ".2f", cmap = "coolwarm")
data_train.info()
data_train.describe()
training_data.info()
training_data.describe()
sns.barplot(x="Age", y="Survived", hue="Age", data=data_train);
sns.barplot(x="Sex", y="Survived", hue="Sex", data=data_train);
sns.barplot(x="SibSp", y="Survived", hue="SibSp", data=data_train);
sns.barplot(x="Pclass", y="Survived", hue="Pclass", data=data_train);
sns.barplot(x="Parch", y="Survived", hue="Parch", data=data_train);
sns.barplot(x="Fare", y="Survived", hue="Fare", data=data_train);
sns.barplot(x="Embarked", y="Survived", hue="Sex", data=training_data);
sns.pointplot(x="Pclass", y="Survived", hue="Sex", data=training_data,

              palette={"male": "blue", "female": "pink"},

              markers=["*", "o"], linestyles=["-", "--"]);

test_data = pd.read_csv('../input/test.csv')

test_data.sample(5)

transformed_test = transform_features(test_data)

transformed_test.head()

sns.barplot(x="Age", y="Survived", hue="Sex", data=transformed_train);



classifiers = {}

clf = MLPClassifier()

clf.set_params(hidden_layer_sizes =(100,100), max_iter = 1000,alpha = 0.01, momentum = 0.7)

nn_clf = clf.fit(X_train,Y_train)

nn_predict = nn_clf.predict(X_test)

nn_acc = accuracy_score(Y_test,nn_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Artificial Nueral Network:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["NN"]=clf

clf = linear_model.Perceptron()

clf.set_params(max_iter = 1000,alpha = 0.0001)

pt_clf = clf.fit(X_train,Y_train)

pt_predict = pt_clf.predict(X_test)

pt_acc = accuracy_score(Y_test,pt_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Perceptron:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["PT"]=clf
clf = MLPClassifier()

clf.set_params(hidden_layer_sizes =(100,100,100,100), max_iter = 100,alpha = 0.3, momentum = 0.7,activation = "relu")

nn_clf = clf.fit(X_train,Y_train)

nn_predict = nn_clf.predict(X_test)

nn_acc = accuracy_score(Y_test,nn_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Deep Neural Network:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["DNN"]=clf

clf = svm.SVC()

clf.set_params(C = 100, kernel = "rbf")

svm_clf = clf.fit(X_train,Y_train)

svm_predict = svm_clf.predict(X_test)

svm_acc = accuracy_score(Y_test,svm_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Support Vector Machines:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["SVM"]=clf

clf = MultinomialNB()

clf.set_params(alpha = 0.1)

nb_clf = clf.fit(X_train,Y_train)

nb_predict = nb_clf.predict(X_test)

nb_acc = accuracy_score(Y_test,nb_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Multinomial Naive Bayes:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["NB"]=clf

clf = LogisticRegression()

clf.set_params(C = 10, max_iter = 10)

lr_clf = clf.fit(X_train,Y_train)

lr_predict = lr_clf.predict(X_test)

lr_acc = accuracy_score(Y_test,lr_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Logistic Regression:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["LR"]=clf

clf = KNeighborsClassifier()

clf.set_params(n_neighbors= 5,leaf_size = 30)

knn_clf = clf.fit(X_train,Y_train)

knn_predict = knn_clf.predict(X_test)

knn_acc = accuracy_score(Y_test,knn_predict)

param =  knn_clf.get_params()

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("k-NN :")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["KNN"]=clf

clf = RandomForestClassifier()

clf.set_params(n_estimators = 100, max_depth = 10)

rf_clf = clf.fit(X_train,Y_train)

rf_predict = rf_clf.predict(X_test)

rf_acc = accuracy_score(Y_test,rf_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("Random Forest Classifier:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["RF"]=clf

clf = AdaBoostClassifier()

clf.set_params(n_estimators = 10, learning_rate = 1)

ada_clf = clf.fit(X_train,Y_train)

ada_predict = ada_clf.predict(X_test)

ada_acc = accuracy_score(Y_test,ada_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("AdaBoost:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["ADA"]=clf

clf = GradientBoostingClassifier()

clf.set_params(n_estimators = 30,learning_rate = 1)

gb_clf = clf.fit(X_train,Y_train)

gb_predict = gb_clf.predict(X_test)

gb_acc = accuracy_score(Y_test,gb_predict)

accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

print("GradientBoostingClassifier:")

print (accuracy.mean(), " - ",f_score.mean())

classifiers["GB"]=clf

print ("accuracy","              ","F-score")

for clf in classifiers.values():

    accuracy = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'accuracy')

    f_score = cross_val_score(clf, X_train , Y_train, cv=10,scoring = 'f1_micro')

    for i in classifiers:

        if classifiers[i]== clf:

            print (i),

            break

    print ( " : ",accuracy.mean(), "  ",f_score.mean())

    