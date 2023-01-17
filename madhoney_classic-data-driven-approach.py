import warnings



import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import sklearn

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import Binarizer

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier



warnings.filterwarnings("ignore")

sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = 12,8
train_df = pd.read_csv("../input/train.csv")

train_df.head()
train_df.info()
train_df.describe()
train_df.drop(["Cabin","Ticket","Name"], axis=1, inplace=True)
to_numb = {

    "S":0,

    "Q":1,

    "C":2,

    np.nan:0

}

train_df["Embarked"] = train_df["Embarked"].map(lambda x: to_numb[x])

train_df["Sex"] = LabelEncoder().fit_transform(train_df["Sex"])
class History():

    """Class for log changes in data and algorithms"""

    def __init__(self):

        self.value = []

        

    def add(self, message, level=0):

        message = "  "*level + message

        self.value.append(message)

        

    def print_log(self):

        for message in self.value:

            print(message)

class Test:

    """Class for testing different algorithms

        and data on the same test_set"""

    

    def __init__(self):  

        self.best_accuracy = {"acc":0, "desc": "Nothing"}

        self.history = History()

        self.algorithms = []

        

    def update_test_data(self, X_test, y_test, data_desc):

        self.test_data = {}

        X_test = Imputer(strategy="median").fit_transform(X_test)

        self.test_data["X"] = X_test

        self.test_data["y"] = y_test.values.reshape(-1, 1)

        self.test_data["desc"] = data_desc

        self.history.add("Updated test_data to:\n {}".format(data_desc))

        

    def update_train_data(self, X_train, y_train, data_desc):

        X_train = Imputer(strategy="median").fit_transform(X_train)

        self.train_data = {}

        self.train_data["X"] = X_train

        self.train_data["y"] = y_train.values.reshape(-1, 1)

        self.train_data["desc"] = data_desc

        self.history.add("Updated train_data to:\n {}".format(data_desc))

        

    def fit_algorithm(self, algorithm, desc):

        algorithm.fit(self.train_data["X"], self.train_data["y"])

        train_score = algorithm.score(self.train_data["X"], self.train_data["y"])

        test_score = algorithm.score(self.test_data["X"], self.test_data["y"])

        if test_score > self.best_accuracy["acc"]:

            self.best_accuracy["acc"] = test_score

            self.best_accuracy["desc"] = desc

            print("New best_accuracy: {} algo:{}".format(test_score, desc))   

        message = """Fitted algorithm:\t {}\n Train score:\t {}

                     \n Test score:\t {}\n""".format(desc, train_score, test_score)

        self.history.add(message)

    

    def show_history(self):

        self.history.print_log()

        

    def append_algo(self, algorithm, desc):

        self.algorithms.append([algorithm, desc])

        

    def try_all(self):

        for algo in self.algorithms:

            self.fit_algorithm(algo[0], algo[1])

            

    def create_train_test(self, train_df, message, oneDimension = False):

        train_label = train_df["Survived"]

        train_data = train_df.drop("Survived", axis = 1)

        if oneDimension:

            train_data = train_data.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3)

        

        test.update_train_data(X_train, y_train, message)

        test.update_test_data(X_test, y_test, message)

        

        
test = Test()

tree = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3)

forest = RandomForestClassifier(n_estimators=100, min_samples_leaf=3)

svm = SVC(C=0.1)

gradboost = GradientBoostingClassifier(learning_rate = 0.01, n_estimators=100)

sgd = SGDClassifier(alpha=0.001)



test.append_algo(tree, "Decision Tree")

test.append_algo(forest, "Random Forest")

test.append_algo(svm, "SVM")

test.append_algo(gradboost, "Gradient Boosting")

test.append_algo(sgd, "SGD")
sns.heatmap(train_df.corr())

plt.show()
sns.countplot(x="Survived", hue="Sex" , data=train_df)

plt.show()
train_data = train_df[["Sex", "Survived"]]

test.create_train_test(train_data, "Only sex feature")
test.try_all()

test.show_history()
sns.boxplot(y="Age", x="Survived", data=train_df)

plt.show()
train_data = train_df[["Sex", "Age", "Survived"]]

test.create_train_test(train_data, "Age feature added")

test.try_all()

test.show_history()
train_df["isChild"] = train_df["Age"] < 18

train_data = train_df[["Sex", "Age", "isChild", "Survived"]]

test.create_train_test(train_data, "IsChild feature added")
test.try_all()

test.show_history()
sns.countplot(x="Embarked", hue="Survived", data=train_df)

plt.show()
train_data = pd.get_dummies(train_df["Embarked"], prefix="Embarked")[["Embarked_0","Embarked_1"]]

train_data = pd.concat((train_data, train_df[["Sex", "Age", "Survived", "Fare"]]), axis=1)
test.create_train_test(train_data, "Embarked Added")

test.try_all()

test.show_history()
sns.countplot(x="Pclass", hue="Survived", data=train_df)

plt.show()
pclass = pd.get_dummies(train_df["Pclass"], prefix="Pclass")[["Pclass_1", "Pclass_2"]]

train_data = pd.concat((train_data, pclass), axis=1)
test.create_train_test(train_data, "Added Pclass feature")

test.try_all()

test.show_history()
train_data["SibSp"] = train_df["SibSp"]

test.create_train_test(train_data, "Added SibSp")

test.try_all()

test.show_history()
train_data["Age"] = Imputer().fit_transform(train_data["Age"].reshape(-1,1))

train_label = train_data["Survived"]

train_data = train_data.drop("Survived", axis=1)
forest = RandomForestClassifier(n_estimators=200, oob_score=True)

params = {

    "criterion": ["gini", "entropy"],

    "min_samples_leaf": [2, 3, 4, 5],

    "max_features": [2, 3, 4, 5, 6]

}

grid = GridSearchCV(forest, params, cv=5)

grid.fit(train_data,train_label)
test_df = pd.read_csv("../input/test.csv")

test_df.head()
test_df.info()
test_df["Sex"] = LabelEncoder().fit_transform(test_df["Sex"])

test_df["Embarked"] = test_df["Embarked"].map(lambda x: to_numb[x])

test_df["Age"] = Imputer().fit_transform(test_df["Age"].reshape(-1, 1))

test_df["Fare"] = Imputer().fit_transform(test_df["Fare"].reshape(-1, 1))

embarked = pd.get_dummies(test_df["Embarked"], prefix="Emb")[["Emb_1", "Emb_2"]]

pclass = pd.get_dummies(test_df["Pclass"], prefix="Pc")[["Pc_1", "Pc_2"]]

test_data = pd.concat((embarked, test_df[["Sex", "Age", "Fare"]], pclass), axis=1)
X_train, X_test, y_train, y_test = train_test_split(train_data, train_label, test_size=0.3)
model = Sequential()

model.add(Dense(units=64, activation="relu", input_shape=(8,)))

model.add(Dropout(0.2))

model.add(Dense(128, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(256, activation="relu"))

model.add(Dropout(0.3))

model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train,

          y_train,

          batch_size=32,

          epochs=200)
model.evaluate(X_test, y_test, batch_size=32)
model.summary()