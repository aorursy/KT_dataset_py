# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sklearn as sk

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from mlxtend.classifier import EnsembleVoteClassifier

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix

from sklearn.pipeline import Pipeline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("../input/train.csv")

dataset.head()
dataset.info()
del dataset["PassengerId"]
del dataset["Name"]
del dataset["Ticket"]

del dataset["Cabin"]

dataset.info()
dataset["Age"] = dataset["Age"].fillna(np.mean(dataset["Age"]))
dataset = dataset.dropna()

dataset = pd.get_dummies(dataset,columns=["Sex", "Embarked"])
dataset = pd.get_dummies(dataset,columns=["Pclass"])
sb.heatmap(dataset.corr())
dataset.shape
X = dataset.iloc[:,1:].values

y = dataset.iloc[:,0].values

sd = StandardScaler()

X = sd.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 100)
test_data = pd.read_csv("../input/test.csv")

test_data.shape, dataset.shape

test_data.head()
outdata = test_data["PassengerId"]

del test_data["PassengerId"]

del test_data["Name"]

del test_data["Ticket"]
del test_data["Cabin"]
test_data["Age"] = test_data["Age"].fillna(test_data["Age"].mean())

test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].mean())
test_data.info()
test_data = pd.get_dummies(test_data, columns=["Sex","Embarked", "Pclass"])
lr = MLPClassifier(hidden_layer_sizes=(20))

lr.fit(X_train, y_train)

lr.score(X_test,y_test)
estimators = [

    ("Logistic Regression", LogisticRegression(random_state = 0, C = 1)),

    ("Decision Tree", DecisionTreeClassifier(max_depth = 5)),

    ("Random Forest", RandomForestClassifier(max_depth=3)),

    ("SVM-linear", SVC(C = 1, gamma=0.1, kernel= "linear",probability=True)),

    ("SVM-rbf-gamma1", SVC(C = 1, gamma=1, kernel= "rbf",probability=True)),

    ("SVM-rbf-gamma10", SVC(C = 1, gamma=10, kernel= "rbf",probability=True)),

    ("Neural Network", MLPClassifier(alpha = 0.01, max_iter = 10000, random_state=100, 

                                     activation = "relu", hidden_layer_sizes= (20))),

    ("KNN Classifier", KNeighborsClassifier(n_neighbors= 5))

]



estimators.append(("Ensemble", EnsembleVoteClassifier(clfs = [p[1] for p in estimators])))

i=0

while i < len(estimators):

    model = estimators[i][1]

    model.fit(X_train, y_train)

    y_test_pred = model.predict(X_test)

    y_test_prob = model.predict_proba(X_test)

    roc_auc = roc_auc_score(y_test, y_test_pred)

    accuracy = accuracy_score(y_test, y_test_pred)

    auc_scores = cross_val_score(cv = KFold(n_splits=5), estimator=model,

                                     scoring = "accuracy", X = X, y = y)

    print("%s [AUC: %.2f, Accuracy: %.2f +/- %.2f ]" % (estimators[i][0], 

                                                                  np.mean(auc_scores), accuracy, np.std(auc_scores)))

    print(confusion_matrix(y_test_pred, y_test), auc_scores)

    

    fpr, tpr, threshold = roc_curve(y_test, y_test_prob[:, 1])

    

    plt.title('Receiver Operating Characteristic')

    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)

    plt.legend(loc = 'lower right')

    plt.plot([0, 1], [0, 1],'r--')

    plt.xlim([0, 1])

    plt.ylim([0, 1])

    plt.ylabel('True Positive Rate')

    plt.xlabel('False Positive Rate')

    plt.show()

    i += 1
estimators = [

    ("LogisticRegression", LogisticRegression(random_state = 0, C = 1)),

    ("DecisionTree", DecisionTreeClassifier(random_state=0)),

    ("RandomForest", RandomForestClassifier(max_depth=3)),

    ("SVM", SVC(C = 1, gamma=0.1, kernel= "linear",probability=True)),

    ("SVMrbfgamma1", SVC(C = 1, gamma=1, kernel= "rbf",probability=True)),

    ("SVMrbfgamma10", SVC(C = 1, gamma=10, kernel= "rbf",probability=True)),

    ("NeuralNetwork", MLPClassifier(alpha = 0.01, max_iter = 10000, random_state=100, 

                                     activation = "relu", hidden_layer_sizes= (20))),

    ("KNNClassifier", KNeighborsClassifier(n_neighbors= 5))

]

c_range = 10.0 ** np.arange(-4,4)

pipes =[

    Pipeline([

    ("LogisticRegression", LogisticRegression(random_state = 0, C = 1))]),

    Pipeline([

    ("DecisionTree", DecisionTreeClassifier(random_state=0))]),

    Pipeline([

    ("RandomForest", RandomForestClassifier(max_depth=3))]),

    Pipeline([

    ("SVM", SVC(C = 1, gamma=0.1, kernel= "linear",probability=True))]),

    Pipeline([

    ("NeuralNetwork", MLPClassifier(alpha = 0.01, max_iter = 10000, random_state=100, 

                                     activation = "relu", hidden_layer_sizes= (20)))]),

    Pipeline([

    ("KNNClassifier", KNeighborsClassifier(n_neighbors= 5))])

       ]



parameters = [

    { "LogisticRegression__C" :  c_range},

    { "DecisionTree__max_depth" : np.arange(1,5) },

    { "RandomForest__max_depth" : np.arange(1,5) },

    { "SVM__C" : c_range, "SVM__gamma" : c_range, "SVM__kernel" : ["rbf", "linear"]},

    { "NeuralNetwork__hidden_layer_sizes" : (np.arange(5,50))},

    { "KNNClassifier__n_neighbors" : np.arange(1,10) } 

]



i = 0

while i< len(parameters):



    gs = GridSearchCV(estimator = pipes[i], 

                     param_grid = parameters[i],

                     scoring = "accuracy",

                     cv = 10)



    gs.fit(X, y)

    print("values of: %d, Best params: %s Best score: %.4f" % ( i+1, gs.best_params_, gs.best_score_))

    i += 1





