# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from collections import Counter



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import warnings

warnings.filterwarnings("ignore")

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("/kaggle/input/heart-disease-dataset/heart.csv")
data.head()
data.info()
column_list = data.columns

for i in column_list:

    print("Values of",i,"column\n",data[i].unique())

    print("--------------\n")
g = sns.FacetGrid(data, col = "target")

g.map(sns.distplot, "age", bins = 25)

plt.show()
# Let's check the relationship of each column with heart disease

# we cant visualize each column, that would be meaningless so I'll create new list.

column_list2 = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]



g = sns.factorplot(x = column_list2[0], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[0])

plt.show()
g = sns.factorplot(x = column_list2[1], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[1])

plt.show()
g = sns.factorplot(x = column_list2[2], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[2])

plt.show()
g = sns.factorplot(x = column_list2[3], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[3])

plt.show()
g = sns.factorplot(x = column_list2[4], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[4])

plt.show()
g = sns.factorplot(x = column_list2[5], y = "target", data = data, kind = "bar",size=5)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[5])

plt.show()
g = sns.factorplot(x = column_list2[6], y = "target", data = data, kind = "bar",size=7)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[6])

plt.show()
g = sns.factorplot(x = column_list2[7], y = "target", data = data, kind = "bar",size=6)

g.set_ylabels("Heart Disase Probability")

g.set_xlabels(column_list2[7])

plt.show()
plt.figure(figsize=(15,10))

sns.heatmap(data[column_list2].corr(), annot = True, fmt = ".2f")

plt.show()
dummy_list = ["sex","cp","restecg","exang","slope","thal"]

data = pd.get_dummies(data,columns=dummy_list)

data.head()
# Import Machine Learning Libraries



from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
y = data.target.values

x_data = data.drop(["target"],axis=1)
# big values can be dominated low values so we use normalization method 

x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 42)

print("x_train",len(x_train))

print("x_test",len(x_test))

print("y_train",len(y_train))

print("y_test",len(y_test))
random_state = 42

classifier = [KNeighborsClassifier(),

              SVC(random_state = random_state,probability=True),

              DecisionTreeClassifier(random_state = random_state),

             RandomForestClassifier(random_state = random_state),

             LogisticRegression(random_state = random_state),

             ]



knn_param_grid = {"n_neighbors": np.linspace(1,19,10, dtype = int).tolist(),

                 "weights": ["uniform","distance"],

                 "metric":["euclidean","manhattan"]}



svm_param_grid = {"kernel" : ["rbf"],

                 "gamma": [0.001, 0.01, 0.1, 1],

                 "C": [1,10,50,100,200,300,1000]}



dt_param_grid = {"min_samples_split" : range(10,500,20),

                "max_depth": range(1,20,2)}





rf_param_grid = {"max_features": [1,3,10],

                "min_samples_split":[2,3,10],

                "min_samples_leaf":[1,3,10],

                "bootstrap":[False],

                "n_estimators":[100,300],

                "criterion":["gini"]}



logreg_param_grid = {"C":np.logspace(-3,3,7),

                    "penalty": ["l1","l2"]}





classifier_param = [knn_param_grid,

                   svm_param_grid,

                    dt_param_grid,

                   rf_param_grid,

                   logreg_param_grid,

                   knn_param_grid]
cv_result = []

best_estimators = []

for i in range(len(classifier)):

    clf = GridSearchCV(classifier[i], param_grid=classifier_param[i], cv = StratifiedKFold(n_splits = 10), scoring = "accuracy", n_jobs = -1,verbose = 1)

    clf.fit(x_train,y_train)

    cv_result.append(clf.best_score_ * 100)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":[ "KNeighborsClassifier", "SVM","Decision Tree Classifier",

             "Random Forest Classifier","LogisticRegression",

            ]})



g = sns.barplot("Cross Validation Means", "ML Models", data = cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Scores")
# We must detect the best k value of knn model so we will determine best_estimators list

best_estimators
# We must find predicted values of each models. After that, we will compare with real values.

knn9 = KNeighborsClassifier(n_neighbors = 9)

knn9.fit(x_train, y_train)

y_head_knn = knn9.predict(x_test)



svm = SVC(random_state = 1)

svm.fit(x_train,y_train)

y_head_svm = svm.predict(x_test)



rf = RandomForestClassifier(n_estimators = 500, random_state = 1)

rf.fit(x_train,y_train)

y_head_rf = rf.predict(x_test)
# We find confusion matrix of models below.

cm_knn = confusion_matrix(y_test,y_head_knn)

cm_svm = confusion_matrix(y_test,y_head_svm)

cm_rf = confusion_matrix(y_test,y_head_rf)



# Let's visualize them

plt.figure(figsize=(12,6))

plt.suptitle("Confusion Matrices",fontsize=24, color="red")

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(1,3,1)

plt.title("K Nearest Neighbors Confusion Matrix",fontsize=10,color="blue")

sns.heatmap(cm_knn,annot=True, cmap="YlGnBu",fmt="d",cbar=False, annot_kws={"size": 18})



plt.subplot(1,3,2)

plt.title("Support Vector Machine Confusion Matrix",fontsize=10,color="blue")

sns.heatmap(cm_svm,annot=True, cmap="YlGnBu",fmt="d",cbar=False, annot_kws={"size": 18})



plt.subplot(1,3,3)

plt.title("Random Forest Confusion Matrix",fontsize=10,color="blue")

sns.heatmap(cm_svm,annot=True, cmap="YlGnBu",fmt="d",cbar=False, annot_kws={"size": 18})



plt.show()

votingC = VotingClassifier(estimators = [("knn",best_estimators[0]),

                                        ("svm",best_estimators[1]),

                                        ("rf",best_estimators[3])],

                                        voting = "soft", n_jobs = -1)

votingC = votingC.fit(x_train, y_train)

print("Accuracy of Ensemble: {:.2f}".format(accuracy_score(votingC.predict(x_test),y_test)*100))
best_accuracies_each_classes = {}

lr = LogisticRegression()

lr.fit(x_train, y_train)

accuracy_lr_train = round(lr.score(x_train, y_train)*100,2) 

accuracy_lr_test = round(lr.score(x_test,y_test)*100,2)

best_accuracies_each_classes["Logistic Regression"] = lr.score(x_test,y_test)*100

print("Training Accuracy: {}%".format(accuracy_lr_train))

print("Testing Accuracy: {}%".format(accuracy_lr_test))



score_list_test = []

for i in range(1,21):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train,y_train)

    score_list_test.append(knn.score(x_test,y_test))

    

best_accuracies_each_classes["KNN"] = max(score_list_test)*100

print("Best Test KNN Score accuracy is: {:.2f}".format(max(score_list_test)*100))



plt.figure(figsize=(15,5))

plt.plot(range(1,21),score_list_test)

plt.xlabel("K Values")

plt.ylabel("Accuracy")

plt.show()

svm = SVC(random_state = 1)

svm.fit(x_train,y_train)



best_accuracies_each_classes["SVM"] = svm.score(x_test,y_test)*100

print("Accuracy of SVM Algo: {:.2f}".format(svm.score(x_test,y_test)*100))
nb = GaussianNB()

nb.fit(x_train,y_train)



best_accuracies_each_classes["Naive Bayes"] = nb.score(x_test,y_test)*100

print("Accuracy of Naive Bayes: {:.2f}".format(nb.score(x_test,y_test)*100))
dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)



best_accuracies_each_classes["Decision Tree"] = dt.score(x_test,y_test)*100

print("Accuracy of Decision Tree: {:.2f}".format(dt.score(x_test,y_test)*100))


rf = RandomForestClassifier(n_estimators = 500, random_state = 1)

rf.fit(x_train,y_train)



best_accuracies_each_classes["Random Forest"] = rf.score(x_test,y_test)*100

print("Accuracy of  Random Forest is: {:.2f}".format(rf.score(x_test,y_test)*100))
plt.figure(figsize=(8,5))

sns.barplot( y=list(best_accuracies_each_classes.keys()), x=list(best_accuracies_each_classes.values()))

plt.xlabel("Accuracy")

plt.ylabel("Classification Methods")

plt.show()
# I did not sort models ın an organized way so I filled by manually according to first graphic's sort.



cv_results = pd.DataFrame({"Cross Validation Means":cv_result, "ML Models":[ "KNeighborsClassifier", "SVM","Decision Tree Classifier",

             "Random Forest Classifier","LogisticRegression",

            ]})



plt.figure(figsize=(18,6))

plt.suptitle("Comparing Models",fontsize=24, color="red")

plt.subplots_adjust(wspace = 0.8, hspace= 0.4)



plt.subplot(1,2,1)

plt.title("With Parameter Grid",fontsize=10,color="blue")

sns.barplot("Cross Validation Means", "ML Models", data = cv_results)





plt.subplot(1,2,2)

plt.title("Without Parameter Grid",fontsize=10,color="blue")

sns.barplot( y=['KNeighborsClassifier',

                 'SVM',

                 'Decision Tree Classifier',

                 'Random Forest Classifier',

                 'LogisticRegression',

                'Naive Bayes'] , x=[99.02597402597402,87.01298701298701,  98.05194805194806,98.05194805194806,80.84415584415584, 71.1038961038961])

plt.xlabel("Accuracy")

plt.ylabel("Classification Methods")

plt.show()