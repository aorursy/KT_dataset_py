import numpy as np

import sklearn

from sklearn import model_selection

from sklearn.model_selection import cross_validate

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns
data = pd.read_csv("../input/heart.csv")
# Preprocess the data

data.replace('?',-99999, inplace=True)

print(data.axes)



print(data.columns)
#visualize and explore the data

print(data.loc[10])



# Print the shape of the dataset

print(data.shape)
#describing the data

print(data.describe())
#Plotting the data

data.hist(figsize=(15,15))

plt.show()
#scattering the plot

scatter_matrix(data, figsize=(20,20))

plt.show()
# Correlation matrix

corrmat = data.corr()

plt.figure(figsize=(15,15))

sns.heatmap(corrmat,cmap='viridis',annot=True,linewidths=0.5,)
# Get all the columns from the dataFrame

columns = data.columns.tolist()



# Filter the columns to remove data we do not want

columns = [c for c in columns if c not in ["target", "chol", "fbs", "restecg", "testbps"]]



# Store the variable we'll be predicting on

target = "target"



X = data[columns]

y = data[target]



# Print shapes

print(X.shape)

print(y.shape)
print(X.loc[26])

print(y.loc[26])
#Creating X and y datasets for training

#from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y, test_size = 0.2)
#Specify the testing option



scoring = 'accuracy'



print(X_train.shape, X_test.shape)

print(y_train.shape, y_test.shape)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score
# Define models to train





models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 3)))

models.append(('NaiveB', GaussianNB()))

models.append(('CART', DecisionTreeClassifier(max_depth=5)))

models.append(('ADA', AdaBoostClassifier()))

models.append(('RFC', RandomForestClassifier(max_depth=10, n_estimators=40)))
# evaluate each model in turn

results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=5)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
from sklearn.model_selection import StratifiedKFold

from yellowbrick.model_selection import CVScores



_, ax = plt.subplots()



cv = StratifiedKFold(5)



oz = CVScores(RandomForestClassifier(max_depth=10, n_estimators = 40), ax=ax, cv=cv, scoring= 'accuracy')

oz.fit(X,y)

oz.poof()
# Compare Algorithms

fig = plt.figure(figsize = (18,18))

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results, widths = 0.6)

ax.set_xticklabels(names)

plt.show()
from sklearn.ensemble import VotingClassifier



ensemble = VotingClassifier(estimators = models, voting = 'hard', n_jobs = -1)



ensemble.fit(X_train, y_train)



predictions = ensemble.score(X_test, y_test)*100





print("The Voting Classifier Accuracy is: ", predictions)
for name, model in models:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name)

    print(accuracy_score(y_test, predictions)*100)

    print(classification_report(y_test, predictions))
from sklearn.metrics import  confusion_matrix

predict = model.predict(X_test)

print("=== Confusion Matrix ===")

print(confusion_matrix(y_test, predict))

print('\n')



from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, predict)

p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

plt.title('Confusion matrix', y=1.1)

plt.ylabel('Actual label')

plt.xlabel('Predicted label')
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
estimator = model

feature_names = [i for i in X_train.columns]



y_train_str = y_train.astype('str')

y_train_str[y_train_str == '0'] = 'no Disease'

y_train_str[y_train_str == '1'] = 'Disease'

y_train_str = y_train_str.values
from sklearn.tree import export_graphviz #plot tree

import graphviz

export_graphviz(estimator, out_file='tree2.dot', 

                feature_names = feature_names,

                class_names = y_train_str,

                rounded = True, proportion = True, 

                label='root',

                precision = 2, filled = True)

with open("tree2.dot") as f:

    dot_graph = f.read()

graphviz.Source(dot_graph)