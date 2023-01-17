#import libraries

import pandas as pd

import numpy as np

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn import svm

from sklearn import metrics

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
#load the iris data

#url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

iris_dataset = pd.read_csv("../input/Iris.csv")
#is there any missing value in the dataframe

pd.isnull(iris_dataset).sum()
#dimension of the dataset

iris_dataset.shape
#peek at the data

iris_dataset.head()
#del iris_dataset['Id']

iris_dataset.head()
#statistical summary of the dataset

iris_dataset.describe()
#number of instances (rows) that belong to each class

iris_dataset.groupby('Species').size()
#univariate plots to see distribution of the input attributes

iris_dataset.plot(kind='box', subplots=True, layout=(5,5), sharex=False, sharey=False)

plt.show()
#histogram of each input variable to get an idea of the distribution

iris_dataset.hist()

plt.show()
#multivariate plot to spot structured relationships between input variables

scatter_matrix(iris_dataset)

plt.show()
#Create a Validation Dataset

#We will split the loaded dataset into two, 80% of which we will use to train our models 

#and 20% that we will hold back as a validation dataset.

array = iris_dataset.values

X = array[:,0:4]

Y = array[:,4]

Y = Y.astype('int')

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, 

                                                                                random_state=seed)
#Test Harness: 10-fold cross validation to estimate accuracy

#This will split our dataset into 10 parts, train on 9 and test on 1 

#and repeat for all combinations of train-test splits

seed = 7

scoring = 'accuracy'
# Spot Check Algorithms

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))

# evaluate each model in turn

results = []

names = []

for name, model in models:

	kfold = model_selection.KFold(n_splits=10, random_state=seed)

	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

	results.append(cv_results)

	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

	print(msg)
#create a plot of the model evaluation results and 

#compare the spread and the mean accuracy of each model.

fig = plt.figure()

fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
#SVM algorithm is most accurate model

clf=svm.SVC()

clf.fit(X_train,Y_train)

#accuracy of the model on our validation set

y_predict_class=clf.predict(X_validation)

metrics.accuracy_score(Y_validation,y_predict_class)
mat=confusion_matrix(Y_validation,y_predict_class)

cmap=sns.cubehelix_palette(50,hue=0.05,rot=0,light=0.9,dark=0,as_cmap=True)

sns.heatmap(mat,cmap=cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True,fmt="d")

plt.xlabel('Predicted')

plt.ylabel("Actual")

plt.show()
classification_report(Y_validation,y_predict_class)
"""Summary of the Project

1.The dataset contains 150 observations of iris flowers. 

There are four columns of measurements of the flowers in centimeters. 

The fifth column is the species of the flower observed. 

All observed flowers belong to one of three species.



2.We can see that all of the numerical values have the same scale (centimeters) and 

similar ranges between 0 and 8 centimeters.



3.We can see that each class has the same number of instances (50 or 33% of the dataset).



4.It looks like perhaps two of the input variables have a Gaussian distribution. 

This is useful to note as we can use algorithms that can exploit this assumption.



5.Scatterplots suggests a high correlation and a predictable relationship.



6.After splitting the loaded dataset into two, 80% of which we will use to train our models 

and 20% that we will hold back as a validation dataset.

We now have training data in the X_train and Y_train for preparing models and 

a X_validation and Y_validation sets that we can use later.



7.We have uses 10-fold cross validation to estimate accuracy.

This splitted our dataset into 10 parts, train on 9 and test on 1 and repeat for all combinations 

of train-test splits.



8.We are using the metric of ‘accuracy‘ to evaluate models. 

This is a ratio of the number of correctly predicted instances in divided by 

the total number of instances in the dataset multiplied by 100 to give a percentage (e.g. 95% accurate). 

We are using the scoring variable when we run build and evaluate each model next.



9.For evaluating model we are using mixture of simple linear (LR and LDA), 

nonlinear (KNN, CART, NB and SVM) algorithms. 

We reset the random number seed before each run to ensure that the evaluation of each algorithm 

is performed using exactly the same data splits. It ensures the results are directly comparable.



10.We also created a plot of the model evaluation results and compared the spread and the mean accuracy 

of each model. 

There is a population of accuracy measures for each algorithm because each algorithm was evaluated 

10 times (10 fold cross validation).



11.The confusion matrix provides an indication of the three errors made. 

Finally, the classification report provides a breakdown of each class by precision, recall, f1-score and 

support showing excellent results.



"""