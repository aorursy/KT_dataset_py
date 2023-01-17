# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Import required functions
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Supress warnings
import warnings 
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
# Whole dataset on Iris Species
dataset_w = pd.read_csv('../input/Iris.csv')
#Prints out the size of out dataset and also first 5 rows
print('Data shape: ', dataset_w.shape)
dataset_w.head()
#Examine the number of flowers in each category
dataset_w['Species'].value_counts()
#Display the Species names
dataset_w['Species'].unique()
#Summary of Irist dataset
dataset_w.describe()
columns=dataset_w.columns
print(columns)
# box and whisker plots of 4 features
dataset_w[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
import matplotlib
matplotlib.pyplot.show()
# histograms
dataset_w[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']].hist()
matplotlib.pyplot.show()
import seaborn as sns
# A seaborn jointplot shows bivariate scatterplots and univariate histograms in the same figure
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=dataset_w)
sns.FacetGrid(dataset_w, hue="Species", size=5) \
   .map(matplotlib.pyplot.scatter, "SepalLengthCm", "SepalWidthCm") \
   .add_legend()
sns.FacetGrid(dataset_w, hue="Species", size=5) \
   .map(matplotlib.pyplot.scatter, "PetalLengthCm", "PetalWidthCm") \
   .add_legend()
sns.pairplot(dataset_w.drop("Id", axis=1), hue="Species", size=3)
# Split data randomly into training and validation subsets
array = dataset_w.values
X = array[:,1:5]
Y = array[:,5]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
# Training and cross-validation a selection of algorithms
scoring="accuracy"
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

# Make predictions on validation dataset
svc = SVC()
svc.fit(X_train, Y_train)
predictions = svc.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))
