# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading Libraries
import pandas as pd
import numpy as np
from pandas import read_csv
from pandas import set_option
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
#loading dataset to our dataframe
dataset = read_csv("../input/indian_liver_patient.csv")
"""
#making data analysis to our data
1. viewing the shape of our data
2. viewing the first 20 values in our dataset
3. checking missing values in our dataset by describing it
4. taking a brife view of the data
"""
dataset.shape
dataset.head(20)
dataset.describe(include='all')
dataset.dtypes
dataset.groupby('Dataset').size()
dataset.groupby('Gender').size().plot(kind='bar')
dataset.hist(sharex = False, sharey = False, xlabelsize = 1, ylabelsize = 1)
plt.show()
dataset.plot(kind='density', sharex=False,  layout=(4,4), subplots = True, fontsize = 1, legend=False)
plt.show()
dataset.plot(kind='box', subplots = True, layout = (4,4), sharex=False, sharey=False, fontsize=1)
plt.show()
#corrilation visualisation
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(dataset.corr(), vmin = -1, vmax= 1, interpolation='none')
fig.colorbar(cax)
plt.show()
# Data preperation and Feature scaling
#filling missing values with the median value of the attribute

def fill_na_median(dataset, inplace=True):
    return dataset.fillna(dataset.median(), inplace=True)

fill_na_median(dataset['Albumin_and_Globulin_Ratio'])

dataset.describe()

new_dataset=dataset.drop('Gender', axis=1)
new_dataset.shape
array = new_dataset.values
X = array[:, 0:10]
y = new_dataset['Dataset']

validation_size = 0.20
seed = 7
num_fold = 10
scoring = 'accuracy'
# spliting data for trainning and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = validation_size, random_state = seed)
## spot checking algorithm performance
models = []
models.append(('LD', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = KFold(n_splits = num_fold, random_state=seed)
    cv_results = cross_val_score(model, X_train, y_train, cv = kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s %f(%f)" %(name, cv_results.mean(), cv_results.std())
    print(msg)
## Comparing Algorithms
fig = plt.figure()
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
"""
As we can see that both KNN and CART are having accuracy mean score of 100% and standard Deviation of 0, am going to choose one of the 
two to use. And on my side am taking CART as it performs better on classification problems
"""
model2 = DecisionTreeClassifier()
model2.fit(X_train, y_train)

predictions2 = model2.predict(X_test)
accuracy_score(y_test, predictions2)
confusion_matrix(y_test, predictions2)
print(classification_report(y_test, predictions2))