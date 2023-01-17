# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # Visualizing data



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
header = ['Cement','BlastFurnanceSlag','FlyAsh','Water','Superplasticizer','CoarseAggregate','FineAggregate','Age','CompressiveStrength']



dataset = pd.read_csv('../input/Concrete_Data.csv',names=header)
dataset.head()
dataset.describe()
plt.hist(dataset['Cement'])

plt.show()
plt.hist(dataset['BlastFurnanceSlag'])

plt.show()
plt.hist(dataset['FlyAsh'])

plt.show()
plt.hist(dataset['Water'])

plt.show()
plt.hist(dataset['Superplasticizer'])

plt.show()
plt.hist(dataset['CoarseAggregate'])

plt.show()
plt.hist(dataset['FineAggregate'])

plt.show()
plt.hist(dataset['Age'])

plt.show()
dataset.groupby('CompressiveStrength').size()

plt.figure;

fig = plt.gcf()

fig.set_size_inches(18.5, 10.5)

bp=dataset.boxplot()

 #(kind='box', subplots=True, layout=(4,2), sharex=False, sharey=False)

plt.show()
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



array = dataset.values

X = array[:,0:8]

Y = array[:,8]

validation_size = 0.20

seed = 7

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
scoring = 'accuracy'



models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

models.append(('KNN', KNeighborsClassifier()))

models.append(('CART', DecisionTreeClassifier()))

models.append(('NB', GaussianNB()))

models.append(('SVM', SVC()))
results = []

names = []

print('', '')
import warnings

warnings.filterwarnings("ignore")



for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state=seed)

    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)



    msg = "Name of Algorithm : %s: Accuracy:  %f (Standard Deviation %f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)