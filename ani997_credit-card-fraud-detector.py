# Credit card Fraud Detection using Local Outlier Factor and Isolation Forest Algorithm

# Thank You EDUONIX LEARNING SOLUTIONS for providing this kernel





# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import sys

import numpy

import pandas

import matplotlib

import seaborn 

import scipy

import sklearn



print('Python: {}'.format(sys.version))

print('Numpy: {}'.format(numpy.__version__))

print('Pandas: {}'.format(pandas.__version__))

print('Matplotlib: {}'.format(matplotlib.__version__))

print('Seaborn: {}'.format(seaborn.__version__))

print('Scipy: {}'.format(scipy.__version__))

print('Sklearn: {}'.format(sklearn.__version__))
# importing the necessary packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load the dataset

data = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
# exploring the dataset

print(data.columns)
print(data.shape)
print(data.describe())
data = data.sample(frac = 0.5, random_state = 1)

print(data.shape)
# plot histogram of each parameter

data.hist(figsize = (20,20))

plt.show()
# determine the no of fraudulent cases

Fraud = data[data['Class'] == 1]

Valid = data[data['Class'] == 0]



outlier_fraction = len(Fraud) / float(len(Valid))

print(outlier_fraction)



print('Fraud cases: {}'.format(len(Fraud)))

print('Valid cases: {}'.format(len(Valid)))
# correlation matrix

corrmat = data.corr()

fig = plt.figure(figsize = (12,9))



sns.heatmap(corrmat, vmax = .8, square = True)

plt.show()
# Get all the columns from the DataFrame

columns = data.columns.tolist()



# Filter the columns to remove unwanted data

columns = [c for c in  columns if c not in ["Class"]]



# Store the variable we'll be predicting on

target = "Class"



X = data[columns]

Y = data[target]



# Print the shapes of X and Y

print(X.shape)

print(Y.shape)
from sklearn.metrics import classification_report, accuracy_score

from sklearn.ensemble import IsolationForest

from sklearn.neighbors import LocalOutlierFactor



# define a random state

state = 1



# define the outlier detection methods

classifiers = {

    "Isolation Forest": IsolationForest(max_samples=len(X),

                                       contamination = outlier_fraction,

                                       random_state = state),

    "Local Outlier Factor": LocalOutlierFactor(

    n_neighbors = 20,

    contamination = outlier_fraction)

}
# Fit the model

n_outliers = len(Fraud)



for i, (clf_name,clf) in enumerate(classifiers.items()):

    

    # fit the data and tag outliers

    if clf_name == "Local Outlier Factor":

        y_pred = clf.fit_predict(X)

        scores_pred = clf.negative_outlier_factor_

    else:

        clf.fit(X)

        scores_pred = clf.decision_function(X)

        y_pred = clf.predict(X)

        

    # Reshape the prediction values to 0 for valid,1 for fraud

    y_pred[y_pred == 1] = 0

    y_pred[y_pred == -1] = 1

    

    n_errors = (y_pred != Y).sum()

    

    # Run classification metrics

    print('{}: {}'.format(clf_name, n_errors))

    print(accuracy_score(Y,y_pred))

    print(classification_report(Y, y_pred))