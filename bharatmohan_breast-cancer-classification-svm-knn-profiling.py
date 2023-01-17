# Check Python Version

import sys

import scipy

import numpy

import matplotlib

import pandas

import sklearn



print('Python: {}'.format(sys.version))

print('scipy: {}'.format(scipy.__version__))

print('numpy: {}'.format(numpy.__version__))

print('matplotlib: {}'.format(matplotlib.__version__))

print('pandas: {}'.format(pandas.__version__))

print('sklearn: {}'.format(sklearn.__version__))
import pandas as pd

data=pd.read_csv('/kaggle/input/breast-cancer-wisconsin (1).data')
import numpy as np

from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn import model_selection

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt

import pandas as pd
# Load Dataset



names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',

       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',

       'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin (1).data',names=names)

import pandas_profiling as pf 

profile=df.profile_report(style={'full_width':True})
profile
x=profile.get_rejected_variables()

x.append('id')
df.drop(columns=['id'],inplace=True)
# Let explore the dataset and do a few visualizations

print(df.loc[10])



# Print the shape of the dataset

print(df.shape)
# Describe the dataset

print(df.head())
# Plot histograms for each variable

df.hist(figsize = (10, 10))

plt.show()
# Create scatter plot matrix

scatter_matrix(df, figsize = (18,18))

plt.show()
df.replace('?',-9999,inplace=True)
# Create X and Y datasets for training

from sklearn.model_selection import train_test_split

X = np.array(df.drop(['class'], 1))

y = np.array(df['class'])



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Testing Options

seed = 8

scoring = 'accuracy'
# Define models to train

models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))

models.append(('SVM', SVC()))



# evaluate each model in turn

results = []

names = []



for name, model in models:

    kfold = model_selection.KFold(n_splits=10, random_state = seed)

    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Make predictions on validation dataset



for name, model in models:

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print(name)

    print(accuracy_score(y_test, predictions))

    print(classification_report(y_test, predictions))

    

# Accuracy - ratio of correctly predicted observation to the total observations. 

# Precision - (false positives) ratio of correctly predicted positive observations to the total predicted positive observations

# Recall (Sensitivity) - (false negatives) ratio of correctly predicted positive observations to the all observations in actual class - yes.

# F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false 
clf = SVC()



clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)



example_measures = np.array([[4,2,1,1,1,2,3,2,1]])

example_measures = example_measures.reshape(len(example_measures), -1)

prediction = clf.predict(example_measures)

print(prediction)