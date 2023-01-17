#Installation (not to be done on Kaggle, but if we were doing this on a desktop)

#pip install scipy, numpy, matplotlib, pandas, sklearn



# Checking installations and correct versions of libraries

# Python version

import sys

print('Python: {}'.format(sys.version))

# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# dataframe

import numpy

import pandas

myarray = numpy.array([[1, 2, 3], [4, 5, 6]])

rownames = ['Monday', 'Tuesday']

colnames = ['Person_one', 'Person_two', 'Person_three']

mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)

print(mydataframe)
import pandas as pd



# Take Dictionary as input to your dataframe

my_dict = {1: ['1', '4'], 2: ['2', '5'], 3: ['3', '6']}

print(pd.DataFrame(my_dict))
# Take a DataFrame as input to your DataFrame 

my_df = pd.DataFrame(data=[4,5,6,7], index=range(0,4), columns=['Person_one'])

print(pd.DataFrame(my_df))
# Take a Series as input to your DataFrame

my_series = pd.Series({"United Kingdom":"London", "India":"New Delhi", "United States":"Washington DC", "Belgium":"Brussels"})

print(pd.DataFrame(my_series))
# Load CSV using Pandas from URL

import pandas



url = "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal length in cm', 'sepal width in cm', 'petal  length in cm', 'petal  width in cm', 'class']

data = pandas.read_csv(url, names=names)

print(data.head())

# print the shape of data

print(data.shape)
# describe the data

description = data.describe()

print(description)
# print the datatypes

print(data.dtypes)
# looking at how many values of each class of iris flower do I have:

data1 = data.loc[:,"class"]  # edited dataframe with just the "class" column

data1.value_counts()  # counting number of rows for every value of "class"
# Histogram



import matplotlib.pyplot as plt

data.hist()

# figured out the size of the plot I want to show by trial and error

plt.rcParams["figure.figsize"] = [20, 15]

plt.show()
# Scatter plot matrix



import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix

scatter_matrix(data)

# figured out the size of the plot I want to show by trial and error

plt.rcParams["figure.figsize"] = [20, 15]

plt.show()
from sklearn.preprocessing import StandardScaler

import pandas as pd

import numpy

array = data.values

# separate array into input and output components

X = array[:,0:4]  # input

Y = array[:,4]    # output

scaler = StandardScaler().fit(X)

print(scaler)

rescaledX = scaler.transform(X)

# summarize transformed data

numpy.set_printoptions(precision=2)

print(rescaledX[0:5,:])
##ignore all the warnings, there are a lot of warnings in our code since we use a lot of default parameters

import warnings

warnings.filterwarnings("ignore")
# Evaluate using Cross Validation

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



array = data.values

X = array[:,0:4]  # input

Y = array[:,4]    # output

kfold = KFold(n_splits=10, random_state=7)

# K fold cross validation where K = 10, and random number generator seed = 7 (Pseudo-random number generator state used for random sampling.)



model = LogisticRegression()

results = cross_val_score(model, X, Y, cv=kfold)

print("Accuracy: ", (results.mean()*100.0, results.std()*100.0))
# Cross Validation Classification Recall Macro

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression



array = data.values

X = array[:,0:4]

Y = array[:,4]

kfold = KFold(n_splits=10, random_state=7)

model = LogisticRegression()

scoring = 'recall_macro'                       ## neg_log_loss only for binary targets and requires predict_proba support

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print("RecallMacro: ", (results.mean(), results.std()))
# KNN Regression

import pandas as pd

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.neighbors import KNeighborsRegressor

array = data.values





X = array[:,0:4]

Y = array[:,4]

Y = pd.factorize(Y)[0]   ## convert the categorical 'Class' column to numerical data

# using only int variables because clustering can't use categorical (obj) vars

kfold = KFold(n_splits=10, random_state=7)

model = KNeighborsRegressor()

scoring = 'neg_mean_squared_error'

results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

print(results.mean())

# Compare Algorithms

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

array = data.values

X = array[:,0:4]

Y = array[:,4]

# prepare models

models = []

models.append(('LR', LogisticRegression()))

models.append(('LDA', LinearDiscriminantAnalysis()))

# evaluate each model in turn

results = []

names = []

scoring = 'accuracy'

for name, model in models:

    kfold = KFold(n_splits=10, random_state=7)

    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    print(msg)
# Grid Search for Algorithm Tuning

from pandas import read_csv

import numpy

from sklearn.linear_model import Ridge

from sklearn.model_selection import GridSearchCV

array = data.values

X = array[:,0:4]

Y = array[:,4]

Y = pd.factorize(Y)[0]  ## convert the categorical 'Class' column to numerical data

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])

param_grid = dict(alpha=alphas)

model = Ridge()

grid = GridSearchCV(estimator=model, param_grid=param_grid)

grid.fit(X, Y)

print(grid.best_score_)

print(grid.best_estimator_.alpha)
# Random Forest Classification

from pandas import read_csv

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import RandomForestClassifier



array = data.values

X = array[:,0:4]

Y = array[:,4]

num_trees = 100

max_features = 3

kfold = KFold(n_splits=10, random_state=7)

model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)

results = cross_val_score(model, X, Y, cv=kfold)

print(results.mean())
# Save Model Using Pickle

from pandas import read_csv

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import pickle



array = data.values

X = array[:,0:4]

Y = array[:,4]

test_size = 0.33

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Fit the model on 33%

model = LogisticRegression()

model.fit(X_train, Y_train)

# save the model to disk

filename = 'finalized_model.sav'

pickle.dump(model, open(filename, 'wb'))
# load the model from disk

loaded_model = pickle.load(open(filename, 'rb'))

result = loaded_model.score(X_test, Y_test)

print(result)