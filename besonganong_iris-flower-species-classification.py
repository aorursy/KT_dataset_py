# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import datasets

from sklearn import tree

import seaborn as sns # visualization

import random as rnd

import sys



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print(sys.version)
# Acquire the required datasets

iris = datasets.load_iris()

iris
# Convert Iris datasets to pandas Dataframe

df = pd.DataFrame(data = np.c_[iris['data'], iris['target']],

                  columns = iris['feature_names'] + ['target'])



df.head()
# Explore datasets

df.describe()
# Visualize datasets to understand underlying relationships and patterns

sns.set_style('whitegrid')

sns.pairplot(df, hue='target')
# Prepare training datasets



X = iris.data[0:150, :] # X represents training features dataset.

X.shape # Returns 150 rows (or samples) across 4 features (or columns)
Y = iris.target[0:150] # Y represents training target or classification dataset.

Y.shape
# Prepare testing datasets



# First test sample



# Randomize test sample extraction from  training dataset.

setosa_index = rnd.randrange(0, 49)

test_setosa = [iris.data[setosa_index, :]]



# Remove test sample from training dataset

X = np.delete(X, setosa_index, 0) 

Y = np.delete(Y, setosa_index, 0)

test_setosa, iris.target_names[iris.target[setosa_index]], X.shape, Y.shape # Displays sample List, target /

#Classification name, new shape of our X and Y training data.
# Second test sample



# Randomize test sample extraction from  training dataset.

versicolor_index = rnd.randrange(50, 99)

test_versicolor = [iris.data[versicolor_index, :]]



# Remove test sample from training dataset

X = np.delete(X, versicolor_index, 0) 

Y = np.delete(Y, versicolor_index, 0)

test_versicolor, iris.target_names[iris.target[versicolor_index]], X.shape, Y.shape # Displays sample List, target /

#Classification name, new shape of our X and Y training data.
# Third test sample



# Randomize test sample extraction from  training dataset.

virginica_index = rnd.randrange(100, 150)

test_virginica = [iris.data[virginica_index, :]]



# Remove test sample from training dataset

X = np.delete(X, virginica_index, 0) 

Y = np.delete(Y, virginica_index, 0)

test_virginica, iris.target_names[iris.target[virginica_index]], X.shape, Y.shape # Displays sample List, target /

#Classification name, new shape of our X and Y training data.
# Model and train

model_tree = tree.DecisionTreeClassifier()

model_tree.fit(X, Y)
# Predict



# Given a set of features, to which species or classification does this flower belong

pred_tree_setosa = model_tree.predict(test_setosa)

# Print resut

print('Decision Tree predicts {} for test_setosa'.format(iris.target_names[pred_tree_setosa]))
# Given a set of features, to which species or classification does this flower belong

pred_tree_versicolor = model_tree.predict(test_versicolor)

# Print resut

print('Decision Tree predicts {} for test_versicolor'.format(iris.target_names[pred_tree_versicolor]))
# Given a set of features, to which species or classification does this flower belong

pred_tree_virginica = model_tree.predict(test_virginica)

# Print resut

print('Decision Tree predicts {} for test_virginica'.format(iris.target_names[pred_tree_virginica]))