import random

import numpy as np

import pandas as pd



from sklearn.tree import DecisionTreeClassifier, plot_tree

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt



# 30 features, binary target

# first two features matter, the rest are noise

columns = ['x' + str(n) for n in range(1,31)] # x1, x2, x3....

columns.append('target')

rows = []



SEED = 1

np.random.seed(SEED) # make behaviour deterministic

NUMROWS = int(1.0e4) # 10,000 rows of training data





# create matrix of zeros, then overwrite all but the last column with random numbers.

# The unusual-looking slice notation is standard numpy, which has overridden

# the python indexing operator[]

rows = np.zeros((NUMROWS, 31)) 

rows[:,:-1] = np.random.randint(1, 20, size=(NUMROWS,30))



# fill the target column with the values of the function f(x1, x2) using a mapping function

# I did it this way to learn how to use simple maps in numpy

def f(x1, x2):

    return np.bool((np.square(x1)/4 + np.square(x2)/9) <= 16)

vf = np.vectorize(f)

rows[:,-1] = vf(rows[:,0], rows[:,1]) # vf takes the first two columns as input, i.e. features x1 and x2



# turn the numpy.ndarray into a DataFrame for convenience & clarity

df = pd.DataFrame(data=rows, columns=columns)

df.target = df.target.astype('int64')

df
tree = DecisionTreeClassifier(max_features=None, min_samples_leaf=1, max_depth=None)



X, y = df.drop('target', axis=1), df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)
%%time

tree.fit(X_train, y_train)  #fast!
%%time

predictions = tree.predict(X_test)
print("Accuracy on testing data: {:.2f}".format(accuracy_score(y_test, predictions)))
plt.rcParams['figure.figsize'] = (16, 12)

plot_tree(tree, filled=True, feature_names=X.columns);