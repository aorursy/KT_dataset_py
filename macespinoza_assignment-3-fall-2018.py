import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
X = np.linspace(-2, 2, 7)
y = X ** 3

plt.scatter(X, y)
plt.xlabel(r'$x$')
plt.ylabel(r'$y$');
# You code here
# You code here
def regression_var_criterion(X, y, t):
    pass
    # You code here
# You code here
# You code here
#df = pd.read_csv('../input/mlbootcamp5_train.csv',index_col='id', sep=';')
df = pd.read_csv('../input/mlbootcamp5_train.csv',index_col='id', sep=',')
df.head()
# You code here
# You code here
# X_train, X_valid, y_train, y_valid = ...
# You code here
# You code here
tree_params = {'max_depth': list(range(2, 11))}

tree_grid = GridSearchCV # You code here
# You code here
# You code here
# You code here