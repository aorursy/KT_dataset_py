import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/heart-disease-uci/heart.csv')
data.head()
data.isnull().sum()
#Get Target data 

y = data['target']



#Load X Variables into a Pandas Dataframe with columns 

X = data.drop(['target'], axis = 1)
print(f'X : {X.shape}')
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
print(f'X_train : {X_train.shape}')

print(f'y_train : {y_train.shape}')

print(f'X_test : {X_test.shape}')

print(f'y_test : {y_test.shape}')
# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [2,4]

# Minimum number of samples required to split a node

min_samples_split = [2, 5]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2]

# Method of selecting samples for training each tree

bootstrap = [True, False]
# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

print(random_grid)
rf_Model = RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

rf_random = GridSearchCV(estimator = rf_Model, param_grid = random_grid, cv = 3, verbose=2, n_jobs = -1)
rf_random.fit(X_train, y_train)
rf_random.best_params_
print (f'Train Accuracy - : {rf_random.score(X_train,y_train):.3f}')

print (f'Test Accuracy - : {rf_random.score(X_test,y_test):.3f}')