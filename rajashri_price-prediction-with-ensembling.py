import pandas as pd

import time

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



# Import Model Libraries

import xgboost as xgb

from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split



# Train a base decision tree regressor model on the data

from sklearn.tree import DecisionTreeRegressor



import warnings

warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston

boston = load_boston()
bos = pd.DataFrame(boston.data)

bos.columns = boston.feature_names

bos['PRICE'] = boston.target

bos.head()
# Split Train/Test Set

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(bos.drop(["PRICE"], axis=1), bos["PRICE"],random_state=10, test_size=0.25)
# Examine our dataset

X_train_2.head()
# Examine shape of the dataset

X_train_2.shape
# Check for any missing values

X_train_2.isnull().any()
# Train a base decision tree regressor model on the data

from sklearn.tree import DecisionTreeRegressor

# Fit model

tree_reg1 = DecisionTreeRegressor(max_depth=2)

tree_reg1.fit(X_train_2,y_train_2)
# Compute errors/residuals on first tree

r1 = y_train_2 - tree_reg1.predict(X_train_2)
# Fit second model

tree_reg2 = DecisionTreeRegressor(max_depth=2)

tree_reg2.fit(X_train_2,r1)
# Compute errors/residuals on second tree

r2 = r1 - tree_reg2.predict(X_train_2)
# Fit third model

tree_reg3 = DecisionTreeRegressor(max_depth=2)

tree_reg3.fit(X_train_2,r2)
# Add up the predictions of each tree model, which is our ensemble of three trees

y_pred = sum(tree.predict(X_train_2) for tree in (tree_reg1, tree_reg2, tree_reg3))

y_pred[:10]
#actual values

y_train_2[:10]
tree_reg1.predict(X_train_2)[:10]
# Create dataframe of all predictions

predictions = pd.DataFrame(tree_reg1.predict(X_train_2)[:10], columns=['Model_1'])

predictions['Model_2'] = pd.DataFrame(tree_reg2.predict(X_train_2)[:10])

predictions['Model_3'] = pd.DataFrame(tree_reg3.predict(X_train_2)[:10])

predictions['Ensemble'] = pd.DataFrame(y_pred[:10])

predictions['Actual'] = y_train_2.head(10).reset_index()['PRICE']



# Display predictions

predictions
errors = []

for n_estimators in [1,2,3,4,5,6,7,8,9,10]:

    clf = xgb.XGBRegressor(max_depth=2, n_estimators=n_estimators)

    clf.fit(X_train_2, y_train_2, verbose=False)

    errors.append(

        {

            'Tree Count': n_estimators,

            'Average Error': np.average(y_train_2 - clf.predict(X_train_2)),

        })

    

n_estimators_lr = pd.DataFrame(errors).set_index('Tree Count').sort_index()

n_estimators_lr
# Use the Sklearn GradientBoostingRegressor ensemble method to perform the same thing as the previous code above

from sklearn.ensemble import GradientBoostingRegressor



gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=3)

gbrt.fit(X_train_2,y_train_2)
import xgboost as xgb

GradientBoostingRegressor()
# Examine the default parameters

xgb.XGBRegressor()
# Create empty array to store results

results = []

# Create watchlist to keep track of train/validation performance

eval_set = [(X_train_2, y_train_2), (X_test_2, y_test_2)]
# Enumerate through different max_depth values and store results

for max_depth in [2,3,4,5,10,12,15]:

    clf = xgb.XGBRegressor(max_depth=max_depth)

    clf.fit(X_train_2, y_train_2, eval_set=eval_set, verbose=False)

    results.append(

        {

            'max_depth': max_depth,

            'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),

            'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))

        })

    

# Display Results

max_depth_lr = pd.DataFrame(results).set_index('max_depth').sort_index()

max_depth_lr
# Plot Max_Depth Learning Curve

max_depth_lr.plot(title="Max_Depth Learning Curve")
# Reset results array

results = []



for learning_rate in [0.05,0.1,0.2,0.4,0.6,0.8,1]:

    clf = xgb.XGBRegressor(max_depth=2,learning_rate=learning_rate, n_estimators=200)

    clf.fit(X_train_2, y_train_2, eval_set=eval_set, verbose=False)

    results.append(

        {

            'learning_rate': learning_rate,

            'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),

            'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))

        })

    

learning_rate_lr = pd.DataFrame(results).set_index('learning_rate').sort_index()

learning_rate_lr
# Plot Learning Rate

learning_rate_lr.plot(title="Learning Rate Learning Curve")
# Reset results array

results = []



for n_estimators in [50,60,100,150,200,500,750,1000, 1500]:

    clf = xgb.XGBRegressor(max_depth=2,learning_rate=0.10, n_estimators=n_estimators)

    clf.fit(X_train_2, y_train_2, eval_set=eval_set, verbose=False)

    results.append(

        {

            'n_estimators': n_estimators,

            'train_error': metrics.mean_squared_error(y_train_2, clf.predict(X_train_2)),

            'test_error': metrics.mean_squared_error(y_test_2, clf.predict(X_test_2))

        })

    

n_estimators_lr = pd.DataFrame(results).set_index('n_estimators').sort_index()

n_estimators_lr
n_estimators_lr.plot(title="N_Estimators Learning Curve")