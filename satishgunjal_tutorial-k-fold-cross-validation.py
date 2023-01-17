import pandas as pd

import numpy as np

from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from sklearn import linear_model, tree, ensemble
train_data = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')



# Remove rows with missing target values

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data.SalePrice # Target variable             

train_data.drop(['SalePrice'], axis=1, inplace=True) # Removing target variable from training data



train_data.drop(['LotFrontage', 'GarageYrBlt', 'MasVnrArea'], axis=1, inplace=True) # Remove columns with null values



# Select numeric columns only

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

X = train_data[numeric_cols].copy()



print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))



X.head() # Show first 5 training examples
# Lets split the data into 5 folds.  

# We will use this 'kf'(KFold splitting stratergy) object as input to cross_val_score() method

kf =KFold(n_splits=5, shuffle=True, random_state=42)



cnt = 1

# split()  method generate indices to split data into training and test set.

for train_index, test_index in kf.split(X, y):

    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')

    cnt += 1
"""

Why we are using '-' sign to calculate RMSE?

ANS: Classification accuracy is reward function, means something you want to maximize. Mean Square Error is loss function, 

means something you want to minimize. Now if we use 'cross_val_score' function then best score(high value) will give worst 

model in case of loss function! There are other sklearn functions which also depends on 'cross_val_score' to select best model by

looking for highest scores, so a design decision was made for 'cross_val_score' to negate the output of all loss function. 

So that when other sklearn function calls 'cross_val_score' those function can always assume that highest score indicate better model.

In short ignore the negative sign and rate the error based on its absolute value.

"""

def rmse(score):

    rmse = np.sqrt(-score)

    print(f'rmse= {"{:.2f}".format(rmse)}')
score = cross_val_score(linear_model.LinearRegression(), X, y, cv= kf, scoring="neg_mean_squared_error")

print(f'Scores for each fold: {score}')

rmse(score.mean())
score = cross_val_score(tree.DecisionTreeRegressor(random_state= 42), X, y, cv=kf, scoring="neg_mean_squared_error")

print(f'Scores for each fold: {score}')

rmse(score.mean())
score = cross_val_score(ensemble.RandomForestRegressor(random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")

print(f'Scores for each fold are: {score}')

rmse(score.mean())
max_depth = [1,2,3,4,5,6,7,8,9,10]



for val in max_depth:

    score = cross_val_score(tree.DecisionTreeRegressor(max_depth= val, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")

    print(f'For max depth: {val}')

    rmse(score.mean())
estimators = [50, 100, 150, 200, 250, 300, 350]



for count in estimators:

    score = cross_val_score(ensemble.RandomForestRegressor(n_estimators= count, random_state= 42), X, y, cv= kf, scoring="neg_mean_squared_error")

    print(f'For estimators: {count}')

    rmse(score.mean())
train_data = pd.read_csv('/kaggle/input/titanic/train.csv')



# Remove rows with missing target values

train_data.dropna(axis=0, subset=['Survived'], inplace=True)

y = train_data.Survived # Target variable             

train_data.drop(['Survived'], axis=1, inplace=True) # Removing target variable from training data



train_data.drop(['Age'], axis=1, inplace=True) # Remove columns with null values



# Select numeric columns only

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

X = train_data[numeric_cols].copy()



print("Shape of input data: {} and shape of target variable: {}".format(X.shape, y.shape))

pd.concat([X, y], axis=1).head() # Show first 5 training examples
# Lets split the data into 5 folds. 

# We will use this 'kf'(StratiFiedKFold splitting stratergy) object as input to cross_val_score() method

# The folds are made by preserving the percentage of samples for each class.

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)



cnt = 1

# split()  method generate indices to split data into training and test set.

for train_index, test_index in kf.split(X, y):

    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')

    cnt+=1

    

# Note that: 

# cross_val_score() parameter 'cv' will by default use StratifiedKFold spliting startergy if we just specify value of number of folds. 

# So you can bypass above step and just specify cv= 5 in cross_val_score() function
score = cross_val_score(linear_model.LogisticRegression(random_state= 42), X, y, cv= kf, scoring="accuracy")

print(f'Scores for each fold are: {score}')

print(f'Average score: {"{:.2f}".format(score.mean())}')
score = cross_val_score(tree.DecisionTreeClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")

print(f'Scores for each fold are: {score}')

print(f'Average score: {"{:.2f}".format(score.mean())}')
score = cross_val_score(ensemble.RandomForestClassifier(random_state= 42), X, y, cv= kf, scoring="accuracy")

print(f'Scores for each fold are: {score}')

print(f'Average score: {"{:.2f}".format(score.mean())}')
algorithms = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']



for algo in algorithms:

    score = cross_val_score(linear_model.LogisticRegression(max_iter= 4000, solver= algo, random_state= 42), X, y, cv= kf, scoring="accuracy")

    print(f'Average score({algo}): {"{:.3f}".format(score.mean())}')

    

# Note, here we are using max_iter = 4000, so that all the solver gets chance to converge. 
max_depth = [1,2,3,4,5,6,7,8,9,10]



for val in max_depth:

    score = cross_val_score(tree.DecisionTreeClassifier(max_depth= val, random_state= 42), X, y, cv= kf, scoring="accuracy")

    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')
n_estimators = [50, 100, 150, 200, 250, 300, 350]



for val in n_estimators:

    score = cross_val_score(ensemble.RandomForestClassifier(n_estimators= val, random_state= 42), X, y, cv= kf, scoring="accuracy")

    print(f'Average score({val}): {"{:.3f}".format(score.mean())}')