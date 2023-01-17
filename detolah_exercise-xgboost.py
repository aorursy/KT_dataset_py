# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex6 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import cross_val_score



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = low_cardinality_cols + numeric_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



# One-hot encode the data (to shorten the code, we use pandas)

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)
X_train
from xgboost import XGBRegressor



# Define the model

my_model_1 = XGBRegressor(random_state=0)# Your code here



# Fit the model

my_model_1.fit(X_train, y_train)# Your code here



# Check your answer

step_1.a.check()
# Lines below will give you a hint or solution code

#step_1.a.hint()

#step_1.a.solution()
from sklearn.metrics import mean_absolute_error



# Get predictions

predictions_1 = my_model_1.predict(X_valid) # Your code here



# Check your answer

step_1.b.check()
# Lines below will give you a hint or solution code

#step_1.b.hint()

#step_1.b.solution()
# Calculate MAE

mae_1 = mean_absolute_error(predictions_1, y_valid) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_1)



# Check your answer

step_1.c.check()
# Lines below will give you a hint or solution code

#step_1.c.hint()

#step_1.c.solution()
def get_score(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    my_pipeline = Pipeline(steps=[

                              ('model', XGBRegressor(n_estimators=n_estimators,random_state=0, learning_rate=0.055))

                             ])

    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,

                              cv=3,

                              scoring='neg_mean_absolute_error')

    return scores.mean()



results = {}

for i in range(1,20):

    results[50*i] = get_score(50*i);

import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(results.keys(), results.values())

plt.show()
### Define the model

#my_model_2 = XGBRegressor(n_estimators=200,random_state=0, learning_rate=0.05)# Your code here

my_pipeline2 = Pipeline(steps=[('model', XGBRegressor(colsample_bytree=0.4405, gamma=0, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.8817, n_estimators=1055,

                             reg_alpha=0.5540, reg_lambda=0.3571,

                             subsample=0.5100, silent=0,

                             random_state =0, nthread = 2))

                             ])

my_pipeline2.fit(X_train, y_train)# Fit the model

# Your code here



# Get predictions

predictions_2 = my_pipeline2.predict(X_valid) # Your code here



# Calculate MAE

mae_2 = mean_absolute_error(predictions_2, y_valid) # Your code here



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_2)



# Check your answer

step_2.check()
preds_test = my_pipeline2.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)
# Lines below will give you a hint or solution code

#step_2.hint()

#step_2.solution()
# Define the model

my_model_3 = XGBRegressor(n_estimators=1000,random_state=0, learning_rate=0.15)



# Fit the model

my_model_3.fit(X_train, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)], 

             verbose=False) # Your code here



# Get predictions

predictions_3 = my_model_3.predict(X_valid)



# Calculate MAE

mae_3 = mean_absolute_error(predictions_3, y_valid)



# Uncomment to print MAE

print("Mean Absolute Error:" , mae_3)



# Check your answer

step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()