# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.ml_intermediate.ex5 import *

print("Setup Complete")
import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

train_data = pd.read_csv('../input/train.csv', index_col='Id')

test_data = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = train_data.SalePrice              

train_data.drop(['SalePrice'], axis=1, inplace=True)



# Select numeric columns only

numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]

X = train_data[numeric_cols].copy()

X_test = test_data[numeric_cols].copy()
X.head()
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer



my_pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', RandomForestRegressor(n_estimators=50, random_state=0))

])
from sklearn.model_selection import cross_val_score



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(my_pipeline, X, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())


def get_score(n_estimators):

    my_pipeline = Pipeline(steps=[

    ('preprocessor', SimpleImputer()),

    ('model', RandomForestRegressor(n_estimators, random_state=0))

])

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    score=-1*cross_val_score(my_pipeline,X,y,cv=3,scoring='neg_mean_absolute_error')

    # Replace this body with your own code

    return score.mean()



# Check your answer

step_1.check()
# Lines below will give you a hint or solution code

#step_1.hint()

step_1.solution()
results = {}

for i in range(1,9):

    results[50*i] = get_score(50*i)

print(results)

# Check your answer

step_2.check()
# Lines below will give you a hint or solution code

step_2.hint()

step_2.solution()
import matplotlib.pyplot as plt

%matplotlib inline



plt.plot(list(results.keys()), list(results.values()))

plt.show()
n_estimators_best = 200



#Estimating best parameters with RandomizedSearchCV

n_estimators=[*range(50,450,50)]

grid_param={'model__n_estimators':n_estimators}

from sklearn.model_selection import RandomizedSearchCV

RFR=RandomForestRegressor(random_state=0)

RFR_random=RandomizedSearchCV(my_pipeline,estimator=RFR, param_distributions=grid_param,cv=3,n_iter=10, verbose=2,random_state=0, n_jobs=-1)

RFR_random.fit(X,y)

print(RFR_random.best_params_)





#Estimating best parameters with GridSearchCV

n_estimators=[*range(50,450,50)]

grid_param={'model__n_estimators':n_estimators}

from sklearn.model_selection import GridSearchCV

RFR=RandomForestRegressor(random_state=0)

RFR_grid=GridSearchCV(my_pipeline,estimator=RFR, param_grid=grid_param,cv=3,verbose=2, n_jobs=-1)

RFR_grid.fit(X,y)

print(RFR_grid.best_params_)





# Check your answer

#step_3.check()
# Lines below will give you a hint or solution code

#step_3.hint()

#step_3.solution()