# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
RANDOM_STATE=5
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#read in data
train = pd.read_csv(dirname+'/'+filenames[0])
train.Cabin.fillna(value='No Cabin',inplace=True)
gender_submission = pd.read_csv(dirname+'/'+filenames[1])
test = pd.read_csv(dirname+'/'+filenames[2])
test.Cabin.fillna(value='No Cabin',inplace=True)


def CabinEncoder(cab):
    if cab=='No Cabin':
        return 0
    if 'A' in cab:
        return 1
    if 'B' in cab:
        return 2
    if 'C' in cab:
        return 3
    if 'D' in cab:
        return 4
    if 'E' in cab:
        return 5
    if 'F' in cab:
        return 6
    if 'G' in cab:
        return 7
    if 'T' in cab:
        return 8
    return cab
train.Cabin = train.Cabin.apply(CabinEncoder)
test.Cabin = test.Cabin.apply(CabinEncoder)
gender_submission
test
#model training functions
def train_model(X_train, y_train, X_test, n_estimators=100,random_state=RANDOM_STATE):
    model = RandomForestRegressor(n_estimators=100,random_state=RANDOM_STATE)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    return y_pred
    
def train_from_cols(cols):
    return train_model(X_train[cols], y_train, X_test[cols])
def get_mae(y_true,y_pred):
    return mean_absolute_error(y_true,y_pred)

def prob_to_definite(val):
    if val < 0.5:
        return 0
    return 1
#split data
train_indexed = train.set_index('PassengerId')
X_train_full = train_indexed.drop('Survived',axis=1)
y_train_full = pd.Series(train_indexed.Survived, index=train_indexed.index)
X_train, X_test, y_train, y_test = train_test_split(X_train_full,y_train_full,test_size=0.2,random_state=RANDOM_STATE)
#separate columns
cat_columns = [cat for cat in X_train.columns if X_train[cat].dtype == 'object']
num_columns = [num for num in X_train.columns if num not in cat_columns]
col_without_missing = [missing for missing in X_train.columns if not X_train[missing].isnull().any()]
num_col_not_missing = [col for col in num_columns if col in col_without_missing]
num_col_missing = [col for col in num_columns if X_train[col].isnull().any() and X_train[col].dtype in ['int64','float64']]

def get_pipeline(n_estimators=100,new_model=RandomForestRegressor()):
    if n_estimators != -1:
        model=RandomForestRegressor(n_estimators=n_estimators, random_state=RANDOM_STATE)
    else:
        model=new_model
    num_transformer = SimpleImputer(strategy='median')
    cat_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                      ('onehot',OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_columns),
            ('cat', cat_transformer, ['Sex'])
        ])
    my_pipeline=Pipeline([('preprocessor', preprocessor),
                        ('model',model)])
    return my_pipeline


#cross validating mae
min = 1
n_est = 0
for i in [1,50,100,500,1000]:    
    my_pipeline=get_pipeline(n_estimators = i)
    scores = -1 * cross_val_score(my_pipeline, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
    print(scores.mean())
    if scores.mean() < min:
        min = scores.mean()
        n_est = i
print('Mean MAE of lowest: ' + str(min))
print('best n_est: ' + str(n_est))
xgb=XGBRegressor(n_estimators=1000, learning_rate=0.01)
my_pipe=get_pipeline(-1,xgb)
my_pipe.fit(X_train,y_train)
scores = -1 * cross_val_score(my_pipe, X_train, y_train,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
print(scores.mean())


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'model__n_estimators': n_estimators,
               'model__max_features': max_features,
               'model__max_depth': max_depth,
               'model__min_samples_split': min_samples_split,
               'model__min_samples_leaf': min_samples_leaf,
               'model__bootstrap': bootstrap}
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
my_pipeline=get_pipeline(n_estimators=-1,new_model=rf)
rf_random = RandomizedSearchCV(estimator = my_pipeline, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=RANDOM_STATE, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_full, y_train_full)
rf_random.best_params_ = {'model__n_estimators': 1600,
 'model__min_samples_split': 2,
 'model__min_samples_leaf': 1,
 'model__max_features': 'sqrt',
 'model__max_depth': 10,
 'model__bootstrap': True}
#rf_random.best_estimator

scores = -1 * cross_val_score(rf_random.best_estimator_, X_train_full, y_train_full,
                                  cv=5,
                                  scoring='neg_mean_absolute_error')
print(scores.mean())
output = rf_random.best_estimator_.predict(test.set_index('PassengerId'))
output1 = pd.DataFrame({'PassengerId': test.PassengerId,
                       'Survived': output})
output = pd.DataFrame({'PassengerId': test.PassengerId,
                        'Survived':output1['Survived'].apply(prob_to_definite)})

output.to_csv('/kaggle/working/submission.csv', index=False)