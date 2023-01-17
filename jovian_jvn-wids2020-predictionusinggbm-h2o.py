# importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.impute import SimpleImputer

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier ,AdaBoostClassifier

from sklearn.model_selection import train_test_split

import lightgbm as lgb

from sklearn import preprocessing

import h2o

from h2o.estimators.gbm import H2OGradientBoostingEstimator

h2o.init()
# loading dataset 

training_v2 = pd.read_csv("../input/widsdatathon2020/training_v2.csv")
# creating independent features X and dependant feature Y

y = pd.DataFrame(training_v2['hospital_death'])

X = training_v2

X = training_v2.drop('hospital_death',axis = 1)
# Remove Features with more than 75 percent missing values

train_missing = (X.isnull().sum() / len(X)).sort_values(ascending = False)

train_missing = train_missing.index[train_missing > 0.60]

X = X.drop(columns = train_missing)
#Convert categorical variable into dummy/indicator variables.

X = pd.get_dummies(X)
# Imputation transformer for completing missing values.

my_imputer = SimpleImputer()

new_data = pd.DataFrame(my_imputer.fit_transform(X))

new_data.columns = X.columns

X= new_data
# Threshold for removing correlated variables

threshold = 0.9



# Absolute value correlation matrix

corr_matrix = X.corr().abs()

corr_matrix.head()

# Upper triangle of correlations

upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

upper.head()

# Select columns with correlations above threshold

to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

print('There are %d columns to remove.' % (len(to_drop)))

#Drop the columns with high correlations

X = X.drop(columns = to_drop)
# Initialize an empty array to hold feature importances

feature_importances = np.zeros(X.shape[1])



# Create the model with several hyperparameters

model = lgb.LGBMClassifier(objective='binary', boosting_type = 'goss', n_estimators = 10000, class_weight = 'balanced')

for i in range(2):

    

    # Split into training and validation set

    train_features, valid_features, train_y, valid_y = train_test_split(X, y, test_size = 0.25, random_state = i)

    

    # Train using early stopping

    model.fit(train_features, train_y, early_stopping_rounds=100, eval_set = [(valid_features, valid_y)],eval_metric = 'auc', verbose = 200)

    

    # Record the feature importances

    feature_importances += model.feature_importances_

# Make sure to average feature importances! 

feature_importances = feature_importances / 2

feature_importances = pd.DataFrame({'feature': list(X.columns), 'importance': feature_importances}).sort_values('importance', ascending = False)

# Find the features with zero importance

zero_features = list(feature_importances[feature_importances['importance'] == 0.0]['feature'])

print('There are %d features with 0.0 importance' % len(zero_features))

# Drop features with zero importance

X = X.drop(columns = zero_features)
X = y.join(X)

X = h2o.H2OFrame(X)
# split into train and validation sets

train, valid = X.split_frame(ratios = [.8], seed = 1234)
train[0] = train[0].asfactor()

valid[0] = valid[0].asfactor()
param = {

      "ntrees" : 100

    , "max_depth" : 10

    , "learn_rate" : 0.02

    , "sample_rate" : 0.7

    , "col_sample_rate_per_tree" : 0.9

    , "min_rows" : 5

    , "seed": 4241

    , "score_tree_interval": 100

}

from h2o.estimators import H2OXGBoostEstimator

model = H2OXGBoostEstimator(**param)

model.train(x = list(range(1, train.shape[1])), y = 0, training_frame = train,validation_frame = valid)
model.model_performance(valid)