import pip

import sys

if not 'sklearn' in sys.modules.keys():

    pip.main(['install', 'sklearn'])

#if not 'kaggle' in sys.modules.keys():

#    pip.main(['install', 'kaggle'])

import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import SimpleImputer



# Read the data

train = pd.read_csv('../input/train.csv')



# pull data into target (y) and predictors (X)

train_y = train.Quality

predictor_cols = ['fixed.acidity','volatile.acidity','citric.acid','residual.sugar','chlorides','free.sulfur.dioxide','total.sulfur.dioxide','density','pH','sulphates','alcohol']



# Create training predictors data

train_X = train[predictor_cols]



# Create our imputer to replace missing values with the mean e.g.

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

imp = imp.fit(train_X)


# Impute our data, then train

train_X_imp = imp.transform(train_X)



my_model = RandomForestClassifier(n_estimators=100)

my_model.fit(train_X_imp, train_y)
# The snippet below will retrieve the feature importances from the model and make them into a DataFrame.

feature_importances = pd.DataFrame(my_model.feature_importances_,

                                   index = train_X.columns,

                                   columns=['importance']).sort_values('importance', ascending=False)

feature_importances
# Read the test data

test = pd.read_csv('../input/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]



# Impute each test item, then predict

test_X_imp = imp.transform(test_X)



# Use the model to make predictions

predicted_q = my_model.predict(test_X_imp)

# We will look at the predicted Qualities to ensure we have something sensible.

print(predicted_q)

my_submission = pd.DataFrame({'Id': test.Id, 'Quality': predicted_q})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)
# !kaggle competitions submit -c unibs-mldm-classification -f submission.csv -m "Please describe the technique used"