import matplotlib

import sklearn

import matplotlib.pyplot as plt

print(f'matplotlib: {matplotlib.__version__}')

print(f'sklearn   : {sklearn.__version__}')
import pandas as pd

print(f'pandas version: {pd.__version__}')

test = pd.read_csv("../input/titanic/test.csv")

train = pd.read_csv("../input/titanic/train.csv")
train.describe()
train.head(5)
train.dtypes
# For now, drop columns that would take much more work to get into a useable format

def format_data(data):

    # On-hot encode gender & embarked

    data = pd.get_dummies(data, columns=['Sex','Embarked'])

    # Drop columns too complicated for this very simple trial

    data = data.drop(['Name','Ticket','Cabin'], axis=1)

    # Fill null values with the mean of the column

    data.fillna(data.mean(), inplace=True)

    if 'Survived' in data.columns:

        data_y = data['Survived']

        data_x = data.drop(['Survived'], axis=1)

        return data_x, data_y

    else:

        return data



train_x, train_y = format_data(train)

test_x = format_data(test)



# Pull out the 

train_x.describe()
# Import the models

from sklearn.ensemble import RandomForestClassifier



# This is the most simple random forest model that we can derive

model = RandomForestClassifier(random_state=1)

model.fit(train_x, train_y);
# Let's try splitting the data into training and testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, 

                                                    test_size=0.2, 

                                                    random_state=42)

# Now retrain our model on the testing

model.fit(X_train, y_train)



# Print some statistics

from sklearn.metrics import f1_score



def summary_stats(x,y):

    pred = model.predict(x)

    f1   = f1_score(pred, y)

    acc  = model.score(x, y)

    print(f"   F1 score: {f1}")

    print(f"   Accuracy: {acc}")

print(f"Training:")

summary_stats(X_train, y_train)

print(f"Testing:")

summary_stats(X_test, y_test)
# Create the values that we will be testing

search_pars = {

    'n_estimators': [10, 30, 100, 300, 1000],

    'max_features': [0.25, 0.5, 0.75, 1.0],

    'criterion'   : ['gini', 'entropy'] 

}
from sklearn.model_selection import GridSearchCV



# Construct the model

rf_model = RandomForestClassifier(random_state=1)

clf      = GridSearchCV(rf_model, search_pars)

clf.fit(X_train, y_train);
# Get the results

print(clf.best_score_)

print(clf.best_params_)

tune_results = pd.DataFrame(clf.cv_results_)

tune_results.sort_values('rank_test_score')
# Get the best model and re-fit it on all our training data

model = clf.best_estimator_

model.fit(train_x, train_y);
# Now generate some statistics

print("Final model training results:")

summary_stats(train_x, train_y)
# Predictions on test data

pred_test = model.predict(test_x)
submission = pd.DataFrame({"PassengerId": test_x['PassengerId'], 

                           "Survived":pred_test})

submission.describe()
submission.to_csv('submission.csv', index=False)