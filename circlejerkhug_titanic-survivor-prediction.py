# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.




# Read the data

train= pd.read_csv('../input/train.csv')

train['Label'] = 'train'



test = pd.read_csv('../input/test.csv')

test['Label'] = 'test'



# Process as full dataset

train.dropna(axis=0, subset=['Survived'], inplace=True) # Drop rows with uknown survival

X_full = pd.concat([train.drop('Survived', axis = 1), test], axis = 0)

X_full.drop('PassengerId', axis = 1, inplace=True)



# Select categorical columns

print("Categorical features: ", [cname for cname in X_full.columns if X_full[cname].dtype == "object"])



# Select numeric columns

print("Numeric features: ", [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']])



X_full.head()

missing_val_count_by_column = (X_full.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder



# Remove unuseful features

X_full.drop('Name', axis=1, inplace=True)

X_full.drop('Ticket', axis=1, inplace=True)

X_full.drop('Cabin', axis=1, inplace=True)



# Setup method for missing data using a median imputer for important numeric features

num_simple_imputer = SimpleImputer(strategy='median')

numeric_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch']

num_transformed = pd.DataFrame(num_simple_imputer.fit_transform(X_full[numeric_features]))

num_transformed.columns = numeric_features



# Setup one hot enoding for catagorical features

cat_simple_imputer = SimpleImputer(strategy='constant', fill_value='missing')

categorical_features = ['Embarked','Sex', 'Label']

cat_transformed = pd.DataFrame(cat_simple_imputer.fit_transform(X_full[categorical_features]))

cat_transformed.columns = categorical_features

X_dummies = pd.get_dummies(cat_transformed, columns = categorical_features)

X_full = pd.concat([num_transformed, X_dummies], axis = 1)



print(X_full.dtypes)

print(X_full.head())
import seaborn as sns

corr = X_full.corr()

sns.heatmap(corr, cmap = sns.color_palette("coolwarm", 10))
# Split your data

X = X_full[X_full['Label_train'] == 1].copy()

X_test = X_full[X_full['Label_test'] == 1].copy()



# Drop your labels

X.drop('Label_train', axis=1, inplace=True)

X.drop('Label_test', axis=1, inplace=True)

X_test.drop('Label_test', axis=1, inplace=True)

X_test.drop('Label_train', axis=1, inplace=True)

y = train.Survived
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.7, test_size=0.3,

                                                                random_state=0)
from numpy import sort

from sklearn.metrics import accuracy_score

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import mean_absolute_error

from xgboost import XGBClassifier



# Initial model

model = XGBClassifier(random_state = 18)

model.fit(X_train, y_train)



y_pred = model.predict(X_valid)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_valid, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))



# Fit model using each importance as a threshold

thresholds = sort(model.feature_importances_)

models = []

for thresh in thresholds:

    # Select features using threshold

    selection = SelectFromModel(model, threshold=thresh, prefit=True)

    select_X_train = selection.transform(X_train)

    # Train model

    selection_model = XGBClassifier()

    selection_model.fit(select_X_train, y_train)

    models.append([selection_model, selection])

    # Eval model

    select_X_valid = selection.transform(X_valid)

    select_y_pred = selection_model.predict(select_X_valid)

    predictions = [round(value) for value in select_y_pred]

    accuracy = accuracy_score(y_valid, predictions)

    print("Thresh=%.3f, n=%d, Accuracy: %.2f%%" % (thresh, select_X_train.shape[1], accuracy*100.0))

    print("MAE: ", mean_absolute_error(y_valid, predictions))

final_model = models[3][0]

final_selection = models[3][1]

final_X_train = final_selection.transform(X_train)



final_X_test = final_selection.transform(X_test)

final_X_valid = final_selection.transform(X_valid)



final_y_pred = final_model.predict(final_X_valid)

final_predictions = [round(value) for value in final_y_pred]



# Print evaluation metrics

accuracy = accuracy_score(y_valid, final_predictions)

print("n=%d, Accuracy: %.2f%%" % (final_X_train.shape[1], accuracy*100.0))

print("MAE: ", mean_absolute_error(y_valid, final_predictions))
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV



# Set up GridSearchCV in order to determine the best parameters for a gradient boosting model

grid_param = {  

    'n_estimators': [12, 25, 50, 75],

    'max_depth': [3, 4, 5],

    'learning_rate': [0.01, 0.05, 0.1],

    'early_stopping_rounds': [3, 4, 5, 6]

    }



gd_sr = GridSearchCV(estimator = final_model, param_grid = grid_param, 

                     cv = 3, n_jobs = -1, verbose = 2)



gd_sr.fit(X_train, y_train)  

best_parameters = gd_sr.best_params_

print(best_parameters)
selection = SelectFromModel(model, threshold=0.027, prefit=True)

select_X_train = selection.transform(X_train)

select_X_test = selection.transform(X_test)

select_X_valid = selection.transform(X_valid)

another_model = XGBClassifier(early_stopping_rounds=3, learning_rate=0.01, max_depth=5, n_estimators=75)

another_model.fit(select_X_train, y_train)



select_y_pred = another_model.predict(select_X_valid)

select_predictions = [round(value) for value in select_y_pred]



# Print evaluation metrics

accuracy = accuracy_score(y_valid, select_predictions)

print("n=%d, Accuracy: %.2f%%" % (select_X_train.shape[1], accuracy*100.0))

print("MAE: ", mean_absolute_error(y_valid, select_predictions))
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred_rf = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
# Form confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(y_valid, final_y_pred)
final_preds = random_forest.predict(X_test)



# Save test predictions to file

output = pd.DataFrame({'PassengerId':test.PassengerId,'Survived': final_preds})

print(output.head(15))

output.to_csv('submission.csv', index=False)