import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn import metrics

from sklearn.metrics import classification_report, confusion_matrix
file_path = '../input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv'

df = pd.read_csv(file_path)

print(df.shape)

df.head()
# Remove rows with missing target, separate target from predictors

df.dropna(axis=0, subset=['Survived'], inplace=True)

y = df['Survived']

X = df.drop(['Survived'], axis=1)





# Break off validation set from training data.

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=1)
# Number of missing values in each column of training data

missing_val_count_by_column = (X_train_full.isnull().sum())

print(missing_val_count_by_column[missing_val_count_by_column > 0])
# All categorical columns

object_cols = [col for col in X_train_full.columns if X_train_full[col].dtype == "object"]



# Columns that can be safely label encoded

good_label_cols = [col for col in object_cols if 

                   set(X_train_full[col]) == set(X_valid_full[col])]

        

# Problematic columns that will be dropped from the dataset

bad_label_cols = list(set(object_cols)-set(good_label_cols))

        

print('Categorical columns that will be label encoded:', good_label_cols)

print('\nCategorical columns that could be dropped from the dataset:', bad_label_cols)
X_train = X_train_full.drop(['Firstname', 'Lastname'], axis=1)

X_valid = X_valid_full.drop(['Firstname', 'Lastname'], axis=1)
training_countries = X_train['Country'].unique()

valid_countries = X_valid['Country'].unique()

diff_countries = list(set(training_countries) - set(valid_countries))

print("Unique countries in training set:")

print(training_countries)

print("Unique countries in validation set:")

print(valid_countries)

print("Countries in one and not in the other:")

print(diff_countries)
categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]



X_train.drop(['PassengerId'], axis=1, inplace=True)

X_valid.drop(['PassengerId'], axis=1, inplace=True)
y_baseline = np.zeros(len(y_valid))

print('Baseline MAE:', mean_absolute_error(y_valid, y_baseline))
# Preprocessing for categorical data. 

# Set handle_unknown='ignore' so that new categories are set to zeros.

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

rf_model_1 = RandomForestClassifier(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

rf_pipeline_1 = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', rf_model_1)

                     ])



# Preprocessing of training data, fit model 

rf_pipeline_1.fit(X_train, y_train.values.ravel())



# Preprocessing of validation data, get predictions

preds_1 = rf_pipeline_1.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds_1))
X.head()

X_proc = X.drop(['Firstname','Lastname'], axis=1)
# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(rf_pipeline_1, X_proc, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())
def get_score_rf(n_estimators):

    """Return the average MAE over 3 CV folds of random forest model.

    

    Keyword argument:

    n_estimators -- the number of trees in the forest

    """

    # Preprocessing for categorical data

    categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

    ])



    # Bundle preprocessing for numerical and categorical data

    preprocessor = ColumnTransformer(

    transformers=[

        ('cat', categorical_transformer, categorical_cols)

    ])



    my_pipeline = Pipeline(steps=[

                    ('preprocessor', preprocessor),

                    ('model', RandomForestClassifier(n_estimators=n_estimators, random_state=0))

                    ])

    scores = -1 * cross_val_score(my_pipeline, X_proc, y,

                              cv=5,

                              scoring='neg_mean_absolute_error')

    return scores.mean()
results = {}

for i in range(1,9):

    results[i*50] = get_score_rf(i*50)

print(results)
rf_model_final = RandomForestClassifier(n_estimators=50, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

rf_pipeline_final = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', rf_model_final)

                     ])



# Preprocessing of training data, fit model 

rf_pipeline_final.fit(X_train, y_train.values.ravel())



# Preprocessing of validation data, get predictions

preds_rf_final = rf_pipeline_final.predict(X_valid)



print('MAE:', mean_absolute_error(y_valid, preds_rf_final))
# # Preprocessing for categorical data. 

# # Set handle_unknown='ignore' so that new categories are set to zeros.

# categorical_transformer = Pipeline(steps=[

#     ('imputer', SimpleImputer(strategy='most_frequent')),

#     ('onehot', OneHotEncoder(handle_unknown='ignore'))

# ])



# # Bundle preprocessing for numerical and categorical data

# preprocessor = ColumnTransformer(

#     transformers=[

#         ('cat', categorical_transformer, categorical_cols)

#     ])



# # Define model

# xgb_model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, n_jobs=3)



# # Bundle preprocessing and modeling code in a pipeline

# xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

#                       ('model', xgb_model)

#                      ])



# # Preprocessing of training data, fit model 

# xgb_pipeline.fit(X_train, y_train.values.ravel(), model__early_stopping_rounds=5, 

#                  model__eval_set=[(X_valid, y_valid)], model__verbose=False)



# # Preprocessing of validation data, get predictions

# preds_xgb = xgb_pipeline.predict(X_valid)



# print('MAE:', mean_absolute_error(y_valid, preds_xgb))





#xgb_model.fit(X_train, y_train, 

#             early_stopping_rounds=5, 

#             eval_set=[(X_valid, y_valid)], 

#             verbose=False)
# One-hot encode the data

X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)



# Fill any NaN columns (should only happen with country) with 0s

X_train.fillna(0, inplace=True)

X_valid.fillna(0, inplace=True)
# Define the model

xgb_model = xgb.XGBClassifier(random_state=1, learning_rate = 0.05, n_estimators=1000, n_jobs=3)



# Fit the model

xgb_model.fit(X_train,y_train, early_stopping_rounds=5, 

             eval_set=[(X_valid, y_valid)])



# Get predictions

preds_xgb = xgb_model.predict(X_valid) # Your code here



# Calculate MAE

mae_xgb = mean_absolute_error(y_valid, preds_xgb) # Your code here



print("Mean Absolute Error:" , mae_xgb)
