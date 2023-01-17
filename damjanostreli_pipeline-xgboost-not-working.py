import pandas as pd

import numpy as np



from sklearn.model_selection import train_test_split



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_absolute_error



from xgboost import XGBClassifier



#import data

X_train_full = pd.read_csv("../input/titanic/train.csv")

Test_data = pd.read_csv("../input/titanic/test.csv")
#inspect the data



#X_train_full.head()

X_train_full.describe()
# Remove rows with missing target - in this case there are none

# set target and drop target from predictors



y = X_train_full["Survived"]

X_train_full.drop(["Survived"], axis = 1, inplace = True)
# Break off validation set from training data

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y, train_size=0.8, test_size=0.2)
#DEAL WITH MISSING DATA

#get names of columns with missing values



col_with_missing = [col for col in X_train_full.columns if X_train_full[col].isnull().any()]

print(col_with_missing)
#how many values are missing in each column?

missing_values_count_by_column = X_train_full.isnull().sum()

print(missing_values_count_by_column)
#get numerical and categorical columns

all_columns = X_train_full.columns

print(all_columns)



numerical_columns = [col for col in X_train_full.columns if X_train_full[col].dtype in ["int64", "float64"]]

print("Numerical columns: ", numerical_columns)

categorical_columns = [col for col in X_train_full.columns if X_train_full[col].dtype in ["object"]]

print("Categorical columns: ", categorical_columns)



#check if added columns are all columns

len(all_columns) == len(categorical_columns + numerical_columns)
#check cardinality of categorical columns



low_cardinality_cols = [col for col in X_train_full[categorical_columns] if X_train_full[col].nunique() < 10 and 

                        X_train_full[col].dtype == "object"]

print(low_cardinality_cols)



high_cardinality_cols = [col for col in X_train_full[categorical_columns] if X_train_full[col].nunique() >= 10 and 

                        X_train_full[col].dtype == "object"]



len(categorical_columns) == len(low_cardinality_cols + high_cardinality_cols)
X_train_full[categorical_columns].nunique()
#define numerical and categorical transformers (preprocessing)

#remember to change strategies later to see which one performs best



numerical_transformer = SimpleImputer(strategy = "constant")



low_categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "most_frequent")), 

                                               ("OH_encoder", OneHotEncoder(handle_unknown = "ignore"))])



high_categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy = "most_frequent")),

                                                 ("labeler", LabelEncoder())])



# #bundle them together in one preprocessor via ColumnTransformer: (syntax like pipeline but don't forget to add columns)



preprocessor = ColumnTransformer(transformers = [("num", numerical_transformer, numerical_columns),

                                                 ("low_cat", low_categorical_transformer, low_cardinality_cols)])



preprocessor.fit_transform(X_train)

preprocessor.transform(X_train)
#Drop some data

X_train.drop(["Name", "Cabin", "Ticket"], axis = 1, inplace = True)

Test_data.drop(["Name", "Cabin", "Ticket"], axis = 1, inplace = True)
#Fit preprocessor

preprocessor.fit_transform(X_train)

preprocessor.transform(Test_data)
#define the model



my_model = XGBClassifier(n_estimators=300, learning_rate=0.01, n_jobs=6, early_stopping_rounds = 5)

#pipeline preprocessor and model:



my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', my_model)

                             ])
my_pipeline.fit(X_train, y_train)
my_model = XGBClassifier(n_estimators=350, learning_rate=0.005, n_jobs=6, early_stopping_rounds = 5)

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                              ('model', my_model)

                             ])

my_pipeline.fit(X_train, y_train)

preds = my_pipeline.predict(X_valid)

score = mean_absolute_error(preds, y_valid)

print("MAE: ", score)
predictions = my_pipeline.predict(Test_data)

output = pd.DataFrame({'PassengerId': Test_data.PassengerId, 'Survived': predictions})

output.to_csv('XGBoost_submission.csv', index=False)

print("Your submission was successfully saved!")