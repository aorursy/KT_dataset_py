# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv", index_col='Id')

test_df = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv", index_col='Id')

train_df.shape, test_df.shape
train_df.dropna(axis=0, subset=["SalePrice"], inplace=True)

y = train_df.SalePrice

train_df.drop("SalePrice", axis=1, inplace = True)
categorical_cols = [cname for cname in train_df.columns if

                    train_df[cname].nunique() < 10 and 

                    train_df[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in train_df.columns if 

                  train_df[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



def get_score(model, X_train, X_test, y_train, y_test):

    model.fit(X_train, y_train)

    mae = mean_absolute_error(y_test, model.predict(X_test))

    print(f"MAE: {mae}")

    return mae
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                      ])



X_train, X_test, y_train, y_test = train_test_split(train_df[my_cols], y, test_size=0.3, random_state=0)

onehot_mae = get_score(pipe, X_train, X_test, y_train, y_test)
from category_encoders.count import CountEncoder





# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('ce', CountEncoder())

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                      ])



X_train, X_test, y_train, y_test = train_test_split(train_df[my_cols], y, test_size=0.3, random_state=0)

ce_mae = get_score(pipe, X_train, X_test, y_train, y_test)
from category_encoders.target_encoder import TargetEncoder

from sklearn.preprocessing import LabelEncoder



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('te', TargetEncoder()),

    ('fillna', SimpleImputer(strategy='constant', fill_value=0))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                      ])



X_train, X_test, y_train, y_test = train_test_split(train_df[my_cols], y, test_size=0.3, random_state=0)

te_mae = get_score(pipe, X_train, X_test, y_train, y_test)
from category_encoders import CatBoostEncoder



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('cbe', CatBoostEncoder()),

    ('fillna', SimpleImputer(strategy='constant', fill_value=0))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols),

    ])



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor),

                       ('model', model)

                      ])



X_train, X_test, y_train, y_test = train_test_split(train_df[my_cols], y, test_size=0.3, random_state=0)

cbe_mae = get_score(pipe, X_train, X_test, y_train, y_test)
cat_encoders = pd.DataFrame([onehot_mae, ce_mae, te_mae, cbe_mae],

                            columns=["MAE"], index=["One Hot Encoder", "Count Encoder", "Target Encoder", "Cat Boost Encoder"])

cat_encoders.sort_values(by="MAE")
for f1 in categorical_cols:

    for f2 in categorical_cols:

        if f1 == f2:

            continue

        f_name = f1 + "_" + f2

        train_df[f_name] = train_df[f1] + train_df[f2]

train_df.columns
for f1 in categorical_cols:

    for f2 in categorical_cols:

        if f1 == f2:

            continue

        f_name = f1 + "_" + f2

        test_df[f_name] = test_df[f1] + test_df[f2]
categorical_cols = [cname for cname in train_df.columns if

                    train_df[cname].nunique() < 10 and 

                    train_df[cname].dtype == "object"]

# Select numerical columns

numerical_cols = [cname for cname in train_df.columns if 

                  train_df[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols
# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='mean')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])
train = pd.DataFrame(preprocessor.fit_transform(train_df, y).toarray())

feature_cols = numerical_cols + list(preprocessor.named_transformers_["cat"].named_steps['onehot'].get_feature_names())

train.columns = feature_cols

train.index = train_df.index

train.head()
from sklearn.feature_selection import SelectKBest, f_classif



# best_model = None

# best_score = float("inf")

# best_k = -1

# best_cols = -1

# k_list = list(range(50, len(train.columns), 100))

# for k in k_list:

#     # Create the selector, keeping k features

#     selector = SelectKBest(f_classif, k=k)



#     # Use the selector to retrieve the best features

#     X_new = selector.fit_transform(train, y)



#     # Get back the kept features as a DataFrame with dropped columns as all 0s

#     selected_features = pd.DataFrame(selector.inverse_transform(X_new),

#                                      index=train_df.index,

#                                      columns=feature_cols)

#     selected_columns = list(selected_features.columns[selected_features.var() != 0])



#     # Find the columns that were dropped

#     dropped_columns = list(set(feature_cols) - set(selected_columns))

#     model = RandomForestRegressor(n_estimators=100, random_state=0)

#     X_train, X_test, y_train, y_test = train_test_split(train[selected_columns], y, test_size=0.3, random_state=0)

#     score = get_score(model, X_train, X_test, y_train, y_test)

#     if score < best_score:

#         best_score = score

#         best_model = model

#         best_k = k

#         best_cols = selected_columns
# print("Best k:", best_k, "\nBest score:", best_score)

print("Best k:", 1450)
# Create the selector, keeping k features

selector = SelectKBest(f_classif, k=1450)



# Use the selector to retrieve the best features

X_new = selector.fit_transform(train, y)



# Get back the kept features as a DataFrame with dropped columns as all 0s

selected_features = pd.DataFrame(selector.inverse_transform(X_new),

                                 index=train_df.index,

                                 columns=feature_cols)

best_cols = list(selected_features.columns[selected_features.var() != 0])



# Find the columns that were dropped

dropped_columns = list(set(feature_cols) - set(best_cols))

final_model = RandomForestRegressor(n_estimators=100, random_state=0)

X_train, X_test, y_train, y_test = train_test_split(train[best_cols], y, test_size=0.3, random_state=0)

best_score = get_score(final_model, X_train, X_test, y_train, y_test)
from xgboost import XGBRegressor



# n_estimators_list = [100, 300, 500, 700, 900, 1000, 1200]

# learning_rate_list = [0.01, 0.05, 0.1, 0.3]



# best_score = float("inf")

# best_model = -1

# best_ne = -1

# best_lr = -1

# for n_estimators in n_estimators_list:

#     print(f"\nn_estimators: {n_estimators}\n")

#     for learning_rate in learning_rate_list:

#         print(f"learning_rate: {learning_rate}", end=" ")

#         X_train, X_valid, y_train, y_valid = train_test_split(train[best_cols], y, test_size=0.3, random_state=0)

#         my_model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=4)

#         my_model.fit(X_train, y_train, 

#                      early_stopping_rounds=5, 

#                      eval_set=[(X_valid, y_valid)], 

#                      verbose=False)

#         score = get_score(my_model, X_train, X_valid, y_train, y_valid)

#         if score < best_score:

#             best_model = my_model

#             best_ne = n_estimators

#             best_lr = learning_rate

#             best_score = score
print("""

Output:

n_estimators: 100



learning_rate: 0.01 MAE: 68736.31545198345

learning_rate: 0.05 MAE: 17363.63460509418

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 300



learning_rate: 0.01 MAE: 19010.113691495433

learning_rate: 0.05 MAE: 17001.838800299658

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 500



learning_rate: 0.01 MAE: 17035.824727097603

learning_rate: 0.05 MAE: 16952.959492722603

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 700



learning_rate: 0.01 MAE: 16900.231583547375

learning_rate: 0.05 MAE: 16952.959492722603

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 900



learning_rate: 0.01 MAE: 16813.063802083332

learning_rate: 0.05 MAE: 16952.959492722603

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 1000



learning_rate: 0.01 MAE: 16795.14985552226

learning_rate: 0.05 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726



n_estimators: 1200



learning_rate: 0.01 MAE: 16762.28935680651

learning_rate: 0.05 MAE: 16952.959492722603

learning_rate: 0.1 MAE: 16979.821623501713

learning_rate: 0.3 MAE: 18756.26118364726

""")
best_lr = 0.01

best_ne = 1200

X_train, X_valid, y_train, y_valid = train_test_split(train[best_cols], y, test_size=0.3, random_state=0)

final_model = XGBRegressor(n_estimators=best_ne, learning_rate=best_lr, n_jobs=4)

final_model.fit(X_train, y_train, 

                early_stopping_rounds=5, 

                eval_set=[(X_valid, y_valid)], 

                verbose=False)
test = pd.DataFrame(preprocessor.transform(test_df).toarray())

feature_cols = numerical_cols + list(preprocessor.named_transformers_["cat"].named_steps['onehot'].get_feature_names())

test.columns = feature_cols

test.index = test_df.index

test.head()
preds = final_model.predict(test[best_cols])

submission = pd.DataFrame({"Id": test.index,

                           "SalePrice": preds})

submission.to_csv('submission.csv', index=False)