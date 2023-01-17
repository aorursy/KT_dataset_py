# All the imports:



# to work with data frames 

import pandas as pd



# to preprocess the data

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



# for modelling

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor



# for the metrics

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score



# to read the data

X_test = pd.read_csv('../input/test.csv', index_col='Id')

train_data = pd.read_csv('../input/train.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

y_train = train_data.SalePrice

X_train = train_data.drop(['SalePrice'], axis=1)
# Select categorical columns

categorical_cols = [col for col in X_train.columns if

                    X_train[col].nunique() < 10 and 

                    X_train[col].dtype == "object"]



# Select numerical columns

numerical_cols = [col for col in X_train.columns if 

                X_train[col].dtype in ['int64', 'float64']]



my_cols = categorical_cols + numerical_cols

X_train = X_train[my_cols].copy()



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')

                  

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

model = XGBRegressor(n_estimators = 1300, learning_rate=0.04, random_state = 1)



# Bundle preprocessing and modeling code in a pipeline

pipe = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Fit the model 

pipe.fit(X_train, y_train)



# Use cross-validation to compute the average mae score

score = cross_val_score(pipe, X_train, y_train, scoring = "neg_mean_absolute_error", cv = 4)

print(score)

print("Mean score: %d" %(-1 * score.mean()))
# Preprocessing of test data

X_test = X_test[my_cols].copy()

# Predict on the test data

preds_test = pipe.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)