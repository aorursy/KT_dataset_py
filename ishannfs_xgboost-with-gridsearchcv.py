import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV



# Read the data

X = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice              

X.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 

                        X_train_full[cname].dtype == "object"]



# Select numeric columns

numerical_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]



# Select categorical columns

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Keep selected columns only

my_cols = low_cardinality_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()



from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='constant')



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

from xgboost import XGBRegressor



# Define the model

xgb = XGBRegressor()



param_grid = {'n_estimators' : [50, 100, 200, 500, 1000],

              'learning_rate' : [0.3, 0.1, 0.033, 0.011, 0.003, 0.001]

             }



better_xgb = GridSearchCV(xgb, param_grid, cv = 5, verbose = 5)



# Pipeline the model with preprocessing

improved_xgb = Pipeline(steps = [('preprocessor', preprocessor),

                          ('model', better_xgb)])



# fit the model

improved_xgb.fit(X_train, y_train)



pred_imp_xgb = improved_xgb.predict(X_valid)

mae_imp_xgb = mean_absolute_error(pred_imp_xgb, y_valid)



print("Mean Absolute Error with default XGB : 17662.736729452055")

print("MAE with GridSearchCV improved XGB : ", mae_imp_xgb)
imp_xgb_preds = improved_xgb.predict(X_test)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': imp_xgb_preds})

output.to_csv('submission.csv', index=False)