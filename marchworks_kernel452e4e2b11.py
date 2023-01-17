import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from xgboost import XGBRegressor

from sklearn.model_selection import GridSearchCV





# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# "Cardinality" means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
my_imputer = SimpleImputer(strategy='median')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train[numerical_cols]))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid[numerical_cols]))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train[numerical_cols].columns

imputed_X_valid.columns = X_valid[numerical_cols].columns



X_train[numerical_cols] = imputed_X_train[numerical_cols]

X_valid[numerical_cols] = imputed_X_valid[numerical_cols]



my_imputer = SimpleImputer(strategy='constant')

imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train[categorical_cols]))

imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid[categorical_cols]))



# Imputation removed column names; put them back

imputed_X_train.columns = X_train[categorical_cols].columns

imputed_X_valid.columns = X_valid[categorical_cols].columns



X_train[categorical_cols] = imputed_X_train[categorical_cols]

X_valid[categorical_cols] = imputed_X_valid[categorical_cols]



X_train = pd.get_dummies(X_train)

X_valid = pd.get_dummies(X_valid)

X_test = pd.get_dummies(X_test)

X_train, X_valid = X_train.align(X_valid, join='left', axis=1)

X_train, X_test = X_train.align(X_test, join='left', axis=1)



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='median')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



my_pipeline = Pipeline(steps=[

                              ('xgbrg', XGBRegressor())

])



param_grid = {

    "xgbrg__n_estimators": [10, 50, 100, 500],

    "xgbrg__learning_rate": [0.1, 0.5, 1],

}



fit_params = {"xgbrg__eval_set": [(X_valid, y_valid)],

              "xgbrg__early_stopping_rounds": 10, 

              "xgbrg__verbose": False} 



searchCV = GridSearchCV(my_pipeline, cv=5, param_grid=param_grid)

searchCV.fit(X_train, y_train, **fit_params)  
print(searchCV.best_params_)
model = XGBRegressor(learning_rate=0.1, n_estimators=50)

model.fit(X_train, y_train)

preds_test = model.predict(X_test)
output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)