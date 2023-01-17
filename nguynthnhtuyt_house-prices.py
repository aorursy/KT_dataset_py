import pandas as pd

from sklearn.model_selection import train_test_split



# Read the data

X_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice 

X_full.drop(['SalePrice'], axis=1, inplace=True)



# To keep things simple, we'll drop columns with missing values

cols_with_missing = [col for col in X_full.columns if X_full[col].isnull().any()]

X_full.drop(cols_with_missing, axis=1, inplace=True)

X_test_full.drop(cols_with_missing, axis=1, inplace=True)



# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, train_size=0.8, test_size=0.2,random_state=1)



# 'Cardinality' means the number of unique values in a column

# Select categorical columns with relatively low cardinality (convenient but arbitrary)

categorical_cols = [cname for cname in X_train_full.columns if 

                    X_train_full[cname].nunique() < 10 and

                    X_train_full[cname].dtype == 'object']



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                  X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

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



# Define model

model = RandomForestRegressor(n_estimators=100, random_state=1)



# Bundle preprocessing and modeling code in a pipeline

clf = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# Preprocessing of training data, fit model

clf.fit(X_train, y_train)



# Preprocessing of validation data, get predictions

preds = clf.predict(X_valid)



print(mean_absolute_error(y_valid, preds))
test_preds = clf.predict(X_test)
output = pd.DataFrame({'Id': X_test.index, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)