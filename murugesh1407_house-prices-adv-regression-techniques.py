import pandas as pd

from sklearn.model_selection import train_test_split 

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

# Read data

X_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



# Clear rows with missing target values

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)



#Seperate target column from predictors

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



# Train and test data split

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 

                                                                train_size=0.8, test_size=0.2,

                                                                random_state=0)



# Select categorical columns with unique values less than 10

categorical_cols = [cname for cname in X_train_full.columns if

                    X_train_full[cname].nunique() < 10 and 

                    X_train_full[cname].dtype == "object"]



# Select numerical columns

numerical_cols = [cname for cname in X_train_full.columns if 

                X_train_full[cname].dtype in ['int64', 'float64']]



# Keep selected columns only for predictions

my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()

X_valid = X_valid_full[my_cols].copy()

X_test = X_test_full[my_cols].copy()

X_train.head()


from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# Preprocessing for numerical data

numerical_transformer = SimpleImputer(strategy='most_frequent')



# Preprocessing for categorical data

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse = False))

])



# Combined(numerical and categorical) preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



# Define RF model

model = RandomForestRegressor(n_estimators=100, random_state=0)



# Combined preprocessing and modeling code in a pipeline

my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                      ('model', model)

                     ])



# fit preprocessed training data to model 

my_pipeline.fit(X_train, y_train)



# get predictions

preds = my_pipeline.predict(X_valid)



#get mae score

print('MAE:', mean_absolute_error(y_valid, preds))
# fit model with test data

preds_test = my_pipeline.predict(X_test)
# Save test predictions to file

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)