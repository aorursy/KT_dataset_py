import pandas as pd
from sklearn.model_selection import train_test_split

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


# threshold value was set. Columns with data less than the threshold value were dropped.

thres = int(0.8*len(X_train_full)) # 934

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality and less missing values
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object" and
                    X_train_full[cname].notna().sum() > thres]

# Selecting numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64'] and
                X_train_full[cname].notna().sum() > thres]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

print('Number of selected numerical columns: ',len(numerical_cols))
print('Number of selected categorical columns: ',len(categorical_cols))
# identifying the columns with missing values. These missing data in the below columns will be imputed with values

emp_cols_dict = {col:X_train[col].isnull().sum() for col in X_train.columns if X_train[col].isnull().any()}
emp_cols_dict
# importing the necessary libraries for preprocessing, building a pipeline.

# Hyperparameter Tuning for the SimpleImputer strategy with Median yielded better results

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# Preprocessing for numerical data using SimpleImputer()
numerical_transformer = SimpleImputer(strategy='median') 

# Preprocessing for categorical data using SimpleImputer() and OneHotEncoder()
categorical_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data using a ColumnTransformer()
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])



# Define model

# Using the XG boost regressor model to estimate the model parameter and make predictions 
# I experimented with the values of n_estimators ranging from 100-1000 and 
# learning_rate values from 0.1-0.09 in order to identify the best accuracy

model = XGBRegressor(n_estimators=750, learning_rate = 0.06,random_state=0)


# Bundle the preprocessing and modeling code in a pipeline
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)
                             ])

# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)

# Evaluate the model
score = mean_absolute_error(y_valid, preds)
print('MAE:', score)


# Preprocessing of test data, fit model
preds_test = my_pipeline.predict(X_test) # Your code here


# Save test predictions to file
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)