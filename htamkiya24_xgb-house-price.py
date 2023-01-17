import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
# Read the data
X_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col='Id')

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
categorical_cols = [col_name for col_name in X_train_full.columns if
                    X_train_full[col_name].nunique() < 10 and 
                    X_train_full[col_name].dtype == "object"]

# Select numerical columns
numerical_cols = [col_name for col_name in X_train_full.columns if 
                X_train_full[col_name].dtype in ['int64', 'float64']]


y_train.head()
my_columns = categorical_cols + numerical_cols
X_train = X_train_full[my_columns].copy()
X_valid = X_valid_full[my_columns].copy()
X_test = X_test_full[my_columns].copy()
# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='median')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
from xgboost import XGBRegressor as xgbr
model = xgbr(random_state=42,n_estimators=500,learning_rate=0.09)
# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)])
clf.fit(X_train, y_train)
preds=clf.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, preds))
# preds_test = clf.predict(X_test) # Your code here

# # Save test predictions to file
# output = pd.DataFrame({'Id': X_test.index,
#                        'SalePrice': preds_test})
# output.to_csv('submission1.csv', index=False)
