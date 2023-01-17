# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.ml_intermediate.ex6 import *
print("Setup Complete")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer 
# Read the data
Xa = pd.read_csv('../input/train.csv', index_col='Id')
X_test_full = pd.read_csv('../input/test.csv', index_col='Id')

print(Xa.shape)
print(X_test_full.shape)

# Remove rows with missing target, separate target from predictors
Xa.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = Xa.SalePrice              
Xa.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(Xa, y, train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
low_cardinality_cols = [cname for cname in X_train_full.columns if X_train_full[cname].nunique() < 10 and 
                        X_train_full[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X_train_full.columns if X_train_full[cname].dtype in ['int64', 'float64']]
# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X=Xa[my_cols].copy()
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()

#dropping the columns 'Alley','PoolQC', 'Fence','MiscFeature' as many missing values(alomst 80%) are missing
X_train.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
X_valid.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
nm=list(set(numeric_cols)-set(['Alley','PoolQC','Fence','MiscFeature']))
mn=list(set(low_cardinality_cols)-set(['Alley','PoolQC','Fence','MiscFeature']))
#X.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
X_test.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)
my_imputer1=SimpleImputer()
my_imputer2=SimpleImputer(strategy='most_frequent')


X_train = pd.get_dummies(X_train)
X_valid = pd.get_dummies(X_valid)
X_test = pd.get_dummies(X_test)
X_trains, X_valids = X_train.align(X_valid, join='left', axis=1)

X_trains, X_tests = X_train.align(X_test, join='left', axis=1)
from xgboost import XGBRegressor
print(len(X_trains.columns))
print(len(X_tests.columns))

print()

# Check your answer
#step_1.a.check()
#%%RM_IF(PROD)%
from sklearn.utils.validation import check_is_fitted
#my_model_1 = XGBRegressor(n_estimators=100,random_state=0)
#my_model_1.fit(X_train,y_train)
#step_1.a.assert_check_failed()
from sklearn.metrics import mean_absolute_error


# Define the model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
my_model_2 = XGBRegressor(n_estimators=1000,random_state=0,n_jobs=5,learning_rate=0.0445)# Your code here

# Fit the model
my_model_2.fit(X_trains,y_train) # Your code here
#my_model_2.fit(X_train,y_train)
preds=my_model_2.predict(X_valids)
print(mean_absolute_error(y_valid,preds))

# Check your answer
#step_2.check()
po=my_model_2.predict(X_tests)
output=pd.DataFrame({'Id':X_tests.index,'SalePrice':po})
output.to_csv('submission.csv',index=False)