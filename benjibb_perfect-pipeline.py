import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
# Part 1 Read data
data = pd.read_csv('../input/train.csv')
# if no saleprice, drop the row
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Part 2 Create X, y
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1)

# Part 3 Choose numeric and categorical
# categorical
low_cardinality_cols = [cname for cname in X.columns if 
                                X[cname].nunique() < 10 and
                                X[cname].dtype == "object"]
# numeric
numeric_cols = [cname for cname in X.columns if 
                X[cname].dtype in ['int64', 'float64']]
# combined
my_cols = low_cardinality_cols + numeric_cols
X = X[my_cols]

# One hot encoded
X = pd.get_dummies(X)

# Part 4 Imputation with track what was imputed
# find out missing data column
cols_with_missing = (col for col in X.columns
                    if X[col].isnull().any())

# adding binary column if the column has missing data
for col in cols_with_missing:
    X[col +'_was_missing'] = X[col].isnull()
train_X, test_X, train_y, test_y = train_test_split(X, y)
my_pipeline = make_pipeline(Imputer(), RandomForestRegressor())
my_pipeline.fit(train_X, train_y)
predictions = my_pipeline.predict(test_X)
mean_absolute_error(predictions, test_y)
# Evaluate test set
rtest = pd.read_csv('../input/test.csv')
rtest2 = pd.read_csv('../input/test.csv')
# Part 3 Choose numeric and categorical
# categorical
rlow_cardinality_cols = [cname for cname in rtest.columns if 
                                rtest[cname].nunique() < 10 and
                                rtest[cname].dtype == "object"]
# numeric
rnumeric_cols = [cname for cname in rtest.columns if 
                rtest[cname].dtype in ['int64', 'float64']]
# combined
rmy_cols = rlow_cardinality_cols + rnumeric_cols
rtest = rtest[rmy_cols]

# One hot encoded
rtest = pd.get_dummies(rtest)

# Part 4 Imputation with track what was imputed
# find out missing data column
cols_with_missing = (col for col in rtest.columns
                    if rtest[col].isnull().any())

# adding binary column if the column has missing data
for col in cols_with_missing:
    rtest[col +'_was_missing'] = rtest[col].isnull()
    
# Get missing columns and add them to the test dataset
missing_cols = set( X.columns ) - set( rtest.columns )
for c in missing_cols:
    rtest[c] = 0
rtest = rtest[X.columns]

my_imputer = Imputer()
rtest = my_imputer.fit_transform(rtest)

predicted_prices = my_pipeline.predict(rtest)
# Submission 
my_submission = pd.DataFrame({'Id': rtest2.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
