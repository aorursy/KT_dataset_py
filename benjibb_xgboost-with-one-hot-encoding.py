import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
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
my_imputer = Imputer()
new_X = my_imputer.fit_transform(X)
train_X, test_X, train_y, test_y = train_test_split(new_X, y)
my_model = XGBRegressor(n_estimators = 2380, learning_rate=0.02)
my_model.fit(train_X, train_y, early_stopping_rounds=30, eval_set=[(test_X, test_y)], verbose=False)

predictions = my_model.predict(test_X)
print("Testing Mean Absolute Error : " + str(mean_squared_error(predictions, test_y)))
#def get_mse(n_estimators):
#    my_model = XGBRegressor(n_estimators = n_estimators, learning_rate=0.02)
#    my_model.fit(train_X, train_y, early_stopping_rounds=30, eval_set=[(test_X, test_y)], verbose=False)
#    predictions = my_model.predict(test_X)
#    return mean_squared_error(predictions, test_y)

#def n_estimators_tester(start, end, step):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    saver = []
#    for i in range(start, end, step):
#        saver_value = get_mse(i)
#        print ("n_estimators = %d, mse = %s" % (i, saver_value))
#        saver.append(saver_value)
#    plt.plot(saver)

    # 3000 def get_mse(n_estimators):
#    my_model = XGBRegressor(n_estimators = n_estimators, learning_rate=0.02)
#    my_model.fit(train_X, train_y, early_stopping_rounds=30, eval_set=[(test_X, test_y)], verbose=False)
#    predictions = my_model.predict(test_X)
#    return mean_squared_error(predictions, test_y)

#def n_estimators_tester(start, end, step):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    saver = []
#    for i in range(start, end, step):
#        saver_value = get_mse(i)
#        print ("n_estimators = %d, mse = %s" % (i, saver_value))
#        saver.append(saver_value)
#    plt.plot(saver)

    # 2400 771847772.5037601
#n_estimators_tester(2380, 2490, 5)
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
missing_cols = set(X.columns)  - set( rtest.columns )
for c in missing_cols:
    rtest[c] = 0
rtest = rtest[X.columns]

my_imputer = Imputer()
rtest = my_imputer.fit_transform(rtest)

predicted_prices = my_model.predict(rtest)
# Submission 
my_submission = pd.DataFrame({'Id': rtest2.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)


