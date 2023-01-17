import pandas as pd



# Load data

melb_data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')



from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

y = train.SalePrice

features_all = train.drop(['SalePrice'], axis = 1)

X = features_all.select_dtypes(exclude = ['object'])
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor



train_X, test_X, train_y, test_y = train_test_split(X, y, train_size = 0.7, test_size = 0.3,random_state = 5)



def mae_score(train_X, test_X, train_y, test_y):

    model = RandomForestRegressor()

    model.fit(train_X, train_y)

    predicted_y = model.predict(test_X)

    mae = mean_absolute_error(test_y, predicted_y)

    return mae
col_with_missed_val = [col for col in X.columns 

                                if X[col].isnull().any()]

train_X_no_missed = train_X.drop(col_with_missed_val, axis = 1)

test_X_no_missed = test_X.drop(col_with_missed_val, axis = 1)

mae_score(train_X_no_missed, test_X_no_missed, train_y, test_y)
from sklearn.impute import SimpleImputer



train_X_copy = train_X.copy()

test_X_copy = test_X.copy()



impute = SimpleImputer()

train_X_imputed = impute.fit_transform(train_X_copy)

test_X_imputed = impute.transform(test_X_copy)



mae_score(train_X_imputed, test_X_imputed, train_y, test_y)

from sklearn.impute import SimpleImputer



train_X_copy = train_X.copy()

test_X_copy = test_X.copy()



col_with_missed_val = [col for col in X.columns

                                if X[col].isnull().any()]

for col in col_with_missed_val:

    train_X_copy[col + '_was_missed'] = train_X_copy[col].isnull()

    test_X_copy[col + '_was_missed'] = test_X_copy[col].isnull()



impute = SimpleImputer()

train_X_imputed = impute.fit_transform(train_X_copy)

test_X_imputed = impute.transform(test_X_copy)

mae_score(train_X_imputed, test_X_imputed, train_y, test_y)