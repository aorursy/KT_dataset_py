import pandas as pd



train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')



# use all data for fitting one hot encoder and imputing

all_data = pd.concat([train_data.drop(columns=['SalePrice']), test_data])



# target

y = train_data.SalePrice
import numpy as np



from sklearn.base import TransformerMixin



class DataFrameImputer(TransformerMixin):



    def __init__(self):

        """Impute missing values.



        Columns of dtype object are imputed with the most frequent value 

        in column.



        Columns of other types are imputed with mean of column.



        """

    def fit(self, X, y=None):



        self.fill = pd.Series([X[c].value_counts().index[0]

            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],

            index=X.columns)



        return self



    def transform(self, X, y=None):

        return X.fillna(self.fill)
imputer = DataFrameImputer().fit(all_data)

train_data = imputer.transform(train_data)

test_data = imputer.transform(test_data)

all_data = imputer.transform(all_data)
all_data.describe()
train_data.columns
train_data.groupby('LotFrontage').size()
#train_data.describe()
categorical = [

    'LotConfig',

    'Neighborhood',

    'BldgType',

    'HouseStyle',

    'ExterQual',

    'ExterCond',

    'HeatingQC',

    'CentralAir',

    'KitchenQual',

    'GarageType',

    'Exterior1st',

    'Functional',

    'MSZoning',

    'BsmtCond',

    'GarageQual',

    'GarageFinish',

    'Foundation'

]
from sklearn.preprocessing import OneHotEncoder



one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

one_hot_encoder.fit(all_data[categorical])
def feature_engineering(data):

    

    features = [

        'LotArea',

        #'LotFrontage',

        'YearBuilt',

        '1stFlrSF',

        '2ndFlrSF',

        'FullBath',

        'HalfBath',

        'BsmtFullBath',

        'BsmtHalfBath',

        'BedroomAbvGr',

        'GrLivArea',

        'TotRmsAbvGrd',

        'GarageArea',

        'GarageCars',

        'PoolArea',

        'BsmtFinSF1',

        'MiscVal',

        'OverallCond',

        'OverallQual',

        'KitchenAbvGr',

        'Fireplaces',

        'YearRemodAdd',

        #'WoodDeckSF'

    ]



    X = data[features]

    total_bath = X.FullBath + X.BsmtFullBath + 0.5 * (X.HalfBath + X.BsmtHalfBath)

    d = {

        'TotalBath': total_bath,

        'TotalSqFt': X['1stFlrSF'] + X['2ndFlrSF'] + X.BsmtFinSF1,

        'BedToBath': X.BedroomAbvGr / total_bath

    }

    X = pd.concat([X, pd.DataFrame(d)], axis=1)

    

    return pd.concat([X, pd.DataFrame(one_hot_encoder.transform(data[categorical]))], axis=1)
#feature_engineering(train_data)
#help(RandomForestRegressor)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(feature_engineering(train_data), y, random_state=1)



# Define the model. Set random_state to 1

for i in [1]:

    rf_model = RandomForestRegressor(n_estimators=700, max_depth=30, random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
rf_model_on_full_data = RandomForestRegressor(n_estimators=700, max_depth=30)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(feature_engineering(train_data), y)
X_test = feature_engineering(test_data)

test_preds = rf_model_on_full_data.predict(X_test)



output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)