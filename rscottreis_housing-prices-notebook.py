!pip install pycaret
from pycaret.regression import *

import pandas as pd



train = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv', index_col=0)

test = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv', index_col=0)

full_data = pd.concat([train.drop('SalePrice', axis=1), test])



train.head()
CATEGORICAL_FEATURES = [

    'LotConfig',

    'Neighborhood',

    'HouseStyle',

    'Exterior1st',

    'Exterior2nd',

    'Foundation',

    'has_garage',

    'has_pool',

]



# ORDINAL_FEATURES = {

#     'OverallQual': ['1','2','3','4','5','6','7','8','9','10'],

#     'OverallCond': ['1','2','3','4','5','6','7','8','9','10'],

# }



NUMERIC_FEATURES = [

    'LotArea',

    'BedroomAbvGr',

    'total_sf', # 1stFlrSF + 2ndFlrSF

    'bathrooms', # FullBath + (HalfBath * 0.5)

    'age',

    # 'remodel_age',

]



# ALL_FEATURES = CATEGORICAL_FEATURES + list(ORDINAL_FEATURES.keys()) + NUMERIC_FEATURES

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
def preprocess(df):

    df['has_garage'] = pd.notna(df['GarageType'])

    df['has_pool'] = pd.notna(df['PoolQC'])

    df['total_sf'] = df['1stFlrSF'] + df['2ndFlrSF']

    df['bathrooms'] = df['FullBath'] + (df['HalfBath'] * 0.5)

    df['age'] = 2020 - df['YearBuilt']

    return df[ALL_FEATURES]



p_train = preprocess(train).join(train.SalePrice)

p_test = preprocess(test)
env = setup(data = p_train,

            target='SalePrice', 

            categorical_features = CATEGORICAL_FEATURES,

            # ordinal_features = ORDINAL_FEATURES,

            numeric_features = NUMERIC_FEATURES,

            normalize = True,

            remove_outliers = True,

            silent = True)
# compare_models(turbo=True) # easy pick: catboost
cb = create_model('catboost')

tuned_cb = tune_model('catboost', optimize='mae')
predictions = predict_model(tuned_cb, data=p_test)

submission = pd.concat([pd.Series(p_test.index), predictions['Label']], axis=1)

submission.columns = ['Id', 'SalePrice']

submission.to_csv('submission.csv',index=False)

submission.head()