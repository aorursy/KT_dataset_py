import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split
# Read the data

X_full = pd.read_csv('../input/train.csv', index_col='Id')

X_test_full = pd.read_csv('../input/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X_full.SalePrice

X_full.drop(['SalePrice'], axis=1, inplace=True)



categorical_cols = [cname for cname in X_full.columns if X_full[cname].dtype == "object"]

numerical_cols = [cname for cname in X_full.columns if X_full[cname].dtype in ['int64', 'float64']]
# Shape

X_full.shape
# Missing values, categorical

mis_cat = X_full[categorical_cols].isnull().sum()

mis_cat[mis_cat > 0]
# Missing values, contineous

mis_num = X_full[numerical_cols].isnull().sum()

mis_num[mis_num > 0]
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# We create an explicit for feature for 'HasShed' from 'MiscFeature'

X_full['Has_shed'] = np.where(X_full['MiscFeature']=='Shed',1,0)

X_full_final = X_full.drop('MiscFeature',axis=1)



# We need a bunch of transformers based on the above

# Preprocessing for numerical data

numerical_transformer_median = SimpleImputer(strategy='median') 



numerical_transformer_constant = SimpleImputer(strategy='constant',fill_value=0) 



# Preprocessing for categorical data

categorical_transformer_constant = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='NotPresentInTheHouse')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



categorical_transformer_mode = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])



categorical_transformer_other = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))

])





# Column categories

num_col_median = ['LotFrontage','GarageYrBlt']



num_col_constant = ['MasVnrArea']



cat_col_constant = (['Alley',

                         'MasVnrType',

                         'BsmtQual',

                         'BsmtCond',

                         'BsmtExposure',

                         'BsmtFinType1',

                         'BsmtFinType2',

                         'FireplaceQu',

                         'GarageType',

                         'GarageFinish',

                         'GarageQual',

                         'GarageCond',

                         'PoolQC',

                         'Fence'])



cat_col_mode = ['Electrical']



cat_col_other = (['MSZoning',

             'Street',

             'LotShape',

             'LandContour',

             'Utilities',

             'LotConfig',

             'LandSlope',

             'Neighborhood',

             'Condition1',

             'Condition2',

             'BldgType',

             'HouseStyle',

             'RoofStyle',

             'RoofMatl',

             'Exterior1st',

             'Exterior2nd',

             'ExterQual',

             'ExterCond',

             'Foundation',

             'Heating',

             'HeatingQC',

             'CentralAir',

             'KitchenQual',

             'Functional',

             'PavedDrive',

             'SaleType',

             'SaleCondition'])
# Bundle preprocessing for numerical and categorical data

preprocessor = ColumnTransformer(

    transformers=[

        ('num_median', numerical_transformer_median, num_col_median),

        ('num_constant', numerical_transformer_constant, num_col_constant),

        ('cat_constant', categorical_transformer_constant, cat_col_constant),

        ('cat_mode', categorical_transformer_mode, cat_col_mode),

        ('cat_other', categorical_transformer_other, cat_col_other),

    ], remainder = 'passthrough' )
# Preprocessing

# Break off validation set from training data

X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full_final, y, train_size=0.8, test_size=0.2,

                                                                random_state=0)



X_train_processed = preprocessor.fit_transform(X_train_full)

X_valid_processed = preprocessor.transform(X_valid_full)
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error



import warnings

warnings.filterwarnings('ignore')



#n_estimators = [500,1000,2000]

#learning_rate = [0.1,0.05,0.025,0.01]



# A hacky grid search, best results for n_estimators=2000 and learning_rate=0.05 with MAE around 15,400

#for n in n_estimators:

#    for l in learning_rate:

#

#        model = XGBRegressor(n_estimators=n, learning_rate=l)

#        model.fit(X_train_processed, y_train,verbose=False)

#        

#        predictions = model.predict(X_valid_processed)

#

#        print('n_estimators: ', n, 'lr: ', l, 'mae:', mean_absolute_error(predictions,y_valid))
# Creating final model, based on grid search above



model = XGBRegressor(n_estimators=2000, learning_rate=0.05)



# Training preprocessing

X_full_processed = preprocessor.fit_transform(X_full_final)



# Test preprocessing

X_test_full['Has_shed'] = np.where(X_test_full['MiscFeature']=='Shed',1,0)

X_test_full_final = X_test_full.drop('MiscFeature',axis=1)

X_test_processed = preprocessor.transform(X_test_full_final)



model.fit(X_full_processed,y)

predictions = model.predict(X_test_processed)

# Save test predictions to file

output = pd.DataFrame({'Id': X_test_full.index,

                       'SalePrice': predictions})

output.to_csv('submission.csv', index=False)