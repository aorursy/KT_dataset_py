import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split, cross_validate, LeaveOneOut, GridSearchCV

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor

from sklearn.pipeline import FeatureUnion, Pipeline

from sklearn.decomposition import PCA

from sklearn.feature_selection import SelectKBest, f_regression, SelectFromModel

from sklearn.metrics import mean_squared_log_error,  mean_squared_error

from sklearn.linear_model import Ridge, Lasso

from category_encoders.ordinal import OrdinalEncoder
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv').set_index('Id')

test.isna().sum()[test.isna().sum() > 0]
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv').set_index('Id')

train
X_train, y_train = train.drop('SalePrice', axis=1), train['SalePrice']
X_train.dtypes.value_counts()
X_train.columns
ordinal_features = pd.Index(['LotShape', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',

                   'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'Functional',

                   'GarageFinish', 'GarageQual', 'GarageCond', 'FireplaceQu', 'PoolQC', 'Fence', ])

one_hot_features = pd.Index(['Street', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 

                   'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 

                   'MasVnrType', 'Foundation', 'Heating', 'CentralAir', 'Electrical', 'GarageType', 'PavedDrive', 'SaleType',

                    'SaleCondition', 'Alley', 'MiscFeature', ])

categorical_features =  ordinal_features.union(one_hot_features)

numerical_features = X_train.columns.difference(categorical_features)

num_na_features = pd.Index(['MasVnrArea', 'GarageYrBlt', 'BsmtFinSF1', 'BsmtFinSF2'])
len(ordinal_features) + len(one_hot_features) + len(numerical_features)
qual = {'col': "QUALITY", 'mapping': {

        'NA': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5

}}

qual_features = ['PoolQC', 'GarageCond', 'GarageQual', 'FireplaceQu', 'KitchenQual', 'HeatingQC', 'BsmtCond', 

                'BsmtQual', 'ExterCond', 'ExterQual', ]

categories = [

    {'col': 'LotShape', 'mapping': {

        'NA': 0, 'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4

    }},

    {'col': 'Utilities', 'mapping': {

        'NA': 0, 'ELO': 1, 'NoSeWa': 2, 'NoSewr': 3, 'AllPub': 4

    }},

    {'col': 'LandSlope', 'mapping': {

        'NA': 0, 'Gtl': 1, 'Mod': 2, 'Sev': 3

    }},

    {'col': 'BsmtExposure', 'mapping': {

        'NA': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4

    }},

    {'col': 'BsmtFinType1', 'mapping': {

        'NA': 0, 'Unf': 1, 'LwQ':2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6

    }},

    {'col': 'BsmtFinType2', 'mapping': {

        'NA': 0, 'Unf': 1, 'LwQ':2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6

    }},

    {'col': 'Functional', 'mapping': {

        'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1':6, 'Typ': 7

    }},

    {'col': 'GarageFinish', 'mapping': {

        'NA': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3

    }},

    {'col': 'Fence', 'mapping': {

        'NA': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4

    }},

    *({'col': col, 'mapping': qual['mapping']} for col in qual_features )

]
one_hot_impute = ColumnTransformer([

    ('impute', SimpleImputer(strategy='constant', fill_value='NA'), one_hot_features)

])

one_hot = Pipeline([

    ('impute', one_hot_impute),

    ('encode', OneHotEncoder(sparse=False, handle_unknown='ignore'))

])

ordinal = ColumnTransformer([

    ('encode', OrdinalEncoder(drop_invariant=True, mapping=categories), ordinal_features)

])

numerical = ColumnTransformer([

    ('lot_frontage', SimpleImputer(), numerical_features.difference(num_na_features)),

    ('zero_int', SimpleImputer(strategy='constant', fill_value=0), num_na_features)

])



expand_impute = FeatureUnion([

        ('numerical', numerical),

        ('ordinal', ordinal),

        ('one_hot', one_hot),

])



pipeline = Pipeline([

    ('expand_impute', expand_impute),

    ('pca', PCA(n_components=100, whiten=True, svd_solver='full')),

    ('select', SelectFromModel(Lasso(alpha=1), threshold='0.75*mean', max_features=50)) # mean=> 0.0250, 1.25*mean=>0.0270

    # 50 features - overfit, 0.75*mean - best

])
pipeline.fit(X_train, y_train)
pipeline.transform(X_train).shape
%%time

ridge_param = {

    'regressor__alpha': np.logspace(-1, 2, 4, base=10)

}

ridgeCV = GridSearchCV(

    TransformedTargetRegressor(

        check_inverse=True,

        func=np.log,

        inverse_func=np.exp,

        regressor=Ridge(random_state=42)

    ),

    ridge_param,

    scoring='neg_mean_squared_log_error',

    cv=LeaveOneOut()

).fit(pipeline.transform(X_train), y_train)
# SelectFromModel(threshold=0.75*mean)

(-ridgeCV.best_score_)**(1/2), mean_squared_log_error(y_train, ridgeCV.predict(pipeline.transform(X_train)))**(1/2)
(ridgeCV.cv_results_['mean_test_score'] * -1)**(1/2)
pred = ridgeCV.predict(pipeline.transform(test))

pred = pd.DataFrame(pred, columns=['SalePrice'])

pred['Id'] = test.index

pred.set_index('Id').to_csv('pred.csv')