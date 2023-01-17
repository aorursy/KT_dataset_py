# General library imports

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder, RobustScaler

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor

# Input data files are available in the "../input/" directory.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

# Read the data

X = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')

X_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')



# Remove rows with missing target, separate target from predictors

X.dropna(axis=0, subset=['SalePrice'], inplace=True)

y = X.SalePrice

X.drop(['SalePrice'], axis=1, inplace=True)
def validate(Z,y):

    # "Cardinality" means the number of unique values in a column

    # Select categorical columns with relatively low cardinality (convenient but arbitrary)

    categorical_cols = [cname for cname in Z.columns if

                        Z[cname].dtype not in ['int64', 'float64']]



    # Select numerical columns

    numerical_cols = [cname for cname in Z.columns if 

                    Z[cname].dtype in ['int64', 'float64']]



    # Keep selected columns only

    my_cols = categorical_cols + numerical_cols

    # X = X[my_cols].copy()

    

    numerical_transformer = Pipeline(steps=[

        ('num_imputer', SimpleImputer(strategy='median')),

        ('num_scaler', RobustScaler())

    ])

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    preprocessor = ColumnTransformer(

        transformers=[

            ('num', numerical_transformer, numerical_cols),

            ('cat', categorical_transformer, categorical_cols)

        ])



    model = XGBRegressor(random_state=0, 

                          learning_rate=0.01, n_estimators=500,

                          max_depth=4,colsample_bytree=0.5, subsample=0.5)



    pipeline = Pipeline(steps=[('preprocessor', preprocessor),

                               ('model', model)

                              ])

    

    scores = -1 * cross_val_score(pipeline, Z[my_cols], y,

                              cv=5, n_jobs=-1,

                              scoring='neg_mean_absolute_error')

    return scores.mean()



ref = validate(X,y)

print(ref)
X['MSSubClass'] = X['MSSubClass'].astype(str)
missing_X  = X[X.columns[X.isnull().any()]]

plot = sns.barplot(x=missing_X.columns, y=missing_X.isnull().mean())

plot.set_xticklabels(plot.get_xticklabels(), rotation=60)
no_feature_columns = [

    'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 

    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'

    ]



X_tmp = X.copy()

X_tmp.fillna({x:'None' for x in no_feature_columns}, inplace=True)

print("fillNA = " + str(validate(X_tmp,y)-ref))

del X_tmp



X_tmp = X.copy()

X_tmp.drop(columns=no_feature_columns, inplace=True)

print("drop = " + str(validate(X_tmp,y)-ref))

del X_tmp



X_tmp = X.copy()

for col in no_feature_columns:

    if col in ['BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

               'GarageFinish', 'GarageQual', 'GarageCond']:

        X_tmp.drop(columns=col, inplace=True)

    else:

        mask = ~X_tmp[col].isna()

        X_tmp.loc[mask, col] = True

        X_tmp.fillna({col:False}, inplace=True)

print("Boolean = " + str(validate(X_tmp,y)-ref))

del X_tmp
print(f"Column\tfillna\tdrop\tBoolean")

results = np.zeros([len(no_feature_columns), 3])

for idx,col in enumerate(no_feature_columns):

    X_tmp = X.fillna({col:'None'})

    results[idx, 0] = validate(X_tmp, y)

    del X_tmp

    

    X_tmp = X.drop(columns=col)

    results[idx, 1] = validate(X_tmp, y)

    del X_tmp

        

    mask = ~X[col].isna()

    X_tmp = X.fillna({col:False})

    X_tmp.loc[mask, col] = True

    results[idx, 2] = validate(X_tmp, y)

    del X_tmp

    

    print(f"{col}\t{results[idx, 0]}\t{results[idx, 1]}\t{results[idx, 2]}")
sns.heatmap(results - ref, xticklabels=['fillna', 'drop', 'Boolean'], yticklabels=no_feature_columns)
sns.distplot(X.MasVnrArea.dropna())

(X.MasVnrArea.isna() == X.MasVnrType.isna()).mean()  # check if the same elements are missing in both columns
X[['MasVnrArea', 'MasVnrType']]
X_tmp = X.copy()

X_tmp.MasVnrArea.fillna(0)

X_tmp.MasVnrType.fillna('None')

print(validate(X_tmp, y) - ref)

del X_tmp
sns.distplot(X.LotFrontage.dropna())

sns.distplot(np.sqrt(X.LotArea/2))



mask = ~X.LotFrontage.isna()

np.corrcoef([X.LotFrontage[mask], np.sqrt(X.LotArea[mask]/2)])
sns.regplot(np.sqrt(X.LotArea/2), X.LotFrontage)

LotFrontage_linear_model = LinearRegression()

LotFrontage_linear_model.fit(np.array(np.sqrt(X.LotArea[mask]/2)).reshape(-1,1), X.LotFrontage[mask])

X_tmp = X.copy()

X_tmp.loc[~mask, 'LotFrontage'] = LotFrontage_linear_model.predict(np.array(np.sqrt(X.LotArea[~mask]/2)).reshape(-1,1))

sns.regplot(np.sqrt(X.LotArea[~mask]/2),X_tmp.LotFrontage[~mask])

print(validate(X_tmp, y) - ref)

del X_tmp
g = sns.lmplot(x='LotArea', y='LotFrontage', hue='Neighborhood', col='Neighborhood', data=X, col_wrap=5)

g.set(xlim=[0, 50000], ylim=[0,500])
g = sns.lmplot(x='LotArea', y='LotFrontage', col='LotConfig', hue='LotConfig', data=X)

g.set(xlim=[0, 100000], ylim=[0,500])
def LotConfig_fit(df):

    LotFrontage_models = {}  # i'll store all models here so we can use them later

    for name,group in df.groupby('LotConfig'):

        mask = ~group.LotFrontage.isnull()

        LotFrontage_models[name] = LinearRegression()

        LotFrontage_models[name].fit(np.array(group.LotArea[mask]).reshape(-1,1), group.LotFrontage[mask])

    return LotFrontage_models



def LotConfig_imput(x):

    return float(LotFrontage_models[x.LotConfig].predict(np.array(x.LotArea).reshape(-1,1)))



X_tmp = X.copy()

LotFrontage_models = LotConfig_fit(X_tmp)

X_tmp.loc[X_tmp.LotFrontage.isnull(), 'LotFrontage'] = X_tmp.loc[X_tmp.LotFrontage.isnull()].apply(LotConfig_imput, axis=1)

print(validate(X_tmp, y) - ref)

del X_tmp
X.Electrical.describe()
X.MiscFeature.describe()

X.loc[X.MiscVal > 0, ['MiscFeature', 'MiscVal']]
X_tmp = X.copy()

#X_tmp.MiscFeature.fillna('None', inplace=True)

X_tmp.drop(columns='MiscFeature', inplace=True)

print(validate(X_tmp,y)-ref)

del X_tmp
from sklearn.base import BaseEstimator, TransformerMixin



class CustomCatImputer(BaseEstimator, TransformerMixin):

    #Class Constructor 

    def __init__(self):

        self.LotFrontage_models = {}

    

    def fit(self, X, y=None):

        return self 



    def transform(self, X, y=None):

        # House does not have feature

        no_feature_columns = [

            'Alley', 'FireplaceQu', 'PoolQC', 'Fence', 

            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'

            ]



        X.fillna({x:'None' for x in no_feature_columns}, inplace=True)

        

        # Masonry Veneer

        X.MasVnrType.fillna('None', inplace=True)

        

        # MiscFeature

        X.drop(columns='MiscFeature', inplace=True)



        return X

    

class CustomNumImputer(BaseEstimator, TransformerMixin):

    #Class Constructor 

    def __init__(self):

        self.LotFrontage_models = {}

    

    def fit(self, X, y=None):

        # LotFrontage

        for name,group in X.groupby('LotConfig'):

            mask = ~group.LotFrontage.isnull()

            self.LotFrontage_models[name] = LinearRegression()

            self.LotFrontage_models[name].fit(np.array(group.LotArea[mask]).reshape(-1,1), group.LotFrontage[mask])

        return self 



    def transform(self, X, y=None):

        # Masonry Veneer

        X.MasVnrArea.fillna(0, inplace=True)

        

        # LotFrontage

        X.loc[X.LotFrontage.isnull(), 'LotFrontage'] = X.loc[X.LotFrontage.isnull()].apply(

            lambda x: float(self.LotFrontage_models[x.LotConfig].predict(np.array(x.LotArea).reshape(-1,1))),

            axis=1)

        X.drop(columns=['LotConfig','LotArea'], inplace=True)

        

        return X
def validate2(Z,y):

    # Custom transformer columns

    custom_cat_cols = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MasVnrType', 'MiscFeature',

                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

                   'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

    custom_num_cols = ['MasVnrArea', 'LotFrontage', 'LotConfig', 'LotArea']

    

    # Generic transformer columns

    categorical_cols = [cname for cname in Z.columns if

                        (Z[cname].dtype not in ['int64', 'float64']) and (cname not in custom_cat_cols)]

    numerical_cols = [cname for cname in Z.columns if 

                    (Z[cname].dtype in ['int64', 'float64']) and (cname not in ['MasVnrArea', 'LotFrontage'])]



    custom_num_transformer = Pipeline(steps=[

        ('custom_num_imputer', CustomNumImputer()),

        ('custom_num_scaler', RobustScaler())

    ])

    custom_cat_transformer = Pipeline(steps=[

        ('custom_cat_imputer', CustomCatImputer()),

        ('custom_onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])

    numerical_transformer = Pipeline(steps=[

        ('num_imputer', SimpleImputer(strategy='median')),

        ('num_scaler', RobustScaler())

    ])

    categorical_transformer = Pipeline(steps=[

        ('imputer', SimpleImputer(strategy='most_frequent')),

        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

    ])



    preprocessor = ColumnTransformer(

        transformers=[

            ('custom_num', custom_num_transformer, custom_num_cols),

            ('custom_cat', custom_cat_transformer, custom_cat_cols),

            ('num', numerical_transformer, numerical_cols),

            ('cat', categorical_transformer, categorical_cols)

        ])

    

    model = XGBRegressor(random_state=0, 

                          learning_rate=0.01, n_estimators=300,

                          max_depth=4,colsample_bytree=0.5, subsample=0.5)



    pipeline = Pipeline(steps=[

                                ('preprocessor', preprocessor),

                                ('model', model)

                              ])

    

    scores = -1 * cross_val_score(pipeline, Z, y,

                              cv=5, n_jobs=-1,

                              scoring='neg_mean_absolute_error')

    return scores.mean()



ref = validate2(X,y)

print(ref)
# Custom transformer columns

custom_cat_cols = ['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MasVnrType', 'MiscFeature',

               'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

               'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']

custom_num_cols = ['MasVnrArea', 'LotFrontage', 'LotConfig', 'LotArea']



# Generic transformer columns

categorical_cols = [cname for cname in X.columns if

                    (X[cname].dtype not in ['int64', 'float64']) and (cname not in custom_cat_cols)]

numerical_cols = [cname for cname in X.columns if 

                (X[cname].dtype in ['int64', 'float64']) and (cname not in ['MasVnrArea', 'LotFrontage'])]



custom_num_transformer = Pipeline(steps=[

    ('custom_num_imputer', CustomNumImputer()),

    ('custom_num_scaler', RobustScaler())

])

custom_cat_transformer = Pipeline(steps=[

    ('custom_cat_imputer', CustomCatImputer()),

    ('custom_onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])

numerical_transformer = Pipeline(steps=[

    ('num_imputer', SimpleImputer(strategy='median')),

    ('num_scaler', RobustScaler())

])

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='most_frequent')),

    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))

])



preprocessor = ColumnTransformer(

    transformers=[

        ('custom_num', custom_num_transformer, custom_num_cols),

        ('custom_cat', custom_cat_transformer, custom_cat_cols),

        ('num', numerical_transformer, numerical_cols),

        ('cat', categorical_transformer, categorical_cols)

    ])



model = XGBRegressor(random_state=0, 

                      learning_rate=0.01, n_estimators=3460,

                      max_depth=4,colsample_bytree=0.5, subsample=0.5)



pipeline = Pipeline(steps=[

                            ('preprocessor', preprocessor),

                            ('model', model)

                          ])



# Multiply by -1 since sklearn calculates *negative* MAE

scores = -1 * cross_val_score(pipeline, X, y,

                              cv=5, n_jobs=-1,

                              scoring='neg_mean_absolute_error')



print("Average MAE score:", scores.mean())

# score = 14316 @ n_estimators = 3460 (14865 @ 1000)
pipeline.fit(X, y)

preds_test = pipeline.predict(X_test)

output = pd.DataFrame({'Id': X_test.index,

                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)