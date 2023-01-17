import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error

from __future__ import print_function
import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train = train[train.GrLivArea < 4000]
train.shape
missing = train.isnull().sum().sort_values(ascending = False)
missing_pct = missing /len(train)
features_to_discard = list(missing_pct[missing_pct > 0.8].index)

print('features discared:', features_to_discard)
train.drop(features_to_discard, axis=1, inplace = True)
nominal_features = ['MSSubClass', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 
                   'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                   'GarageType', 'SaleType', 'SaleCondition']
ordinal_features = ['Street', 'CentralAir', 'LotShape', 'Utilities', 'LandSlope', 'OverallQual', 'OverallCond', 'ExterCond', 'ExterQual', 
                   'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'Electrical', 'KitchenQual',
                   'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']
#the propotion of bedroom, bathroom, kitchen compare to total rooms maybe a good feature
discrete_features = ['YearBuilt', 'YearRemodAdd', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
                    'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars', 'YrSold', ]
#the proportion of living area compare to total area maybe a good feature
continuous_features = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
                      'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                      'PoolArea', 'MiscVal', ]


misc_features = ['MoSold', ]
# Building Block for the full pipeline

oridnal_map = {
    #qual, cond ect.
    'No':0, 'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5, 
    #yes or no
    'N':0, 'Y': 2, 
    # fence feature
    'MnWw':1, 'GdWo':2, 'MnPrv':3, 'GdPrv':4,
    #LotShape
    'IR3':1, 'IR2':2, 'IR1':3, 'Reg':4,
    #Utilities
    'ELO':1, 'NoSeWa':2, 'NoSewr':3, 'AllPub':4,
    #Land Slope
    'Sev':1, 'Mod':2, 'Gtl':3,
    #BsmtFin Type 1/2
    'Unf':1, 'LwQ':2, 'Rec':3, 'BLQ':4, 'ALQ':5, 'GLQ':6,
    #Electrical
    'Mix':1, 'FuseP':2, 'FuseF':3, 'FuseA':4, 'SBrkr':5,
    #Functional
    'Sal':1, 'Sev':2, 'Maj2':3, 'Maj1':4, 'Mod':5, 'Min2':6, 'Min1':7, 'Typ':8,
    #Garage Finish
    'Unf':1, 'RFn':2, 'Fin':3,
    #Paved Drive
    'N':0, 'P':1, 'Y':2,
    #Street
    'Grvl':1, 'Pave':2,
    #Basement exposure
    'Mn':2, 'Av':3, 'Gd':4
}


class OrdImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        df_x = pd.DataFrame(X, columns=ordinal_features)
        df_x = df_x.fillna('No')
        df_x =  df_x.applymap(lambda x:oridnal_map.get(x,x))
        result = df_x.values
        return result
    
class CatImputerEncoder(BaseEstimator, TransformerMixin):
    """ Impute categorical features, using most frequent strategy 
        and Encode categorical features using OneHot Encoder
    """
    def __init__(self, features):
        self.features = features
        
    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)
        tmp = X.fillna(self.fill).values
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.encoder.fit(tmp)
        self.categories_ = [feat + '_' + str(it) for feat, items in zip(self.features, self.encoder.categories_) for it in items]
        return self

    def transform(self, X, y=None):
        return self.encoder.transform(X)


class FeatureEngneer(BaseEstimator, TransformerMixin):
    def __init__(self):
        """
        Feature Engneering here!!
        """
        return
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return X

class Normalizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        from scipy.stats import skew
        tmp = pd.DataFrame(X)
        n_uniq_values = tmp.apply(lambda x:len(x.unique()))
        n_uniq_values_gt_100 = n_uniq_values[n_uniq_values > 100].index
        sk = skew(tmp[n_uniq_values_gt_100])
        self.index2normalize_ = n_uniq_values_gt_100[np.abs(sk) > 0.5]
        return self
    def transform(self, X, y = None):
        X[:, self.index2normalize_] = np.log1p(X[:, self.index2normalize_])
        return X

class FeatureEngneer(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):
        return self
    def transform(self, X, y = None):
        return None

class Wrapper(BaseEstimator, TransformerMixin):
        def __init__(self):
            """
            For ordinal features, impute with 'No'
            For discrete and nominal features, impute with most frequent values
            For continuous features, impute with mean/median, making it a hyperparameter for later search
            """
            self.imputer =  ColumnTransformer([
                ('ordinal', OrdImputer(), ordinal_features),
                ('discrete', Imputer(strategy = 'most_frequent'), discrete_features),
                ('continuous', Imputer(strategy='mean'), continuous_features),
                 ('nominal', CatImputerEncoder(features = nominal_features), nominal_features),
            ])
            self.feature_engneer = FeatureEngneer()
            
        def fit(self, X, y = None):
            self.imputer.fit(X)
            self.feature_engneer.fit(X)
            return self
        
        def transform(self, X, y = None):
            transformed = self.imputer.transform(X)
            fe = self.feature_engneer.transform(transformed)
            if fe:
                return np.c_[transformed, fe]
            else:
                return transformed

from sklearn.preprocessing import RobustScaler

norm = Normalizer()
std_scaler = RobustScaler()

full_pipeline = Pipeline([
    
    # Including purifying the data and possibly some feature engneering.
    # This is ugly, but I dont have a better solution right now.
    ('expansion', Wrapper()),  
    ('normalization', norm),
    ('standardization', std_scaler),
])
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(train, test_size = 0.2, shuffle= True, random_state = 1)

print('train set:', train_set.shape)
print('test set:', test_set.shape)
from sklearn.linear_model import Lasso, LassoCV
from sklearn.metrics import mean_squared_error, make_scorer

scorer = make_scorer(mean_squared_error, greater_is_better=False)

lasso_reg = Lasso()

prepare_predict_pipeline = Pipeline([
    ('prepare', full_pipeline),
    ('predict', lasso_reg),
])

params = [
    {
     'predict__alpha':[0.0005]
    }
]

grid_search = GridSearchCV(prepare_predict_pipeline, param_grid=params, scoring=scorer, cv = 5, verbose=1)
grid_search.fit(train_set, np.log(train_set.SalePrice.copy()))
cvres = grid_search.cv_results_
sorted([(np.sqrt(-score), para) for score, para in zip(cvres['mean_test_score'], cvres['params'])], reverse=False)[:10]
pred = grid_search.best_estimator_.predict(test_set)
print('result on test set:' ,np.sqrt(mean_squared_error(np.log(test_set.SalePrice), pred)))
y = np.log(train.SalePrice.copy())
grid_search.best_estimator_.fit(train, y)
pred = grid_search.best_estimator_.predict(test)

result = pd.DataFrame()
result['Id'] = test.Id
result['SalePrice'] = np.exp(pred)
result.to_csv('it2_result.csv', index=False)