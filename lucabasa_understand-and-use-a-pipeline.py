import pandas as pd

import numpy as np



from sklearn.model_selection import StratifiedShuffleSplit



from sklearn.base import BaseEstimator, TransformerMixin



from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import StandardScaler, RobustScaler

from sklearn.impute import SimpleImputer



import warnings



pd.set_option('max_columns', 500)
def make_test(train, test_size, random_state, strat_feat=None):

    if strat_feat:

        

        split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)



        for train_index, test_index in split.split(train, train[strat_feat]):

            train_set = train.loc[train_index]

            test_set = train.loc[test_index]

            

    return train_set, test_set
df_train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')



df_train.head()
df_train.info()
train_set, test_set = make_test(df_train, 

                                test_size=0.2, random_state=654, 

                                strat_feat='Neighborhood')
tmp = train_set[['GrLivArea', 'TotRmsAbvGrd']].copy()

tmp.head()
scaler = StandardScaler()  # initialize a StandardScaler object (more on this later)



tmp = scaler.fit_transform(tmp)  # apply a fit and a transform method (more on this later)



tmp
class df_scaler(TransformerMixin):

    def __init__(self, method='standard'):

        self.scl = None

        self.scale_ = None

        self.method = method

        if self.method == 'sdandard':

            self.mean_ = None

        elif method == 'robust':

            self.center_ = None



    def fit(self, X, y=None):

        if self.method == 'standard':

            self.scl = StandardScaler()

            self.scl.fit(X)

            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)

        elif self.method == 'robust':

            self.scl = RobustScaler()

            self.scl.fit(X)

            self.center_ = pd.Series(self.scl.center_, index=X.columns)

        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)

        return self



    def transform(self, X):

        # X has to be a dataframe

        Xscl = self.scl.transform(X)

        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)

        return Xscaled
tmp = train_set[['GrLivArea', 'TotRmsAbvGrd']].copy()

tmp.head()
scaler = df_scaler()  # initialize the oject



tmp = scaler.fit_transform(tmp)  # apply a fit and a transform method we defined above



tmp.head()  # this time it is a dataframe, we can use `head`
scaler.mean_
scaler.scale_
class general_cleaner(BaseEstimator, TransformerMixin):

    '''

    This class applies what we know from the documetation.

    It cleans some known missing values

    If flags the missing values



    This process is supposed to happen as first step of any pipeline

    '''

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X, y=None):

        #LotFrontage

        X.loc[X.LotFrontage.isnull(), 'LotFrontage'] = 0

        #Alley

        X.loc[X.Alley.isnull(), 'Alley'] = "NoAlley"

        #MSSubClass

        X['MSSubClass'] = X['MSSubClass'].astype(str)

        #MissingBasement

        fil = ((X.BsmtQual.isnull()) & (X.BsmtCond.isnull()) & (X.BsmtExposure.isnull()) &

              (X.BsmtFinType1.isnull()) & (X.BsmtFinType2.isnull()))

        fil1 = ((X.BsmtQual.notnull()) | (X.BsmtCond.notnull()) | (X.BsmtExposure.notnull()) |

              (X.BsmtFinType1.notnull()) | (X.BsmtFinType2.notnull()))

        X.loc[fil1, 'MisBsm'] = 0

        X.loc[fil, 'MisBsm'] = 1 # made explicit for safety

        #BsmtQual

        X.loc[fil, 'BsmtQual'] = "NoBsmt" #missing basement

        #BsmtCond

        X.loc[fil, 'BsmtCond'] = "NoBsmt" #missing basement

        #BsmtExposure

        X.loc[fil, 'BsmtExposure'] = "NoBsmt" #missing basement

        #BsmtFinType1

        X.loc[fil, 'BsmtFinType1'] = "NoBsmt" #missing basement

        #BsmtFinType2

        X.loc[fil, 'BsmtFinType2'] = "NoBsmt" #missing basement

        #BsmtFinSF1

        X.loc[fil, 'BsmtFinSF1'] = 0 # No bsmt

        #BsmtFinSF2

        X.loc[fil, 'BsmtFinSF2'] = 0 # No bsmt

        #BsmtUnfSF

        X.loc[fil, 'BsmtUnfSF'] = 0 # No bsmt

        #TotalBsmtSF

        X.loc[fil, 'TotalBsmtSF'] = 0 # No bsmt

        #BsmtFullBath

        X.loc[fil, 'BsmtFullBath'] = 0 # No bsmt

        #BsmtHalfBath

        X.loc[fil, 'BsmtHalfBath'] = 0 # No bsmt

        #FireplaceQu

        X.loc[(X.Fireplaces == 0) & (X.FireplaceQu.isnull()), 'FireplaceQu'] = "NoFire" #missing

        #MisGarage

        fil = ((X.GarageYrBlt.isnull()) & (X.GarageType.isnull()) & (X.GarageFinish.isnull()) &

              (X.GarageQual.isnull()) & (X.GarageCond.isnull()))

        fil1 = ((X.GarageYrBlt.notnull()) | (X.GarageType.notnull()) | (X.GarageFinish.notnull()) |

              (X.GarageQual.notnull()) | (X.GarageCond.notnull()))

        X.loc[fil1, 'MisGarage'] = 0

        X.loc[fil, 'MisGarage'] = 1

        #GarageYrBlt

        X.loc[X.GarageYrBlt > 2200, 'GarageYrBlt'] = 2007 #correct mistake

        X.loc[fil, 'GarageYrBlt'] = 0

        #GarageType

        X.loc[fil, 'GarageType'] = "NoGrg" #missing garage

        #GarageFinish

        X.loc[fil, 'GarageFinish'] = "NoGrg" #missing

        #GarageQual

        X.loc[fil, 'GarageQual'] = "NoGrg" #missing

        #GarageCond

        X.loc[fil, 'GarageCond'] = "NoGrg" #missing

        #Fence

        X.loc[X.Fence.isnull(), 'Fence'] = "NoFence" #missing fence

        #Pool

        fil = ((X.PoolArea == 0) & (X.PoolQC.isnull()))

        X.loc[fil, 'PoolQC'] = 'NoPool' 

        

        del X['Id']

        del X['MiscFeature']

        del X['MSSubClass']

        del X['Neighborhood']  # this should be useful

        del X['Condition1']

        del X['Condition2']

        del X['ExterCond']  # maybe ordinal

        del X['Exterior1st']

        del X['Exterior2nd']

        del X['Functional']

        del X['Heating']

        del X['PoolQC']

        del X['RoofMatl']

        del X['RoofStyle']

        del X['SaleCondition']

        del X['SaleType']

        del X['Utilities']

        del X['BsmtCond']

        del X['Electrical']

        del X['Foundation']

        del X['Street']

        del X['Fence']

        del X['LandSlope']

        

        return X
tmp = train_set.copy()



gt = general_cleaner()



tmp = gt.fit_transform(tmp)



tmp.head()
class df_imputer(BaseEstimator, TransformerMixin):

    '''

    Just a wrapper for the SimpleImputer that keeps the dataframe structure

    '''

    def __init__(self, strategy='mean'):

        self.strategy = strategy

        self.imp = None

        self.statistics_ = None



    def fit(self, X, y=None):

        self.imp = SimpleImputer(strategy=self.strategy)

        self.imp.fit(X)

        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)

        return self



    def transform(self, X):

        # X is supposed to be a DataFrame

        Ximp = self.imp.transform(X)

        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)

        return Xfilled

    

    

class df_scaler(BaseEstimator, TransformerMixin):

    '''

    Wrapper of StandardScaler or RobustScaler

    '''

    def __init__(self, method='standard'):

        self.scl = None

        self.scale_ = None

        self.method = method

        if self.method == 'sdandard':

            self.mean_ = None

        elif method == 'robust':

            self.center_ = None

        self.columns = None  # this is useful when it is the last step of a pipeline before the model



    def fit(self, X, y=None):

        if self.method == 'standard':

            self.scl = StandardScaler()

            self.scl.fit(X)

            self.mean_ = pd.Series(self.scl.mean_, index=X.columns)

        elif self.method == 'robust':

            self.scl = RobustScaler()

            self.scl.fit(X)

            self.center_ = pd.Series(self.scl.center_, index=X.columns)

        self.scale_ = pd.Series(self.scl.scale_, index=X.columns)

        return self



    def transform(self, X):

        # assumes X is a DataFrame

        Xscl = self.scl.transform(X)

        Xscaled = pd.DataFrame(Xscl, index=X.index, columns=X.columns)

        self.columns = X.columns

        return Xscaled



    def get_feature_names(self):

        return list(self.columns)  # this is going to be useful when coupled with FeatureUnion

    



class dummify(BaseEstimator, TransformerMixin):

    '''

    Wrapper for get dummies

    '''

    def __init__(self, drop_first=False, match_cols=True):

        self.drop_first = drop_first

        self.columns = []  # useful to well behave with FeatureUnion

        self.match_cols = match_cols



    def fit(self, X, y=None):

        self.columns = []  # for safety, when we refit we want new columns

        return self

    

    def match_columns(self, X):

        miss_train = list(set(X.columns) - set(self.columns))

        miss_test = list(set(self.columns) - set(X.columns))

        

        err = 0

        

        if len(miss_test) > 0:

            for col in miss_test:

                X[col] = 0  # insert a column for the missing dummy

                err += 1

        if len(miss_train) > 0:

            for col in miss_train:

                del X[col]  # delete the column of the extra dummy

                err += 1

                

        if err > 0:

            warnings.warn('The dummies in this set do not match the ones in the train set, we corrected the issue.',

                         UserWarning)

            

        return X

        

    def transform(self, X):

        X = pd.get_dummies(X, drop_first=self.drop_first)

        if (len(self.columns) > 0): 

            if self.match_cols:

                X = self.match_columns(X)

            self.columns = X.columns

        else:

            self.columns = X.columns

        return X

    

    def get_features_name(self):

        return self.columns
tmp = train_set[['HouseStyle']].copy()



dummifier = dummify()



tmp = dummifier.transform(tmp)  # no reason to call the fit method here



tmp.sample(5)
class feat_sel(BaseEstimator, TransformerMixin):

    '''

    This transformer selects either numerical or categorical features.

    In this way we can build separate pipelines for separate data types.

    '''

    def __init__(self, dtype='numeric'):

        self.dtype = dtype



    def fit( self, X, y=None ):

        return self 



    def transform(self, X, y=None):

        if self.dtype == 'numeric':

            num_cols = X.columns[X.dtypes != object].tolist()

            return X[num_cols]

        elif self.dtype == 'category':

            cat_cols = X.columns[X.dtypes == object].tolist()

            return X[cat_cols]
tmp = train_set.copy()



selector = feat_sel()  # it is numeric by default



tmp = selector.transform(tmp)  # no reason to fit again



tmp.head()
tmp = train_set[['RoofMatl']].copy()



dummifier = dummify()



dummifier.fit_transform(tmp).sum()  # to get how many dummies are present
test_set.RoofMatl.value_counts()
tmp = test_set[['RoofMatl']].copy()



dummifier.transform(tmp).sum()  # the same instance as before
class tr_numeric(BaseEstimator, TransformerMixin):

    def __init__(self, SF_room=True):

        self.columns = []  # useful to well behave with FeatureUnion

        self.SF_room = SF_room

        



    def fit(self, X, y=None):

        return self

    



    def remove_skew(self, X, column):

        X[column] = np.log1p(X[column])

        return X





    def SF_per_room(self, X):

        if self.SF_room:

            X['sf_per_room'] = X['GrLivArea'] / X['TotRmsAbvGrd']

        return X

    



    def transform(self, X, y=None):

        for col in ['GrLivArea', '1stFlrSF', 'LotArea']: # they can also be inputs

            X = self.remove_skew(X, col)



        X = self.SF_per_room(X)

        

        self.columns = X.columns 

        return X

    



    def get_features_name(self):  # again, it will be useful later

        return self.columns
numeric_pipe = Pipeline([('fs', feat_sel(dtype='numeric')),  # select only the numeri features

                         ('imputer', df_imputer(strategy='median')),  # impute the missing values with the median of each column

                         ('transf', tr_numeric(SF_room=True)),  # remove skew and create a new feature

                         ('scl', df_scaler(method='standard'))])  # scale the data



full_pipe = Pipeline([('gen_cl', general_cleaner()), ('num_pipe', numeric_pipe)])  # put the cleaner on top because we like it clean
tmp = train_set.copy()



tmp = full_pipe.fit_transform(tmp)



tmp.head()
tmp.info()
tmp = test_set.copy()  # not ready to work on those sets yet



tmp = full_pipe.transform(tmp)  # the fit already happened with the training set, we don't want to fit again



tmp.head()
full_pipe.get_params()
class make_ordinal(BaseEstimator, TransformerMixin):

    '''

    Transforms ordinal features in order to have them as numeric (preserving the order)

    If unsure about converting or not a feature (maybe making dummies is better), make use of

    extra_cols and include_extra

    '''

    def __init__(self, cols, extra_cols=None, include_extra=True):

        self.cols = cols

        self.extra_cols = extra_cols

        self.mapping = {'Po':1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}

        self.include_extra = include_extra

    



    def fit(self, X, y=None):

        return self

    



    def transform(self, X, y=None):

        if self.extra_cols:

            if self.include_extra:

                self.cols += self.extra_cols

            else:

                for col in self.extra_cols:

                    del X[col]

        

        for col in self.cols:

            X.loc[:, col] = X[col].map(self.mapping).fillna(0)

        return X



    

class recode_cat(BaseEstimator, TransformerMixin):        

    '''

    Recodes some categorical variables according to the insights gained from the

    data exploration phase. Not presented in this notebook

    '''

    def fit(self, X, y=None):

        return self

    

    

    def tr_GrgType(self, data):

        data['GarageType'] = data['GarageType'].map({'Basment': 'Attchd',

                                                  'CarPort': 'Detchd', 

                                                  '2Types': 'Attchd' }).fillna(data['GarageType'])

        return data

    

    

    def tr_LotShape(self, data):

        fil = (data.LotShape != 'Reg')

        data['LotShape'] = 1

        data.loc[fil, 'LotShape'] = 0

        return data

    

    

    def tr_LandCont(self, data):

        fil = (data.LandContour == 'HLS') | (data.LandContour == 'Low')

        data['LandContour'] = 0

        data.loc[fil, 'LandContour'] = 1

        return data

    

    

    def tr_LandSlope(self, data):

        fil = (data.LandSlope != 'Gtl')

        data['LandSlope'] = 0

        data.loc[fil, 'LandSlope'] = 1

        return data

    

    

    def tr_MSZoning(self, data):

        data['MSZoning'] = data['MSZoning'].map({'RH': 'RM', # medium and high density

                                                 'C (all)': 'RM', # commercial and medium density

                                                 'FV': 'RM'}).fillna(data['MSZoning'])

        return data

    

    

    def tr_Alley(self, data):

        fil = (data.Alley != 'NoAlley')

        data['Alley'] = 0

        data.loc[fil, 'Alley'] = 1

        return data

    

    

    def tr_LotConfig(self, data):

        data['LotConfig'] = data['LotConfig'].map({'FR3': 'Corner', # corners have 2 or 3 free sides

                                                   'FR2': 'Corner'}).fillna(data['LotConfig'])

        return data

    

    

    def tr_BldgType(self, data):

        data['BldgType'] = data['BldgType'].map({'Twnhs' : 'TwnhsE',

                                                 '2fmCon': 'Duplex'}).fillna(data['BldgType'])

        return data

    

    

    def tr_MasVnrType(self, data):

        data['MasVnrType'] = data['MasVnrType'].map({'BrkCmn': 'BrkFace'}).fillna(data['MasVnrType'])

        return data





    def tr_HouseStyle(self, data):

        data['HouseStyle'] = data['HouseStyle'].map({'1.5Fin': '1.5Unf', 

                                                         '2.5Fin': '2Story', 

                                                         '2.5Unf': '2Story', 

                                                         'SLvl': 'SFoyer'}).fillna(data['HouseStyle'])

        return data

    

    

    def transform(self, X, y=None):

        X = self.tr_GrgType(X)

        X = self.tr_LotShape(X)

        X = self.tr_LotConfig(X)

        X = self.tr_MSZoning(X)

        X = self.tr_Alley(X)

        X = self.tr_LandCont(X)

        X = self.tr_BldgType(X)

        X = self.tr_MasVnrType(X)

        X = self.tr_HouseStyle(X)

        return X
cat_pipe = Pipeline([('fs', feat_sel(dtype='category')),

                     ('imputer', df_imputer(strategy='most_frequent')), 

                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',

                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 

                     ('recode', recode_cat()), 

                     ('dummies', dummify())])



full_pipe = Pipeline([('gen_cl', general_cleaner()), ('cat_pipe', cat_pipe)])





tmp = train_set.copy()



tmp = full_pipe.fit_transform(tmp)



tmp.head()
tmp = test_set.copy()



tmp = full_pipe.transform(tmp)



tmp.head()
full_pipe.get_params()
class FeatureUnion_df(TransformerMixin, BaseEstimator):

    '''

    Wrapper of FeatureUnion but returning a Dataframe, 

    the column order follows the concatenation done by FeatureUnion



    transformer_list: list of Pipelines



    '''

    def __init__(self, transformer_list, n_jobs=None, transformer_weights=None, verbose=False):

        self.transformer_list = transformer_list

        self.n_jobs = n_jobs

        self.transformer_weights = transformer_weights

        self.verbose = verbose  # these are necessary to work inside of GridSearch or similar

        self.feat_un = FeatureUnion(self.transformer_list, 

                                    self.n_jobs, 

                                    self.transformer_weights, 

                                    self.verbose)

        

    def fit(self, X, y=None):

        self.feat_un.fit(X)

        return self



    def transform(self, X, y=None):

        X_tr = self.feat_un.transform(X)

        columns = []

        

        for trsnf in self.transformer_list:

            cols = trsnf[1].steps[-1][1].get_features_name()  # getting the features name from the last step of each pipeline

            columns += list(cols)



        X_tr = pd.DataFrame(X_tr, index=X.index, columns=columns)

        

        return X_tr



    def get_params(self, deep=True):  # necessary to well behave in GridSearch

        return self.feat_un.get_params(deep=deep)
numeric_pipe = Pipeline([('fs', feat_sel('numeric')),

                         ('imputer', df_imputer(strategy='median')),

                         ('transf', tr_numeric())])





cat_pipe = Pipeline([('fs', feat_sel('category')),

                     ('imputer', df_imputer(strategy='most_frequent')), 

                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',

                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 

                     ('recode', recode_cat()), 

                     ('dummies', dummify())])





processing_pipe = FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),

                                                 ('num_pipe', numeric_pipe)])





full_pipe = Pipeline([('gen_cl', general_cleaner()), 

                      ('processing', processing_pipe), 

                      ('scaler', df_scaler())])  # the scaler is here to have also the ordinal features scaled



tmp = df_train.copy()



tmp = full_pipe.fit_transform(tmp)



tmp.head()
tmp = test_set.copy()



tmp = full_pipe.transform(tmp)



tmp.head()
full_pipe.get_params()
from sklearn.linear_model import Lasso

from sklearn.model_selection import KFold, GridSearchCV



folds = KFold(5, shuffle=True, random_state=541)



df_train['Target'] = np.log1p(df_train.SalePrice)



del df_train['SalePrice']



train_set, test_set = make_test(df_train, 

                                test_size=0.2, random_state=654, 

                                strat_feat='Neighborhood')



y = train_set['Target'].copy()

del train_set['Target']



y_test = test_set['Target']

del test_set['Target']





def grid_search(data, target, estimator, param_grid, scoring, cv):

    

    grid = GridSearchCV(estimator=estimator, param_grid=param_grid, 

                        cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    

    pd.options.mode.chained_assignment = None  # this is because the gridsearch throws a lot of pointless warnings

    tmp = data.copy()

    grid = grid.fit(tmp, target)

    pd.options.mode.chained_assignment = 'warn'

    

    result = pd.DataFrame(grid.cv_results_).sort_values(by='mean_test_score', 

                                                        ascending=False).reset_index()

    

    del result['params']

    times = [col for col in result.columns if col.endswith('_time')]

    params = [col for col in result.columns if col.startswith('param_')]

    

    result = result[params + ['mean_test_score', 'std_test_score'] + times]

    

    return result, grid.best_params_
lasso_pipe = Pipeline([('gen_cl', general_cleaner()),

                       ('processing', processing_pipe),

                       ('scl', df_scaler()), 

                       ('lasso', Lasso(alpha=0.01))])



res, bp = grid_search(train_set, y, lasso_pipe, 

            param_grid={'processing__num_pipe__transf__SF_room': [True, False], 

                        'processing__num_pipe__imputer__strategy': ['mean', 'median'],

                        'processing__cat_pipe__dummies__drop_first': [True, False],

                        'lasso__alpha': [0.1, 0.01, 0.001]},

            cv=folds, scoring='neg_mean_squared_error')



res
bp
from sklearn.metrics import mean_squared_error, mean_absolute_error



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



def cv_score(df_train, y_train, kfolds, pipeline):

    oof = np.zeros(len(df_train))

    train = df_train.copy()

    

    for train_index, test_index in kfolds.split(train.values):

            

        trn_data = train.iloc[train_index][:]

        val_data = train.iloc[test_index][:]

        

        trn_target = y_train.iloc[train_index].values.ravel()

        val_target = y_train.iloc[test_index].values.ravel()

        

        pipeline.fit(trn_data, trn_target)



        oof[test_index] = pipeline.predict(val_data).ravel()

            

    return oof





def get_coef(pipe):

    imp = pipe.steps[-1][1].coef_.tolist()

    feats = pipe.steps[-2][1].get_feature_names()  # again, this is why we implemented that method

    result = pd.DataFrame({'feat':feats,'score':imp})

    result = result.sort_values(by=['score'],ascending=False)

    return result



def _plot_diagonal(ax):

    xmin, xmax = ax.get_xlim()

    ymin, ymax = ax.get_ylim()

    low = min(xmin, xmax)

    high = max(xmin, xmax)

    scl = (high - low) / 100

    

    line = pd.DataFrame({'x': np.arange(low, high ,scl), # small hack for a diagonal line

                         'y': np.arange(low, high ,scl)})

    ax.plot(line.x, line.y, color='black', linestyle='--')

    

    return ax





def plot_predictions(data, true_label, pred_label, feature=None, hue=None, legend=False):

    

    tmp = data.copy()

    tmp['Prediction'] = pred_label

    tmp['True Label'] = true_label

    tmp['Residual'] = tmp['True Label'] - tmp['Prediction']

    

    diag = False

    alpha = 0.7

    label = ''

    

    fig, ax = plt.subplots(1,2, figsize=(15,6))

    

    if feature is None:

        feature = 'True Label'

        diag = True

    else:

        legend = 'full'

        sns.scatterplot(x=feature, y='True Label', data=tmp, ax=ax[0], label='True',

                         hue=hue, legend=legend, alpha=alpha)

        label = 'Predicted'

        alpha = 0.4



    sns.scatterplot(x=feature, y='Prediction', data=tmp, ax=ax[0], label=label,

                         hue=hue, legend=legend, alpha=alpha)

    if diag:

        ax[0] = _plot_diagonal(ax[0])

    

    sns.scatterplot(x=feature, y='Residual', data=tmp, ax=ax[1], 

                    hue=hue, legend=legend, alpha=0.7)

    ax[1].axhline(y=0, color='r', linestyle='--')

    

    ax[0].set_title(f'{feature} vs Predictions')

    ax[1].set_title(f'{feature} vs Residuals')
lasso_oof = cv_score(train_set, y, folds, lasso_pipe)



lasso_oof[:10]
get_coef(lasso_pipe)  # it has been fitted in the cv_score function

# to be fair, these coefficients refer only to the last of the 5 folds
plot_predictions(train_set, y, lasso_oof)
plot_predictions(train_set, y, lasso_oof, feature='GrLivArea')
numeric_pipe = Pipeline([('fs', feat_sel('numeric')),

                         ('imputer', df_imputer(strategy='mean')),  # tuned above

                         ('transf', tr_numeric(SF_room=True))])  # tuned above





cat_pipe = Pipeline([('fs', feat_sel('category')),

                     ('imputer', df_imputer(strategy='most_frequent')), 

                     ('ord', make_ordinal(['BsmtQual', 'KitchenQual','GarageQual',

                                           'GarageCond', 'ExterQual', 'HeatingQC'])), 

                     ('recode', recode_cat()), 

                     ('dummies', dummify(drop_first=True))])  # tuned above





processing_pipe = FeatureUnion_df(transformer_list=[('cat_pipe', cat_pipe),

                                                    ('num_pipe', numeric_pipe)])



lasso_pipe = Pipeline([('gen_cl', general_cleaner()), 

                 ('processing', processing_pipe),

                  ('scl', df_scaler()), ('lasso', Lasso(alpha=0.01))])  # tuned above



lasso_oof = cv_score(train_set, y, folds, lasso_pipe)



get_coef(lasso_pipe)
plot_predictions(train_set, y, lasso_oof)
plot_predictions(train_set, y, lasso_oof, feature='GrLivArea')
lasso_pred = lasso_pipe.predict(test_set)



plot_predictions(test_set, y_test, lasso_pred)
plot_predictions(test_set, y_test, lasso_pred, feature='GrLivArea')
print('Score in 5-fold cv')

print(f'\tRMSE: {round(np.sqrt(mean_squared_error(y, lasso_oof)), 5)}')

print(f'\tMAE: {round(mean_absolute_error(np.expm1(y), np.expm1(lasso_oof)), 2)} dollars')

print('Score on holdout test')

print(f'\tRMSE: {round(np.sqrt(mean_squared_error(y_test, lasso_pred)), 5)}')

print(f'\tMAE: {round(mean_absolute_error(np.expm1(y_test), np.expm1(lasso_pred)), 2)} dollars')
df_test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')



sub = df_test[['Id']].copy()



predictions = lasso_pipe.predict(df_test)
import df_pipeline as dfp
dummifier = dfp.dummify()



tmp = train_set[['HouseStyle']].copy()



tmp = dummifier.transform(tmp)



tmp.sample(5)