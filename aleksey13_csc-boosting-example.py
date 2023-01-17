# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

from sklearn import ensemble

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.preprocessing import OneHotEncoder, StandardScaler



import sklearn as skl

import lightgbm as lgb

import xgboost as xgb

import catboost as cb

from catboost import CatBoostClassifier, Pool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

print("scikit-learn",skl.__version__)

print("lightgbm",lgb.__version__)

print("xgboost",xgb.__version__)

print("catboost",cb.__version__)
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.preprocessing import FunctionTransformer

from sklearn_pandas import DataFrameMapper



from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelBinarizer, OneHotEncoder, LabelEncoder

from sklearn.impute import SimpleImputer



from sklearn.utils import column_or_1d

from sklearn.utils.validation import check_is_fitted



from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.pipeline import FeatureUnion, make_union

from sklearn_pandas import DataFrameMapper, gen_features



# https://www.kaggle.com/gautham11/building-predictive-models-with-sklearn-pipelines

class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):

        self.dtype = dtype

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.select_dtypes(include=[self.dtype])



class StringIndexer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        return self

    def transform(self, X):

        assert isinstance(X, pd.DataFrame)

        return X.apply(lambda s: s.cat.codes.replace(

            {-1: len(s.cat.categories)}

        ))



class CustomImputer(BaseEstimator, TransformerMixin):

    def __init__(self, strategy='mean', filler='NA'):

        self.strategy = strategy

        self.fill = filler



    def fit(self, X, y=None):

        if self.strategy in ['mean', 'median']:

            if not all([dtype in [np.number, np.int] for dtype in X.dtypes]):

                raise ValueError('dtypes mismatch np.number dtype is required for ' + self.strategy)

        if self.strategy == 'mean':

            self.fill = X.mean()

        elif self.strategy == 'median':

            self.fill = X.median()

        elif self.strategy == 'mode':

            self.fill = X.mode().iloc[0]

        elif self.strategy == 'fill':

            if type(self.fill) is list and type(X) is pd.DataFrame:

                self.fill = dict([(cname, v) for cname, v in zip(X.columns, self.fill)])

        return self



    def transform(self, X, y=None):

        if self.fill is None:

            self.fill = 'NA'

        return X.fillna(self.fill)

    

def CustomMapper(result_column='mapped_col', value_map={}, default=np.nan):

    def mapper(X, result_column, value_map, default):

        def colmapper(col):

            return col.apply(lambda x: value_map.get(x, default))

        mapped_col = X.apply(colmapper).values

        mapped_col_names = [result_column + '_' + str(i) for i in range(mapped_col.shape[1])]

        return pd.DataFrame(mapped_col, columns=[mapped_col_names])

    return FunctionTransformer(

        mapper,

        validate=False,

        kw_args={'result_column': result_column, 'value_map': value_map, 'default': default}

    )





class SafeLabelEncoder(LabelEncoder):

    

    @staticmethod

    def _get_unseen():

        return 99999

    

    def fit_transform(self, y):

        f



    def transform(self, y):

        check_is_fitted(self, 'classes_')

        y = column_or_1d(y, warn=True)

        classes = np.unique(y)

        # Check not too many:

        unseen = self._get_unseen()

        if len(classes) >= unseen:

            raise ValueError('Too many factor levels in feature. Max is %i' % unseen)

        e = np.array([

                         np.searchsorted(self.classes_, x) if x in self.classes_ else unseen

                         for x in y

                         ])



        return e
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv')

print(df['is_canceled'].value_counts() / len(df))



pd.set_option('display.max_columns', 50)

# df[df['is_canceled'] == 1]

df
# df['reservation_status'].value_counts()

# tdf = pd.get_dummies(df['reservation_status'])

# tdf['is_canceled'] = df['is_canceled']

# tdf.corr()
df = pd.read_csv('/kaggle/input/hotel-booking-demand/hotel_bookings.csv',parse_dates=['reservation_status_date'])

# possible target leak

df = df.drop(columns=['reservation_status','reservation_status_date'])
df['global_week'] = df['arrival_date_week_number'] + (df['arrival_date_year'] - 2015) * 53

val_df = df[df['global_week'] > 121]

learn_df = df[df['global_week'] <= 121]

print(f"Learn {len(learn_df)*100/len(df)}%, Val {len(val_df)*100/len(df)}%")

del df
def to_X_y(df):

    return df.drop('is_canceled', axis=1), df['is_canceled']



val_X,val_y = to_X_y(val_df)

learn_X,learn_y = to_X_y(learn_df)
learn_df.info()
learn_df.head()
learn_df.describe()
# Check na values 

print("Nan in each columns")

for i,line in enumerate(str(learn_df.isna().sum()).split('\n')):

    print(i,'\t',line)
na_cols = learn_X.columns[learn_X.isna().sum() > 0]

print("NA columns:",na_cols)

print("NA before",learn_X.isna().sum().sum())

for col in na_cols:

    learn_X[col].fillna(learn_X[col].mode()[0], inplace=True)

print("NA after",learn_X.isna().sum().sum())
# select_dtypes :( doesn't work



cat_columns = [x for i,x in enumerate(learn_X.columns) if learn_X[x].dtype == np.object]

cat_columns_idx = [i for i,x in enumerate(learn_X.columns) if learn_X[x].dtype == np.object]

num_columns = [x for i,x in enumerate(learn_X.columns) if learn_X[x].dtype in ['int','float']]

print("Cat columns:",cat_columns)

print("Number columns:",num_columns)
# LabelEncoder can't handle unknown labels.

# For example after e.fit(['a','b']), e.transform(['a','b','c']) return error.

class LabelEncoderUnknownHack(LabelEncoder):

    __UNKNOWN_CLS = 999999

    def transform(self, X):

        class_to_i = {c:i for i,c in enumerate(self.classes_)}

        vf = np.vectorize(lambda x: class_to_i[x] if x in class_to_i else self.__UNKNOWN_CLS)

        return vf(X)

    

### Preprocessing pipline ###

# Numerical features

num_data_pipeline = DataFrameMapper(

        [(num_columns, [

                CustomImputer(strategy='median'),

                StandardScaler()

            ], {'alias': 'num_data'})],

    input_df=True ,df_out=True)



# Categorical one hot

# Apply for every cat feature

cat_data_pipeline = DataFrameMapper(

    [([column], [

        CustomImputer(strategy='mode'),

        OneHotEncoder(handle_unknown='ignore')

    ], {'alias': 'cat_'+column}) for column in cat_columns],

    input_df=True ,df_out=True)



# Categorical labeling

cat_data_pipeline_labeling = DataFrameMapper(

    [([column], [

        CustomImputer(strategy='mode'),

        LabelEncoderUnknownHack()

    ], {'alias': 'cat_'+column}) for column in cat_columns],

    input_df=True ,df_out=True)



# Pipeline 1

features_pipeline = make_union(num_data_pipeline, cat_data_pipeline)

# Pipeline 2

features_pipeline_labeling = make_union(num_data_pipeline, cat_data_pipeline_labeling)



### Function for testing ###

def test_model(model_handler,params,n_split=5):

    aucs = []

    kf = KFold(n_splits = n_split, shuffle = True, random_state = 2)

    for i,(train_index, test_index) in enumerate(kf.split(learn_X)):

        print(f"Fold {i+1}/{n_split} ...")

        X_train, X_test = learn_X.iloc[train_index], learn_X.iloc[test_index]

        y_train, y_test = learn_y.iloc[train_index], learn_y.iloc[test_index]

        y_pred_proba = model_handler(X_train,y_train,X_test,y_test,params)

        fpr, tpr, _ = roc_curve(y_test,  y_pred_proba)

        auc = roc_auc_score(y_test, y_pred_proba)

        aucs.append(auc)

        plt.plot(fpr,tpr,label=f"Fold {i+1}, auc={str(auc)}")

    

    plt.legend(loc=4)

    plt.show()

    m = np.mean(aucs)

    s = np.std(aucs)

    print("AUC: %.4f±%.4f" % (m,s))
def scikit_learn_handler(X_train,y_train,X_test,y_test,params):    

    

    features_pipeline.fit(X_train)



    X = features_pipeline.transform(X_train)

    X2 = features_pipeline.transform(X_test)



    model = ensemble.GradientBoostingClassifier(**params)

    model.fit(X, y_train)

    return model.predict_proba(X2)[:,1]
%%time

params = {

        'n_estimators': 10, 

        'max_depth': 3, 

        'learning_rate': 0.05

    }

    

test_model(scikit_learn_handler,params)
%%time

params = {

        'n_estimators': 10, 

        'max_depth': 3, 

        'learning_rate': 0.05,

        'subsample': 0.5

    }

    

test_model(scikit_learn_handler,params)
def xg_boost_handler(X_train,y_train,X_test,y_test,params):

    

    features_pipeline.fit(X_train)

    

    X = features_pipeline.transform(X_train)

    X2 = features_pipeline.transform(X_test)    

    

    model = xgb.XGBClassifier(**params)

    model.fit(X, y_train)

    return model.predict_proba(X2)[:,1]
%%time

params = {

    'n_estimators': 10, 

    'max_depth': 3, 

    'learning_rate': 0.05, 

    'tree_method': 'exact'

}



test_model(xg_boost_handler,params)
%%time

params = {

    'n_estimators': 10, 

    'max_depth': 3, 

    'learning_rate': 0.05, 

    'tree_method': 'hist'

}



test_model(xg_boost_handler,params)
%%time

params = {

    'n_estimators': 10, 

    'max_depth': 3, 

    'learning_rate': 0.05, 

    'tree_method': 'hist',

    'n_jobs': 4

}



test_model(xg_boost_handler,params)
def lightgbm_handler(X_train,y_train,X_test,y_test,params):

    

    features_pipeline.fit(X_train)

    

    X = features_pipeline.transform(X_train)

    X2 = features_pipeline.transform(X_test) 

    

    model = lgb.LGBMClassifier(**params)

    model.fit(X, y_train)

    return model.predict_proba(X2)[:,1]
%%time

params = {

    'n_estimators': 10, 

    'max_depth': 3, 

    'learning_rate': 0.05,

}

test_model(lightgbm_handler, params)
%%time

params = {

    'n_estimators': 100, 

    'max_depth': 3, 

    'learning_rate': 0.05,

}

test_model(lightgbm_handler, params)
########## !!!!! IN PGROGRESS  !!!! ##########

# import warnings

# from sklearn.exceptions import DataConversionWarning

# warnings.filterwarnings(action='ignore', category=DataConversionWarning)



# TODO fix label encoding



def lightgbm_handler_cat(X_train,y_train,X_test,y_test,params):

    

    features_pipeline_labeling.fit(X_train)

    

    X = features_pipeline_labeling.transform(X_train)

    X2 = features_pipeline_labeling.transform(X_test) 

    

    model = lgb.LGBMClassifier(**params)

    model.fit(X, y_train, categorical_feature=cat_columns_idx)

    return model.predict_proba(X2)[:,1]



# %%time

params = {

    'n_estimators': 100, 

    'max_depth': 3, 

    'learning_rate': 0.05

}

test_model(lightgbm_handler_cat, params)
def catboost_handler(X_train,y_train,X_test,y_test,params):

    features_pipeline.fit(X_train)

    

    X = features_pipeline.transform(X_train)

    X2 = features_pipeline.transform(X_test) 



    model = CatBoostClassifier(**params)



    model.fit(X,y_train,silent=True)

    return model.predict_proba(X2)[::,1]
%%time

params = {

    'n_estimators': 100, 

    'max_depth': 3, 

    'learning_rate': 0.05

}

    

test_model(catboost_handler,params)
def catboost_handler_cat(X_train,y_train,X_test,y_test,params):

    train_dataset= cb.Pool(X_train, y_train, cat_features=cat_columns)

    eval_dataset = Pool(X_test, y_test, cat_features=cat_columns)



    model = CatBoostClassifier(**params)



    model.fit(train_dataset,eval_set=eval_dataset,silent=True)

    return model.predict_proba(X_test)[::,1]
%%time

params = {

    'n_estimators': 100, 

    'max_depth': 3, 

    'learning_rate': 0.05

}

    

test_model(catboost_handler_cat,params)
from sklearn.metrics import log_loss



def experiment_lgbm(params,sample_size = 1000):

    X_train,y_train,X_test,y_test = learn_X[:sample_size],learn_y[:sample_size],learn_X[sample_size:],learn_y[sample_size:],



    features_pipeline.fit(X_train)



    X = features_pipeline.transform(X_train)

    X2 = features_pipeline.transform(X_test) 



    model = lgb.LGBMClassifier(**params)

    model.fit(X, y_train, eval_metric='binary', eval_set=[(X,y_train),(X2,y_test)],eval_names=['train','test'],verbose=False)

    ax1 = lgb.plot_metric(model, metric='binary_logloss')

    plt.show()

    print("Evaluation results:")

    for title in model.evals_result_.keys():

        r = model.evals_result_[title]

        for metric in r.keys():

            r_m = r[metric]

            print("Type:",title,"\tmetric:",metric,"\tmin: %.5f, max:%.5f" % (np.min(r_m), np.max(r_m)))

    return model,features_pipeline



def calc_validation(model,f_pipeline):

    X = f_pipeline.transform(val_X)

    y_pred_proba = model.predict_proba(X)[:,1]

    loss = log_loss(val_y, y_pred_proba)

    print("Validation LogLoss: %.4f" % loss)
params = {

    'n_estimators': 100, 

    'max_depth': 3, 

    'learning_rate': 0.05,

    'objective': 'binary',

    'metric': ['binary_logloss']

}

m,p = experiment_lgbm(params,sample_size = 1000)

calc_validation(m,p)
params = {

    'n_estimators': 15, 

    'max_depth': 3, 

    'learning_rate': 0.05,

    'objective': 'binary',

    'metric': ['binary_logloss']

}

m,p = experiment_lgbm(params,sample_size = 1000)

calc_validation(m,p)
def experiment_lgbm_full_learn(params):

    features_pipeline.fit(learn_X)

    X = features_pipeline.transform(learn_X)

    model = lgb.LGBMClassifier(**params)

    model.fit(X, learn_y,verbose=False)

    y_pred_proba = model.predict_proba(X)[:,1]

    loss = log_loss(learn_y, y_pred_proba)

    print("Train LogLoss: %.4f" % loss)



    return model,features_pipeline



m,p = experiment_lgbm_full_learn(params)

calc_validation(m,p)