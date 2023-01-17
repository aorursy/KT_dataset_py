from collections import defaultdict

from operator import itemgetter

import string

import random

from typing import Tuple, List
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import OrdinalEncoder
import catboost

from catboost import CatBoostClassifier, Pool

import h2o

from h2o.estimators.random_forest import H2ORandomForestEstimator

import lightgbm as lgb

import sklearn

from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
print('Versions:')

print('CatBoost: {}'.format(catboost.__version__))

print('H2O: {}'.format(h2o.__version__))

print('LightGBM: {}'.format(lgb.__version__))

print('scikit-learn: {}'.format(sklearn.__version__))

print('XGBoost: {}'.format(xgb.__version__))
sns.set()

sns.set_context('talk')
h2o.init()

h2o.no_progress()
def return_positive_semi_definite_matrix(n_dim: int) -> np.ndarray:

    """Return positive semi-definite matrix.

    

    Args:

        n_dim (int): size of square matrix to return

    Returns:

        p (np.array): positive semi-definite array of shape (n_dim, n_dim)

    """

    m = np.random.randn(n_dim, n_dim)

    p = np.dot(m, m.T)

    return p
def sigmoid(x: np.array) -> np.array:                                                                  

    """Return sigmoid(x) for some activations x.

    

    Args:

        x (np.array): input activations

    Returns:

        s (np.array): sigmoid(x)

    """

    s = 1 / (1 + np.exp(-x))

    return s
def return_weak_features_and_targets(

    num_features: int,

    num_samples: int,

    mixing_factor: float,

) -> Tuple[np.array, np.array]:

    """Return weakly predictive features and a target variable.

    

    Create a multivariate Gaussian-distributed set of features and a 

    response variable that is conditioned on a weighted sum of the features.

    

    Args:

        num_features (int): number of variables in Gaussian distribution

        num_samples (int): number of samples to take

        mixing_factor (float): squashes the weighted sum into the linear 

            regime of a sigmoid.  Smaller numbers squash closer to 0.5.

    Returns:

        X (np.array): weakly predictive continuous features 

            (num_samples, num_features)

        Y (np.array): targets (num_samples,)

    """

    cov = return_positive_semi_definite_matrix(num_features)

    X = np.random.multivariate_normal(

        mean=np.zeros(num_features), cov=cov, size=num_samples)

    weights = np.random.randn(num_features)

    y_probs = sigmoid(mixing_factor * np.dot(X, weights))

    y = np.random.binomial(1, p=y_probs)

    return X, y
def return_c_values(cardinality: int) -> Tuple[list, list]:

    """Return categorical values for C+ and C-.

    

    Create string values to be used for the categorical variable c.

    We build two sets of values C+ and C-.  All values from C+ end with

    "A" and all values from C- end with "B". The cardinality input 

    determines len(c_pos) + len(c_neg). 



    Args:

        cardinality (int): cardinality of c

    Returns:

        c_pos (list): categorical values from C+ sample

        c_neg (list): categorical values from C- sample

    """

    suffixes = [

        "{}{}".format(i, j) 

        for i in string.ascii_lowercase 

        for j in string.ascii_lowercase]

    c_pos = ["{}A".format(s) for s in suffixes][:int(cardinality/2)]

    c_neg = ["{}B".format(s) for s in suffixes][:int(cardinality/2)] 

    return c_pos, c_neg    
def return_strong_features(

    y_vals: np.array, 

    cardinality: int, 

    z_pivot: int=10

) -> Tuple[np.array, np.array]:                       

    """Return strongly predictive features.

    

    Given a target variable values `y_vals`, create a categorical variable 

    c and continuous variable z such that y is perfectly predictable from 

    c and z, with y = 1 iff c takes a value from C+ OR z > z_pivot.



    Args:

        y_vals (np.array): targets

        cardinality (int): cardinality of the categorical variable, c

        z_pivot (float): mean of z

    Returns:

        c (np.array): strongly predictive categorical variable 

        z (np.array): strongly predictive continuous variable

    """

    z = np.random.normal(loc=z_pivot, scale=5, size=2 * len(y_vals))

    z_pos, z_neg = z[z > z_pivot], z[z <= z_pivot]

    c_pos, c_neg = return_c_values(cardinality)

    c, z = list(), list()

    for y in y_vals:

        coin = np.random.binomial(1, 0.5)

        if y and coin:

            c.append(random.choice(c_pos + c_neg))

            z.append(random.choice(z_pos))

        elif y and not coin:

            c.append(random.choice(c_pos))

            z.append(random.choice(z_neg))

        else:

            c.append(random.choice(c_neg))

            z.append(random.choice(z_neg))

    return np.array(c), np.array(z)
def return_main_dataset(

    num_weak: int,

    num_samp: int,

    cardinality: int=100,

    mixing_factor: float=0.025,

) -> pd.DataFrame:

    """Generate training samples.

    

    Generate a dataset with features c and z that are perfectly predictive 

    of y and additional features x_i that are weakly predictive of y and

    correlated with eachother. 

    

    Args:

        num_weak (int): number of weakly predictive features x_i to create

        num_samp (int): number of sample to create

        cardinality (int): cardinality of the predictive categorical variable.

          half of these values will be correlated with y=1 and the other 

          with y=0.

        mixing_factor (float): see `return_weak_features_and_targets`

    Returns:

        df (pd.DataFrame): dataframe with y, z, c, and x_i columns

    """

    X, y = return_weak_features_and_targets(num_weak, num_samp, mixing_factor)

    c, z = return_strong_features(y, cardinality)

    xcol_names = ['x{}'.format(i) for i in range(num_weak)]

    df = pd.DataFrame(X, columns=xcol_names)

    df['y'] = y

    df['z'] = z

    df['c'] = c

    df['c'] = df['c'].astype('category')

    df = df[['y', 'c', 'z'] + xcol_names]

    return df
def encode_as_onehot(df_main: pd.DataFrame) -> pd.DataFrame:

    """Replace string values for c with one-hot encoding."""

    df_onehot = pd.get_dummies(df_main, 'c')

    df_onehot['y'] = df_main['y'].copy()

    return df_onehot



def encode_as_int(df_main: pd.DataFrame) -> pd.DataFrame:

    """Replace string values for c with integer encoding."""

    ord_enc = OrdinalEncoder(dtype=np.int)

    c_encoded = ord_enc.fit_transform(df_main[['c']])

    df_catnum = df_main.copy()

    df_catnum['c'] = c_encoded

    df_catnum['c'] = df_catnum['c'].astype('category')

    return df_catnum, ord_enc

    

def encode_as_magic_int(df_main: pd.DataFrame) -> pd.DataFrame:

    """Replace string values for c with "magic" integer encoding.

    

    A magic encoding is one in which the sorted integer values keep all 

    C+ values (values of c that end with "A") next to each other and all 

    C- values (values of c that end with "B") next to eachother.   

    """

    values = sorted(df_main['c'].unique(), key=lambda x: x[-1])

    ord_enc = OrdinalEncoder(categories=[values], dtype=np.int)

    c_encoded = ord_enc.fit_transform(df_main[['c']])

    df_catnum = df_main.copy()

    df_catnum['c'] = c_encoded

    df_catnum['c'] = df_catnum['c'].astype('category')

    return df_catnum, ord_enc
datasets_sml = {}

encoders_sml = {}

datasets_sml['main'] = return_main_dataset(

    num_weak=2, num_samp=20, cardinality=4, mixing_factor=0.025)

datasets_sml['int'], encoders_sml['int'] = encode_as_int(datasets_sml['main'])

datasets_sml['mint'], encoders_sml['mint'] = encode_as_magic_int(datasets_sml['main'])

datasets_sml['ohe'] = encode_as_onehot(datasets_sml['main'])
datasets_sml['main'].head()
datasets_sml['main'].dtypes
datasets_sml['int'].head()
datasets_sml['int'].dtypes
datasets_sml['mint'].head()
datasets_sml['mint'].dtypes
datasets_sml['ohe'].head()
datasets_sml['ohe'].dtypes
datasets = {}

encoders = {}

datasets['main'] = return_main_dataset(

num_weak=100, num_samp=10_000, cardinality=200, mixing_factor=0.025)

datasets['int'], encoders['int'] = encode_as_int(datasets['main'])

datasets['mint'], encoders['mint'] = encode_as_magic_int(datasets['main'])

datasets['ohe'] = encode_as_onehot(datasets['main'])
fig, axes = plt.subplots(1, 2, figsize=(15,9))

bool_mask = datasets['main']['y']==1

yticks = np.array([10, 30, 50, 70, 90, 110, 130, 150, 170, 190])



titles = ['normal integer encoding', 'magic integer encoding']

ds_names = ['int', 'mint']

for iax, (title, ds_name) in enumerate(zip(titles, ds_names)):



    ax = axes[iax]

    df = datasets[ds_name]

    

    xpts = df.loc[~bool_mask, 'z']

    ypts = df.loc[~bool_mask, 'c']

    ax.scatter(xpts, ypts, s=6, label='y=0')



    xpts = df.loc[bool_mask, 'z']

    ypts = df.loc[bool_mask, 'c']

    ax.scatter(xpts, ypts, s=6, label='y=1')



    ax.set_xlabel('z')

    ax.set_ylabel('c (string, integer encoding)')

    ax.set_ylim(-20, 220)

    ax.legend()

    ax.set_title(title)



    ax.set_yticks(yticks)

    str_ytls = encoders[ds_name].inverse_transform(yticks.reshape(-1,1)).flatten()

    str_ytls = ['({},{})'.format(s,i) for i,s in zip(yticks, str_ytls)]

    ax.set_yticklabels(str_ytls);



plt.suptitle('Strong Features')

plt.tight_layout()

plt.subplots_adjust(top=0.85)
fig, axes = plt.subplots(

    2, 3, sharex=True, sharey=True, figsize=(15,12))

dataset = datasets['main']

bool_mask = dataset['y']==1



axes = axes.flatten()

for iax, (xa, xb) in enumerate([

    ('x0', 'x1'), ('x10', 'x11'), ('x20', 'x21'), 

    ('x30', 'x31'), ('x40', 'x41'), ('x50', 'x51'),

]):

    ax = axes[iax]

    

    sns.kdeplot(

        dataset.loc[bool_mask, xa], 

        dataset.loc[bool_mask, xb],

        cmap='Reds', 

        shade=False, 

        shade_lowest=False, 

        ax=ax, 

        label='y=1',

        levels=5)



    sns.kdeplot(

        dataset.loc[~bool_mask, xa], 

        dataset.loc[~bool_mask, xb],

        cmap='Blues',

        shade=False,

        shade_lowest=False, 

        ax=ax, 

        label='y=0',

        levels=5,

        linestyles='--'

    )



    ax.set_xlabel(xa)

    ax.set_ylabel(xb)

    ax.set_xlim(-20, 20)

    ax.set_ylim(-20, 20)

    ax.legend()

    

plt.suptitle('Weak Features')

plt.tight_layout()

plt.subplots_adjust(top=0.9)
metric = roc_auc_score

target_col = 'y'

n_splits = 10

test_size = 0.3
def get_feature_names(df, include_c):

    names = [f for f in df.columns if not f.startswith('y')]

    if not include_c:

        names = [f for f in names if not f.startswith('c')]

    return names



def print_auc_mean_std(results):

    print("    AUC: mean={:4.4f}, sd={:4.4f}".format(

        np.mean(results['metric']), np.std(results['metric'])))

    

def print_sorted_mean_importances(results, n=5):

    data = defaultdict(list)

    imps = results['importances']

    for d in imps:

        for fname, imp in d.items():

            data[fname].append(imp)

    mu = {fname: np.mean(vals) for fname, vals in data.items()}

    mu = sorted(mu.items(), key=itemgetter(1), reverse=True)[:n]

    print("    Importances:")

    for fname, val in mu:

        print("{:>20}: {:0.03f}".format(fname, val))
def evaluate_sklearn_model(df_data, feature_names, model):

    metrics, feature_importances = list(), list()

    X = df_data[feature_names]

    y_true = df_data[target_col]

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    for train_idx, test_idx in folds.split(X, y_true):

        model.fit(X.loc[train_idx], y_true.loc[train_idx])

        y_pred = model.predict_proba(X.loc[test_idx])

        metrics.append(metric(y_true[test_idx], y_pred[:, 1]))

        try:

            feature_importances.append(

                dict(zip(feature_names, model.feature_importances_)))

        except AttributeError:  # not a random forest

            feature_importances.append(

                dict(zip(feature_names, model.coef_.ravel()))

            )

    return {'metric': metrics, 'importances': feature_importances}
skl_trials = [

    {

        'name': 'no c',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=False),

            'model': DecisionTreeClassifier(),

        }

    },

    {

        'name': 'onehot encoding',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=True),

            'model': DecisionTreeClassifier(),

        }

    },

    {

        'name': 'int encoding (normal)',

        'params': {

            'df_data': datasets['int'],

            'feature_names': get_feature_names(datasets['int'], include_c=True),

            'model': DecisionTreeClassifier(),

        }

    },

    {

        'name': 'int encoding (magic)',

        'params': {

            'df_data': datasets['mint'],

            'feature_names': get_feature_names(datasets['mint'], include_c=True),

            'model': DecisionTreeClassifier(),

        }

    }

]
for trial in skl_trials:

    results = evaluate_sklearn_model(**trial['params'])

    print('trial: {}'.format(trial['name']))

    print_auc_mean_std(results)

    print_sorted_mean_importances(results)

    print()
class H2ODecisionTree:                                                              

    """                                                                             

    Simple class that overloads an H2ORandomForestEstimator to mimic a              

    decision tree classifier. Only train, predict and varimp are implemented.       

    """ 

    def __init__(self):

        self.model = None



    def train(self, x, y, training_frame):

        self.model = H2ORandomForestEstimator(ntrees=1, mtries=len(x))

        self.model.train(x=x, y=y, training_frame=training_frame)



    def predict(self, frame):

        return self.model.predict(frame)

    

    def varimp(self):

        return self.model.varimp()
def evaluate_h2o_model(df_data, feature_names, model):

    h2ofr = h2o.H2OFrame(df_data)

    h2ofr.col_names = list(df_data.columns)

    if 'c' in df_data:

        h2ofr['c'] = h2ofr['c'].asfactor() # make categorical

    metrics, feature_importances = list(), list()

    

    X = df_data[feature_names]

    y = df_data[target_col]

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)

    for train_idx, test_idx in folds.split(X, y):

        train_idx, test_idx = sorted(train_idx), sorted(test_idx)

        model.train(x=feature_names, y=target_col, training_frame=h2ofr[train_idx, :])

        predictions = model.predict(h2ofr[test_idx, feature_names]).as_data_frame()

        try:

            prediction_scores = predictions['True']

        except KeyError:

            prediction_scores = predictions['predict']

        metrics.append(metric(df_data[target_col].values[test_idx], prediction_scores))

        feature_importances.append(dict([(v[0], v[3]) for v in model.varimp()]))

    return {'metric': metrics, 'importances': feature_importances}
h2o_trials = [

    {

        'name': 'no c',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=False),

            'model': H2ODecisionTree(),

        }

    },

    {

        'name': 'onehot encoding',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=True),

            'model': H2ODecisionTree(),

        }

    },

    {

        'name': 'int encoding (normal)',

        'params': {

            'df_data': datasets['int'],

            'feature_names': get_feature_names(datasets['int'], include_c=True),

            'model': H2ODecisionTree(),

        }

    },

    {

        'name': 'int encoding (magic)',

        'params': {

            'df_data': datasets['mint'],

            'feature_names': get_feature_names(datasets['mint'], include_c=True),

            'model': H2ODecisionTree(),

        }

    },

    {

        'name': 'main',

        'params': {

            'df_data': datasets['main'],

            'feature_names': get_feature_names(datasets['main'], include_c=True),

            'model': H2ODecisionTree(),

        }

    },

]
for trial in h2o_trials:

    results = evaluate_h2o_model(**trial['params'])

    print('trial: {}'.format(trial['name']))

    print_auc_mean_std(results)

    print_sorted_mean_importances(results)

    print()
class LightGBMDecisionTree:

    

    def __init__(self):

        self.params = {

            'objective': 'binary',

            'bagging_freq': 0,

        }

        

    def train(self, lgb_train):

        self.gbm = lgb.train(

            self.params, 

            lgb_train, 

            num_boost_round=1)

    

    def predict(self, X):

        return self.gbm.predict(X)
def evaluate_lgb_model(df_data, feature_names, model):        

    metrics, feature_importances = list(), list()

    XX = df_data[feature_names]

    yy = df_data[target_col]

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size) 

    

    for train_idx, test_idx in folds.split(XX, yy):

        

        lgb_train = lgb.Dataset(

            XX.loc[train_idx], 

            label=yy.loc[train_idx],

#            categorical_feature=['c'],  # we use categorical pandas column

        )

        lgb_test = lgb.Dataset(

            XX.loc[test_idx], 

            label=yy.loc[test_idx],

#            categorical_feature=['c'],   # we use categorical pandas column

        )

        

        model.train(lgb_train)

        yy_pred = model.predict(XX.loc[test_idx])

        

        metrics.append(roc_auc_score(yy.loc[test_idx], yy_pred))

        fi = model.gbm.feature_importance(importance_type='gain')

        fi = fi / fi.sum()

        feature_importances.append(

            {name: val for name, val in zip(model.gbm.feature_name(), fi)}

        )

    return {'metric': metrics, 'importances': feature_importances}
lgb_trials = [

    {

        'name': 'no c',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=False),

            'model': LightGBMDecisionTree(),

        }

    },

    {

        'name': 'onehot encoding',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=True),

            'model': LightGBMDecisionTree(),

        }

    },

    {

        'name': 'int encoding (normal)',

        'params': {

            'df_data': datasets['int'],

            'feature_names': get_feature_names(datasets['int'], include_c=True),

            'model': LightGBMDecisionTree(),

        }

    },

    {

        'name': 'int encoding (magic)',

        'params': {

            'df_data': datasets['mint'],

            'feature_names': get_feature_names(datasets['mint'], include_c=True),

            'model': LightGBMDecisionTree(),

        }

    },

]
for trial in lgb_trials:

    results = evaluate_lgb_model(**trial['params'])

    print('trial: {}'.format(trial['name']))

    print_auc_mean_std(results)

    print_sorted_mean_importances(results)

    print()
class XGBoostDecisionTree:

    

    def __init__(self):

        self.params = {

            'booster': 'gbtree',

            'subsample': 1.0,

            'num_parallel_tree': 1,

            'objective': 'binary:logistic',

        }

        

    def train(self, xgb_train):

        self.bst = xgb.train(

            self.params, 

            xgb_train, 

            num_boost_round=1)

    

    def predict(self, X):

        return self.bst.predict(X)
def evaluate_xgb_model(df_data, feature_names, model):        

    metrics, feature_importances = list(), list()

    

    # XGBoost does not play well with dtype='category'

    if 'c' in df_data.columns:

        df_data['c'] = df_data['c'].astype('int')

        

    XX = df_data[feature_names]

    yy = df_data[target_col]

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size) 

    

    for train_idx, test_idx in folds.split(XX, yy):

        

        xgb_train = xgb.DMatrix(

            XX.loc[train_idx], 

            label=yy.loc[train_idx],

        )

        xgb_test = xgb.DMatrix(

            XX.loc[test_idx], 

            label=yy.loc[test_idx],

        )

        

        model.train(xgb_train)

        yy_pred = model.predict(xgb_test)

        

        metrics.append(roc_auc_score(yy.loc[test_idx], yy_pred))

        

        fi = model.bst.get_score(importance_type='gain')

        total = sum(fi.values())

        fi = {k: v/total for k,v in fi.items()}

        feature_importances.append(fi)



    # undo any dtype changes

    if 'c' in df_data.columns:

        df_data['c'] = df_data['c'].astype('category')

        

    return {'metric': metrics, 'importances': feature_importances}
xgb_trials = [

    {

        'name': 'no c',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=False),

            'model': XGBoostDecisionTree(),

        }

    },

    {

        'name': 'onehot encoding',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=True),

            'model': XGBoostDecisionTree(),

        }

    },

    {

        'name': 'int encoding (normal)',

        'params': {

            'df_data': datasets['int'],

            'feature_names': get_feature_names(datasets['int'], include_c=True),

            'model': XGBoostDecisionTree(),

        }

    },

    {

        'name': 'int encoding (magic)',

        'params': {

            'df_data': datasets['mint'],

            'feature_names': get_feature_names(datasets['mint'], include_c=True),

            'model': XGBoostDecisionTree(),

        }

    },

]
for trial in xgb_trials:

    results = evaluate_xgb_model(**trial['params'])

    print('trial: {}'.format(trial['name']))

    print_auc_mean_std(results)

    print_sorted_mean_importances(results)

    print()
class CatBoostDecisionTree:

    

    def __init__(self):

        self.cbc = CatBoostClassifier(

            n_estimators=1,

            subsample=1.0,

            verbose=0)

        

    def train(self, train_pool):

        self.cbc.fit(train_pool)

    

    def predict(self, X):

        return self.cbc.predict_proba(X)
def evaluate_cb_model(df_data, feature_names, model):        

    metrics, feature_importances = list(), list()

            

    XX = df_data[feature_names]

    yy = df_data[target_col]

    folds = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size) 

    

    if 'c' in df_data.columns:

        cat_features = ['c']

    else:

        cat_features = []

    

    for train_idx, test_idx in folds.split(XX, yy):

        

        train_pool = Pool(

            XX.loc[train_idx], 

            label=yy.loc[train_idx],

            cat_features=cat_features,

        )

        test_pool = Pool(

            XX.loc[test_idx], 

            label=yy.loc[test_idx],

            cat_features=cat_features

        )

                

        model.train(train_pool)

        yy_pred = model.predict(test_pool)

        

        metrics.append(roc_auc_score(yy.loc[test_idx], yy_pred[:,1]))

        

        fi = dict(zip(model.cbc.feature_names_, model.cbc.feature_importances_))

        total = sum(fi.values())

        fi = {k: v/total for k,v in fi.items()}

        feature_importances.append(fi)



        

    return {'metric': metrics, 'importances': feature_importances}
cb_trials = [

    {

        'name': 'no c',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=False),

            'model': CatBoostDecisionTree(),

        }

    },

    {

        'name': 'onehot encoding',

        'params': {

            'df_data': datasets['ohe'],

            'feature_names': get_feature_names(datasets['ohe'], include_c=True),

            'model': CatBoostDecisionTree(),

        }

    },

    {

        'name': 'int encoding (normal)',

        'params': {

            'df_data': datasets['int'],

            'feature_names': get_feature_names(datasets['int'], include_c=True),

            'model': CatBoostDecisionTree(),

        }

    },

    {

        'name': 'int encoding (magic)',

        'params': {

            'df_data': datasets['mint'],

            'feature_names': get_feature_names(datasets['mint'], include_c=True),

            'model': CatBoostDecisionTree(),

        }

    },

    {

        'name': 'main',

        'params': {

            'df_data': datasets['main'],

            'feature_names': get_feature_names(datasets['main'], include_c=True),

            'model': CatBoostDecisionTree(),

        }

    }

]
for trial in cb_trials:

    results = evaluate_cb_model(**trial['params'])

    print('trial: {}'.format(trial['name']))

    print_auc_mean_std(results)

    print_sorted_mean_importances(results)

    print()