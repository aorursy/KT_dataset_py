%%time
import pandas as pd 
import numpy as np
import gc; gc.enable()
        
import pathlib as pt

DATA_STORE = '/kaggle/working/data_kaz.h5'
PATH_TRAIN = '/kaggle/input/numerai-round-190/numerai_training_data_t.csv'
PATH_TEST = '/kaggle/input/numerai-round-190/numerai_tournament_data.csv'
TARGET = '/kaggle/input/numerai-round-190/target.csv'
ERAS = '/kaggle/input/numerai-round-190/eras_t.csv'

try:
    
    path = pt.Path(DATA_STORE)
    
    with pd.HDFStore(path) as store: 
        train = store['train']; tournament = store['tournament'];
        target = store['TARGET']; eras = store['ERAS'];
    print('[INFO] read data...')

except (ValueError, KeyError):
    
    path = pt.Path(DATA_STORE)
    
    print('[INFO] create data storage...')
    with pd.HDFStore(path) as store: store.put('train',pd.read_csv(PATH_TRAIN, header=0)\
                                        .apply(lambda x: 
                                                           x.astype(np.float16).round(2) if x.dtype==np.float64
                                        else x),
                                               compression='gzip', complevel=20, format = 'table')
        
    with pd.HDFStore(path) as store: store.put('tournament',pd.read_csv(PATH_TEST, header=0).iloc[0:500000]\
                                        .apply(lambda x: 
                                                           x.astype(np.float16).round(2) if x.dtype==np.float64
                                        else x),
                                               compression='gzip',complevel=20, format = 'table')
        
    with pd.HDFStore(path) as store: store.put('tournament',pd.read_csv(PATH_TEST, header=0).iloc[500001:1000000]\
                                        .apply(lambda x: 
                                                           x.astype(np.float16).round(2) if x.dtype==np.float64
                                        else x),
                                               compression='gzip',complevel=20, append=True, format = 'table')
        
    with pd.HDFStore(path) as store: store.put('tournament',pd.read_csv(PATH_TEST, header=0).iloc[1000001:]\
                                        .apply(lambda x: 
                                                           x.astype(np.float16).round(2) if x.dtype==np.float64
                                        else x),
                                               compression='gzip',complevel=20, append=True, format = 'table')
        
    with pd.HDFStore(path) as store:
        store.put('TARGET',pd.read_csv(TARGET, header=None), format = 'table')
        store.put('ERAS',pd.read_csv(ERAS, header=None), format = 'table')

    print('[INFO] read data...')
    with pd.HDFStore(path) as store: 
        train = store['train'];tournament = store['tournament'];
        target = store['TARGET'];eras = store['ERAS'];

numerai_benchmark = pd.read_csv('/kaggle/input/numerai-round-190/example_predictions_target_kazutsugi.csv')

# preparing train and val data
validation = tournament[tournament['data_type'] == 'validation']
target_val = validation['target_kazutsugi']

erasv = validation.era.str.slice(3).astype(int)
erast =  tournament[tournament['data_type'] == 'test'].era.str.slice(3).astype(int)

# Transform the loaded CSV data into numpy arrays
features = [f for f in list(train) if "feature" in f]

train = train[features]
val = validation[features]

test = tournament[features]
ids = tournament['id']

del tournament, validation
gc.collect()

train['era']=eras
val['era']=erasv

print(train.shape, test.shape, val.shape)
print('loading datasets has done')
from gplearn.functions import make_function
from gplearn.genetic import SymbolicClassifier
def th(x):
    return np.tanh(x)

gptanh = make_function(th, 'tanh', 1)
sample_wts = np.sqrt(np.array([x - 10.0 if x > 10.0 else 0 for x in target.values]) + 1.0)
function_set = ['add', 'sub', 'mul', 'div', 'inv', 'abs', 'neg', 'max', 'min', gptanh]
count = 1
est = SymbolicRegressor(population_size=2000,
                       generations=count,
                       tournament_size=50,  # consider 20, was 50
                       parsimony_coefficient=0.0001,  # oops: 0.0001?
                       function_set=function_set,init_depth=(6, 16),
                       metric='mean absolute error', verbose=1, random_state=42, n_jobs=-1, low_memory=True)
est.fit(train[features], target)
est.predict(val[features])
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

GENS = 500
MAE_THRESH = 2.5
MAX_NO_IMPROVE = 50
np.random.seed(666)
maes = []
gens = []
from gplearn.functions import make_function

def th(x):
    return np.tanh(x)

gptanh = make_function(th, 'tanh', 1)
sample_wts = np.sqrt(np.array([x - 10.0 if x > 10.0 else 0 for x in target.values]) + 1.0)
function_set = ['add', 'sub', 'mul', 'div', 'inv', 'abs', 'neg', 'max', 'min', gptanh]
folds = KFold(n_splits=3, shuffle=True, random_state=42)

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train[features], target.values)):
    print('working fold %d' % fold_)
    X_tr, X_val = train[features].iloc[trn_idx], train[features].iloc[val_idx]
    y_tr, y_val = target.values[trn_idx].ravel(), target.values[val_idx].ravel()
    sample_wts_tr = sample_wts[trn_idx]
    np.random.seed(5591 + fold_)
    best = 1e10
    count = 1
    imp_count = 0
    best_mdl = None
    best_iter = 0
  
    gp = SymbolicRegressor(population_size=2000,
                       generations=count,
                       tournament_size=50,  # consider 20, was 50
                       parsimony_coefficient=0.0001,  # oops: 0.0001?
                       const_range=(-16, 16),  # consider +/-20, was 100
                       function_set=function_set,
                       # stopping_criteria=1.0,
                       # p_hoist_mutation=0.05,
                       # max_samples=.875,  # was in
                       # p_crossover=0.7,
                       # p_subtree_mutation=0.1,
                       # p_point_mutation=0.1,
                       init_depth=(6, 16),
                       warm_start=True,
                       metric='mean absolute error', verbose=1, random_state=42, n_jobs=-1, low_memory=True)

    for run in range(GENS):
        mdl = gp.fit(X_tr, y_tr, sample_weight=sample_wts_tr)
        pred = gp.predict(X_val)
        mae = np.sqrt(mean_absolute_error(y_val, pred))

    if mae < best and imp_count < MAX_NO_IMPROVE:
        best = mae
        count += 1
        gp.set_params(generations=count, warm_start=True)
        imp_count = 0
        best_iter = run
        if mae < MAE_THRESH:
            best_mdl = copy.deepcopy(mdl)
    elif imp_count < MAX_NO_IMPROVE:
        count += 1
        gp.set_params(generations=count, warm_start=True)
        imp_count += 1
    else:
        break

    print('GP MAE: %.4f, Run: %d, Best Run: %d, Fold: %d' % (mae, run, best_iter, fold_))

maes.append(best)
gens.append(run)
      
print('Finish - GP MAE: %.4f, Run: %d, Best Run: %d' % (mae, run, best_iter))

preds = best_mdl.predict(val[features])
print(preds[0:12])
predictions += preds / folds.n_splits
%%time

BENCHMARK = 0.002
BAND = 0.04

TOURNAMENT_NAME = "kazutsugi"
PREDICTION_NAME = f"prediction_{TOURNAMENT_NAME}"
TARGET_NAME = f"target_{TOURNAMENT_NAME}"
train[TARGET_NAME]=target.values
val['target_kazutsugi'] = target_val.values

# The payout function
def payout(scores):
    return ((scores - BENCHMARK)/BAND).clip(lower=-1, upper=1)

def score(df):
    # method="first" breaks ties based on order in array
    return np.corrcoef(df['target_kazutsugi'], df[PREDICTION_NAME].rank(pct=True, method="first"))[0,1]

def evaluation_test(train,val):
    # Check the per-era correlations on the training set
    train_correlations = train.groupby('era').apply(score)
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")

    # Check the per-era correlations on the validation set
    validation_correlations = val.groupby('era').apply(score)
    print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")
    
def create_sub(results, ids):
    import time
    from time import gmtime, strftime
    results_df = pd.DataFrame(data={'probability_kazutsugi': results})
    strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
    joined = pd.DataFrame(ids).join(results_df)
    path = 'predictions_{:}'.format(strftime("%Y-%m-%d_%Hh%Mm%Ss", time.gmtime())) + '.csv'
    print("Writing predictions to " + path.strip())
    joined.to_csv(path, float_format='%.15f', index=False)
    
# multy threading module  '------------------------------------------------------------------------------------------------------------------'
from multiprocessing.pool import ThreadPool
from functools import partial

class parallelization:
    
    @staticmethod
    def map_parallel(fn, lst):
        with ThreadPool(processes=3) as pool:
            return pool.map(fn, lst)

    @staticmethod
    def compute_(est, X, y=None):
        if y is not None:
            
            return est.fit(X, y)
        else:
            return est.predict(X)

    def run_compile(self, models, X_tr, y_tr=None):
        if y_tr is None:
            return self.map_parallel(partial(self.compute_, X=X_tr), models)
        else:
            return self.map_parallel(partial(self.compute_, X=X_tr, y=y_tr), models)
%%time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['OMP_NUM_THREADS'] = '1'

from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet, Lasso
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA 
pca = PCA(n_components=.95)
pca.fit(train[features])

reg1 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.06, 
                     n_estimators=600, max_depth=4,subsample=0.82,
                     nthread=5)

reg2 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.06, 
                     n_estimators=100, max_depth=5,subsample=0.82,
                     nthread=5)

reg3 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.07, 
                    n_estimators=400, num_leaves = 65,subsample=0.82,
                    nthread=5)

reg4 = ElasticNet(alpha=1e-05)

reg5 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.05, 
                    n_estimators=400, max_depth=7,subsample=0.82,
                    nthread=5)

reg6 = LGBMRegressor(n_jobs=1, colsample_bytree=0.1, learning_rate=0.002, 
                    n_estimators=200, max_depth=7,subsample=0.82,
                    nthread=5)

models_list = [reg1,reg2,reg3,reg4,reg5,reg6]

models_input = VotingRegressor((
    ('reg1',reg1), ('reg2',reg2), ('reg3',reg3),
    ('reg4',reg4), ('reg5',reg5), ('reg6',reg6)
))

model_reg = Pipeline([('reg', models_input)])

paralle = parallelization()

fitedmodels = paralle.run_compile(model_reg,X_tr=pca.transform(train[features]), y_tr=target.values.ravel())
# predict
pred_train_ = paralle.run_compile(fitedmodels,pca.transform(train[features]))[0]
pred_val_ = paralle.run_compile(fitedmodels,pca.transform(val[features]))[0]
pred_test_ = paralle.run_compile(fitedmodels,pca.transform(test[features]))[0]

# FIRST LEVEL  '--------------------------------------------------------------------------------------------------------------------'
fitedmodels = paralle.run_compile(models_list,pca.transform(train[features]), target.values.ravel())
sx_train_ = paralle.run_compile(fitedmodels,pca.transform(train[features]))
sx_test_ = paralle.run_compile(fitedmodels,pca.transform(test[features]))
sx_val_ = paralle.run_compile(fitedmodels,pca.transform(val[features]))

sx_train_ = np.vstack(sx_train_).T
sx_test_ = np.vstack(sx_test_).T
sx_val_ = np.vstack(sx_val_).T

# SECOND LEVEL '--------------------------------------------------------------------------------------------------------------------'
reg = LGBMRegressor(n_jobs=-1, colsample_bytree=0.1, learning_rate=0.01, 
                     n_estimators=400, max_depth=7,subsample=0.82,
                     nthread=5)

reg.fit(sx_train_,target.values.ravel())
    
# FINAL PREDCTION '-----------------------------------------------------------------------------------------------------------------'
train[PREDICTION_NAME] = (reg.predict(sx_train_)*.1+pred_train_*.9)
val[PREDICTION_NAME] = (reg.predict(sx_val_)*.1+pred_val_*.9)
test[PREDICTION_NAME] = (reg.predict(sx_test_)*.1+pred_test_*.9)

gc.collect()
evaluation_test(train,val)
# save submission file
ssubm = pd.read_csv("../input/numerai-round-190/example_predictions_target_kazutsugi.csv")
create_sub(test[PREDICTION_NAME], ssubm.id)
function_set = ['add', 'sub', 'mul', 'div',
                'sqrt', 'log', 'abs', 'neg', 'inv',
                'max', 'min']
gp = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0, n_jobs=3)


import gplearn
est = SymbolicClassifier(parsimony_coefficient=.01,
                         feature_names=cancer.feature_names,
                         random_state=1)
est.fit(cancer.data[:400], cancer.target[:400])