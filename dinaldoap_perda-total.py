import os
import pprint as pp
import math as ma
import numpy as np
import pandas as pd
from plotnine import *
from plotnine.data import *
from sklearn import base
from sklearn import tree
from sklearn import naive_bayes as nb
from sklearn import svm as svm
from sklearn import neighbors as nei
from sklearn import preprocessing as pre
from sklearn import feature_extraction as fe
from sklearn import pipeline as pi
from sklearn import model_selection as ms
from sklearn import ensemble as ens
from sklearn.metrics import make_scorer
import xgboost as xgb
import bayes_opt as bo
import category_encoders as ce
# Nível de verbosidade do log:
verbose = 10

# Semente de estado randômico:
seed = round(ma.pi * 10**4)
# Extrai tipos das colunas:
def cols_types_to_df(df, cols):
    return pd.DataFrame(df[cols].dtypes.values.reshape(1, -1), columns=cols, index=['dtype'])
ID_COL = 'id'
TARGET_COL = 'target'
SPECIAL_COL_NAMES = [ID_COL, TARGET_COL]

# Carregando dados:
train = pd.read_csv('../input/train.csv').drop(ID_COL, axis=1)#.sample(frac=0.25, random_state=seed)
headers = train.columns

# Separando o rótulo:
X = train.drop(TARGET_COL, axis=1)
y = train[TARGET_COL]

train.head(10)
IND_COL_NAMES = [x for x in headers if x.startswith('ps_ind')]
cols_types_to_df(train, IND_COL_NAMES)
REG_COL_NAMES = [x for x in headers if x.startswith('ps_reg')]
cols_types_to_df(train, REG_COL_NAMES)
CAR_COL_NAMES = [x for x in headers if x.startswith('ps_car')]
cols_types_to_df(train, CAR_COL_NAMES)
CALC_COL_NAMES = [x for x in headers if x.startswith('ps_calc')]
cols_types_to_df(train, CALC_COL_NAMES)
CAT_COL_NAMES = [x for x in headers if x.endswith('cat')]
cols_types_to_df(train, CAT_COL_NAMES)
BIN_COL_NAMES = [x for x in headers if x.endswith('bin')]
cols_types_to_df(train, BIN_COL_NAMES)
LIN_COL_NAMES = [x for x in headers if (x not in (CAT_COL_NAMES + BIN_COL_NAMES + SPECIAL_COL_NAMES))]
cols_types_to_df(train, LIN_COL_NAMES)
train.describe()
target_counts = train[TARGET_COL].value_counts()
target_ratio = target_counts[0]/target_counts[1]
print("Proporção entre classes 0 e 1: %s" % target_ratio)
(ggplot(train.astype(dtype={'target': object}), aes(x=TARGET_COL, fill=TARGET_COL))
 + geom_bar()   
 + facet_wrap(TARGET_COL)
 )
def drop_cols_func(X):
    '''Remove irrelevant columns.
    
    Parameters
    ----------

    X : pandas.DataFrame
        DataFrame with the features.
        
    Returns
    -------

    p : pandas.DataFrame
        DataFrame without *calc* features.
    '''
    return X.drop(columns=CALC_COL_NAMES)

drop_cols = pre.FunctionTransformer(func=drop_cols_func, validate=False)
drop_cols.transform(X).head()
one_hot = ce.OneHotEncoder(cols=CAT_COL_NAMES, drop_invariant=True, handle_unknown='impute')
one_hot.fit_transform(X).head()
def gini(actual, pred):   
    assert( len(actual) == len(pred) )
    all = np.asarray(np.c_[ actual, pred, np.arange(len(actual)) ], dtype=np.float)    
    # ordena por coluna da classe positiva de pred e por 
    all = all[ np.lexsort((all[:,2], -1*all[:,1])) ]    
    totalLosses = all[:,0].sum()
    giniSum = all[:,0].cumsum().sum() / totalLosses
    
    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_normalized(actual, pred):    
    return gini(actual, pred[:,1]) / gini(actual, actual)

gini_normalized_scorer = make_scorer(score_func=gini_normalized, greater_is_better=True, needs_proba=True)

def cv_metric(estimator, X, y, fit_params=None):    
    pred = ms.cross_val_predict(estimator, X, y, cv=ms.StratifiedKFold(n_splits=3, random_state=seed), verbose=verbose, fit_params=fit_params, method='predict_proba')
    return gini_normalized(y, pred) 

def xgb_metric(pred, dtrain):
    print('xgb_gini_normalized')
    return 'gini', gini_normalized(dtrain.get_labels(), pred)

xgb_fit_params = {        
    #'clf__silent': 1,
    'clf__eval_metric': xgb_metric
}
xgb_params = {
    'objective': 'binary:logistic'
}

def pipeline():
    steps = [('drop_cols', drop_cols),
             ('one_hot', one_hot),
             ('clf', xgb.XGBClassifier(random_state=seed))]
    return pi.Pipeline(steps)

pipe = pipeline()
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.33, random_state=seed)
# AVISO: desativado para não extrapolar o tempo de processamento do Kaggle
'''
param_grid = {'clf': [nb.GaussianNB(), xgb.XGBClassifier(random_state=seed, **xgb_params)]}
grid_search = ms.GridSearchCV(pipe, param_grid=param_grid, cv=ms.StratifiedKFold(n_splits=3, random_state=seed), verbose=verbose, scoring=gini_normalized_scorer, return_train_score=False)
grid_search.fit(X_train, y_train)
pp.pprint(grid_search.best_score_)
pp.pprint(grid_search.best_estimator_)
'''
if os.path.isfile('cache.csv'):
    # AVISO: atualize best_configs abaixo do 'else' sempre que houver uma busca ampla por melhores parâmetros
    best_configs = pd.read_csv('cache.csv').sort_values(by='target', ascending=False)
    best_configs = best_configs.head(5).drop(columns='target').to_dict(orient='list')
    pp.pprint(best_configs)
else:
    # melhores parâmetros hard-coded serão usados em ambientes sem o arquivo 'cache.csv'
    best_configs = {'colsample_bylevel': [1.0, 1.0, 1.0, 1.0, 0.6210536972328008],
         'colsample_bytree': [0.1639149309112309, 0.4719120160563106, 0.2738881448393379, 0.9597000000000002, 0.4998605743150732],
         'gamma': [9.311242926501441, 9.106399343640415, 9.84350195229832, 1.3079, 8.495594735754976],
         'learning_rate': [0.1, 0.1, 0.1, 0.1, 0.3],
         'max_delta_step': [0.0, 0.0, 0.0, 0.0, 9.937144736135796],
         'max_depth': [3.280267137915809, 3.744769687598659, 3.079866163690399, 3.8448, 3.0243950675489177],
         'min_child_weight': [1.0, 1.0, 1.0, 1.0, 1.0],
         'n_estimators': [496.3697387347585, 496.70662978009966, 362.50656654507236, 499.9892, 350.3584899983292],
         'reg_alpha': [9.79619399515934, 0.7119072322272911, 0.05797462113553141, 8.128199999999998, 4.196426179286616],
         'reg_lambda': [1.0, 1.0, 1.0, 1.0, 9.729243825652572],
         'scale_pos_weight': [2.091440738145689, 1.7334510787353241, 1.327362597470453, 2.6812, 1.072600468578576],
         'subsample': [0.7142488578466437, 0.9885285182085388, 0.7352809336635373, 0.9199, 0.8611579899201934]}
    # AVISO: comentar a linha de código abaixo para fazer mais exploração
    #best_configs = pd.DataFrame.from_dict(best_configs).head(1).to_dict(orient='list')
def to_xgb_params(**kwargs):
    params = {}
    params['clf__max_depth'] = int(kwargs['max_depth']) 
    params['clf__learning_rate'] = max(kwargs['learning_rate'], 0)
    params['clf__n_estimators'] = int(kwargs['n_estimators'])
    params['clf__min_child_weight'] = int(kwargs['min_child_weight'])
    params['clf__max_delta_step'] = int(kwargs['max_delta_step'])
    params['clf__subsample'] = max(min(kwargs['subsample'], 1), 0)
    params['clf__colsample_bytree'] = max(min(kwargs['colsample_bytree'], 1), 0)  
    params['clf__colsample_bylevel'] = max(min(kwargs['colsample_bylevel'], 1), 0)
    params['clf__gamma'] = max(kwargs['gamma'], 0)
    params['clf__reg_alpha'] = max(kwargs['reg_alpha'], 0)
    params['clf__scale_pos_weight'] = max(kwargs['scale_pos_weight'], 0.001)
    return params

def xgb_cv(**kwargs):
    xgb = base.clone(pipe)
    xgb.set_params(**to_xgb_params(**kwargs))
    return cv_metric(xgb, X_train, y_train, fit_params=xgb_fit_params)
    
bo_opt = bo.BayesianOptimization(xgb_cv, {'max_depth': (3, 15),
                                      'learning_rate': (0.1, 0.0001),
                                      'n_estimators': (100, 500),
                                      'min_child_weight': (1, 1),
                                      'max_delta_step': (0, 10),
                                      'subsample': (0.5, 1.0),
                                      'colsample_bytree': (0.1, 1.0),
                                      'colsample_bylevel': (0.1, 1.0),
                                      'gamma': (0.0, 10.0),
                                      'reg_alpha': (0.0, 10.0),
                                      'reg_lambda': (1.0, 10.0),
                                      'scale_pos_weight': (1.0, target_ratio),
                                     },
                                    random_state=seed)
                                    
# AVISO: desativado para não extrapolar o tempo de processamento do Kaggle
'''
# manter True para execução de curta duração no Kaggle
# manter False para ampliar a busca por melhores parâmetros
fast_search = True
if fast_search:
    # usa melhores parâmetros como semente para o otimizador
    bo_opt.explore(best_configs)
else:
    if os.path.isfile('cache.csv'):
        # carrega dados da função objetivo
        bo_opt.initialize_df(pd.read_csv('cache.csv'))
    else:
        # explora configuração padrão e 5 configurações aleatórias        
        bo_opt.explore({'max_depth': [3],
                        'learning_rate': [0.1],
                        'n_estimators': [100],
                        'min_child_weight': [1],
                        'max_delta_step': [0],
                        'subsample': [1.0],
                        'colsample_bytree': [1.0],
                        'colsample_bylevel': [1.0],
                        'gamma': [0.0],
                        'reg_alpha': [0.0],
                        'reg_lambda': [1.0],
                        'scale_pos_weight': [1.0]})        
        bo_opt.maximize(init_points=5, n_iter=0, acq='ei')
        # explora as melhores parâmetros até então encontrados
        bo_opt.explore(best_configs)
        bo_opt.points_to_csv('cache.csv')        
# otimiza com 5 rodadas
bo_opt.maximize(init_points=0, n_iter=5, acq='ei')    
bo_opt.points_to_csv('cache.csv')
pp.pprint(bo_opt.res['max'])

# cria modelo otimizado
best_config = bo_opt.res['max']['max_params']
'''
# AVISO: comentar linha de código abaixo para usar resultado da otimização
best_config = {'colsample_bylevel': 0.45879276636748534,
                'colsample_bytree': 0.7824778832374746,
                'gamma': 5.489188804299045,
                'learning_rate': 0.1,
                'max_delta_step': 1.5254539296007386,
                'max_depth': 3.4967941454908598,
                'min_child_weight': 1.0,
                'n_estimators': 461.216046669665,
                'reg_alpha': 8.531590560075347,
                'reg_lambda': 9.89766553793669,
                'scale_pos_weight': 1.2379042177844632,
                'subsample': 0.798095844721928}
opt_params = to_xgb_params(**best_config)
opt_pipe = base.clone(pipe)
opt_pipe.set_params(**opt_params)
pipe.fit(X_train, y_train, **xgb_fit_params)
print('Configuração padrão: {}'.format(gini_normalized(y_test, pipe.predict_proba(X_test))))
opt_pipe.fit(X_train, y_train, **xgb_fit_params)
print('Configuração otimizada: {}'.format(gini_normalized(y_test, opt_pipe.predict_proba(X_test))))
# Wrapper implementado como workaround de um bug de "learning_cuve" do sklearn
# "learning_curve" não expõe o parâmetro "fit_params" e o Wrapper abaixo resolve isso
class FitParamsWrapper(base.BaseEstimator, base.ClassifierMixin):
    def __init__(self, estimator, fit_params):
        self.estimator = estimator
        self.fit_params = fit_params

    def fit(self, X, y=None):
        self.estimator.fit(X, y, **self.fit_params)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X, y=None):        
        return self.estimator.predict_proba(X)
    
    def score(self, X, y=None):        
        return self.estimator.score(X, y)
# AVISO: desativado para não extrapolar o tempo de processamento do Kaggle
'''
sizes, train_scores, test_scores = ms.learning_curve(FitParamsWrapper(opt_pipe, xgb_fit_params), X, y, cv=ms.StratifiedKFold(n_splits=3, random_state=seed), scoring=gini_normalized_scorer, verbose=verbose, random_state=seed)
df_lc = pd.DataFrame(data={'m': sizes, 'train': train_scores.mean(axis=1), 'test': test_scores.mean(axis=1)})
df_lc = pd.melt(df_lc, id_vars=['m'], value_vars=['train', 'test'], var_name='type', value_name='score')
(ggplot(df_lc)
 + aes(x='m', y='score', fill='type', color='type')
 + geom_line()
 )
 '''
opt_pipe.fit(X, y, **xgb_fit_params)
test = pd.read_csv('../input/test.csv')
X_test = test[X.columns]
targets = opt_pipe.predict_proba(X_test)
targets = pd.DataFrame({'id': test.id, 'target': targets[:,1]})
targets.to_csv('submission.csv', index=False)