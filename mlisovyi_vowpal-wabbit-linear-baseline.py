import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import os
print(os.listdir("../input"))
df_trn = pd.read_csv('../input/train.csv')
df_tst = pd.read_csv('../input/test.csv')
print('Train and test shapes are: {}, {}'.format(df_trn.shape, df_tst.shape))
print('Train and test memory footprint: {:.2f} MB, {:.2f} MB'
      .format(df_trn.memory_usage(deep=True).sum()/ 1024**2,
              df_tst.memory_usage(deep=True).sum()/ 1024**2)
     )
w_pos = df_trn['target'].sum()/df_trn.shape[0]
print('Fraction of positive target (insencere) = {:.4f}'.format(w_pos))
df_trn.head()
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
import string
def remove_punctuation(s):
    s = ''.join([i for i in s if i not in frozenset(string.punctuation)])
    return s

X_trn = (df_trn['question_text']
         .apply(remove_punctuation)
         .apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
X_trn2 = (df_trn['question_text']
         .apply(remove_punctuation)
         #.apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
X_tst = (df_tst['question_text']
         .apply(remove_punctuation)
         #.apply(lambda x: ' '.join([w.lower() for w in x.split(' ') if w.lower() not in stops]))
        )
y_trn = df_trn['target']

del df_trn, df_tst
X_trn.head()
import vowpalwabbit as vw
from vowpalwabbit.sklearn_vw import VWClassifier

# VW uses 1/-1 target variables for classification instead of 1/0, so we need to apply mapping
def convert_labels_sklearn_to_vw(y_sklearn):
    return y_sklearn.map({1:1, 0:-1})

# The function to create VW-compatible inputs from the text features and the target
def to_vw(X, y=None, namespace='Name', w=None):
    labels = '1' if y is None else y.astype(str)
    if w is not None:
        labels = labels + ' ' + np.round(y.map({1: w, -1: 1}),5).astype(str)
    prefix = labels + ' |' + namespace + ' '
    if isinstance(X, pd.DataFrame):
        return prefix + X.apply(lambda x: ' '.join(x), axis=1)
    elif isinstance(X, pd.Series):
        return prefix + X
mdl_inputs = {
# VW1x is analogous to the configuration from the kernel cited in the intro
#                 'VW1x': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=1, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5, 
#                               'b':29, 'ngram':2, 'skips': 1, 
#                               'l1':3.4742122764e-09, 'l2':1.24232077629e-11},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
                'VW1': [VWClassifier(quiet=False, convert_to_vw=False, 
                                     passes=3, link='logistic',
                                     random_seed=314),
                             {'pos_threshold':0.5},
                             {},
                             None,
                             None,
                             1./w_pos
                            ],
#                 'VW2': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=5, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
#                 'VW3': [VWClassifier(quiet=False, convert_to_vw=False, 
#                                      passes=10, link='logistic',
#                                      random_seed=314),
#                              {'pos_threshold':0.5},
#                              {},
#                              None,
#                              None,
#                              1./w_pos
#                             ],
         }

# for i in [22]:
#     mdl_inputs['VW_passes3_thrs{}'.format(i)] = mdl_inputs['VW_passes3'].copy()
#     mdl_inputs['VW_passes3_thrs{}'.format(i)][1] = {'pos_threshold':i/100.}
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.base import clone, ClassifierMixin, RegressorMixin

def train_single_model(clf_, X_, y_, random_state_=314, opt_parameters_={}, fit_params_={}):
    '''
    A wrapper to train a model with particular parameters
    '''
    c = clone(clf_)
    
    param_dict = {}
    if 'VW' in type(c).__name__:
        # we need to get ALL parameters, as the VW instance is destroyed on set_params
        param_dict = c.get_params()
        # the threshold is lost in the cloning
        param_dict['pos_threshold'] = clf_.pos_threshold
        param_dict.update(opt_parameters_)
        # the random_state is random_seed so far
        param_dict.update({'random_seed': random_state_})
        if hasattr(c, 'fit_'):
            # reset VW if it has already been trained
            c.get_vw().finish()
            c.vw_ = None 
    else:
        param_dict = opt_parameters_
        param_dict['random_state'] = random_state_
    # Set pre-configured parameters
    c.set_params(**param_dict)
    #print('Threshold = ',c.pos_threshold)
    
    return c.fit(X_, y_, **fit_params_)

def train_model_in_CV(model, X, y, metric, metric_args={},
                            model_name='xmodel',
                            seed=31416, n=5,
                            opt_parameters_={}, fit_params_={},
                            verbose=True,
                            groups=None, 
                            y_eval=None,
                            w_=1.):
    # the list of classifiers for voting ensable
    clfs = []
    # performance 
    perf_eval = {'score_i_oof': 0,
                 'score_i_ave': 0,
                 'score_i_std': 0,
                 'score_i': []
                }

    cv = KFold(n, shuffle=True, random_state=seed) #Stratified

    scores = []
    clfs = []

    for n_fold, (trn_idx, val_idx) in enumerate(cv.split(X, (y!=0).astype(np.int8), groups=groups)):
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_trn_vw = to_vw(X_trn, convert_labels_sklearn_to_vw(y_trn), w=w_).values
        X_val_vw = to_vw(X_val, convert_labels_sklearn_to_vw(y_val), w=w_).values

        #display(y_trn.head())
        clf = train_single_model(model, X_trn_vw, None, 314+n_fold, opt_parameters_, fit_params_)
        #plt.hist(clf.decision_function(X_val_vw), bins=50)
        
        if 'VW' in type(clf).__name__:
            x_thres = np.linspace(0.05, 0.95, num=37)
            y_f1    = []
            for thres in x_thres:
                # predict on the validation sample
                y_pred_tmp = (clf.decision_function(X_val_vw) > thres).astype(int)
                y_f1.append(metric(y_val, y_pred_tmp, **metric_args))
            i_opt = np.argmax(y_f1)

            clf.pos_threshold = x_thres[i_opt]
            #print('Optimal threshold = {:.4f}'.format(clf.pos_threshold))
        
        # predict on the validation sample
        y_pred_tmp = (clf.decision_function(X_val_vw) > clf.pos_threshold).astype(int)
        #store evaluated metric
        scores.append(metric(y_val, y_pred_tmp, **metric_args))
        
        # store the model
        clfs.append(('{}{}'.format(model_name,n_fold), clf))
        
        #cleanup
        del X_trn, y_trn, X_val, y_val, y_pred_tmp, X_trn_vw, X_val_vw

    #plt.show()
    perf_eval['score_i_oof'] = 0
    perf_eval['score_i'] = scores            
    perf_eval['score_i_ave'] = np.mean(scores)
    perf_eval['score_i_std'] = np.std(scores)

    return clfs, perf_eval, None

def print_perf_clf(name, perf_eval, fmt='.4f'):
    print('Performance of the model:')    
    print('Mean(Val) score inner {} Classifier: {:{fmt}}+-{:{fmt}}'.format(name, 
                                                                       perf_eval['score_i_ave'],
                                                                       perf_eval['score_i_std'],
                                                                       fmt=fmt
                                                                     ))
    print('Min/max scores on folds: {:{fmt}} / {:{fmt}}'.format(np.min(perf_eval['score_i']),
                                                            np.max(perf_eval['score_i']),
                                                            fmt=fmt
                                                           ))
    print('OOF score inner {} Classifier: {:{fmt}}'.format(name, perf_eval['score_i_oof'], fmt=fmt))
    print('Scores in individual folds: [{}]'
          .format(' '.join(['{:{fmt}}'.format(c, fmt=fmt) 
                            for c in perf_eval['score_i']
                           ])
                 )
         )
%%time
from sklearn.metrics import f1_score

mdls = {}
results = {}
y_oofs = {}
for name, (mdl, mdl_pars, fit_pars, y_, g_, w_) in mdl_inputs.items():
    print('--------------- {} -----------'.format(name))
    mdl_, perf_eval_, y_oof_ = train_model_in_CV(mdl, X_trn2.iloc[:],
                                                  y_trn.iloc[:], f1_score, 
                                                  metric_args={},
                                                  model_name=name, 
                                                  opt_parameters_=mdl_pars,
                                                  fit_params_=fit_pars, 
                                                  n=5,
                                                  verbose=500, 
                                                  groups=g_, 
                                                  y_eval=None if 'LGBMRanker' not in type(mdl).__name__ else y_rnk_eval,
                                                  w_=w_
                                                )
    results[name] = perf_eval_
    mdls[name] = mdl_
    y_oofs[name] = y_oof_
    print_perf_clf(name, perf_eval_)
%%time
y_subs= {}
X_tst_vw = to_vw(X_tst, None).values
for c in mdl_inputs:
    mdls_= mdls[c]
    y_sub = np.zeros(X_tst_vw.shape[0])
    for mdl_ in mdls_:
        y_sub += mdl_[1].decision_function(X_tst_vw)
    y_sub /= len(mdls_)
    
    y_subs[c] = y_sub
df_sub = pd.read_csv('../input/sample_submission.csv')
s= 'VW1'#VW_passes3_w
df_sub['prediction'] = (y_subs[s] > np.median([mdl_[1].pos_threshold for mdl_ in mdls[s]])).astype(int)
df_sub.to_csv('submission.csv', index=False)
!head submission.csv
