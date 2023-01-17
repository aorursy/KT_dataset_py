import os, re, gc, warnings

import numpy as np

import pandas as pd

from scipy.stats import wasserstein_distance as wd

from tqdm import tqdm_notebook

warnings.filterwarnings("ignore")

gc.collect()
DATA_DIR = '../input/'

FILES={}

for fn in os.listdir(DATA_DIR):

    FILES[ re.search( r'[^_\.]+', fn).group() ] = DATA_DIR + fn



train = pd.read_csv(FILES['train'],index_col='id')

test = pd.read_csv(FILES['test'],index_col='id')

    

CAT_COL='wheezy-copper-turtle-magic'

CATS = sorted(train[CAT_COL].unique())

R = int(np.e*1e7)
from sklearn.preprocessing import StandardScaler



from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

# from sklearn.mixture import GaussianMixture

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# from sklearn.linear_model import LogisticRegression
def fit_set(y, X, T, k, reg):



    oof=pd.Series(0.5,index=X.index,dtype='float')

    preds=pd.DataFrame(index=T.index,dtype='float')

    y_,X_,T_=[],[],[]

    

    for i in tqdm_notebook( CATS ):

        y_i=y[X[CAT_COL]==i]

        X_i=X[X[CAT_COL]==i].drop(CAT_COL,axis=1).dropna(axis=1)

        T_i=T[T[CAT_COL]==i].drop(CAT_COL,axis=1).dropna(axis=1)

        

        o,p = fit_one_model(y_i, X_i, T_i, k, reg)



        oof[o.index]=o

        preds = pd.concat([preds,p])

        X_i[CAT_COL]=i

        T_i[CAT_COL]=i

        

        y_.append(y_i)

        X_.append(X_i)

        T_.append(T_i)

    

    preds = preds.dropna()

    df_y = pd.concat([y for y in y_])

    df_X = pd.concat([x for x in X_])

    df_T = pd.concat([t for t in T_])        

        

    print(f"\n\nOut-of-fold ROC AUC for entire labeled dataset: {roc_auc_score(y.values,oof.values):.5f}")

    return df_y, df_X, df_T, oof, preds
def fit_one_model(y, X:pd.DataFrame, T:pd.DataFrame, k:int=15, reg=.5):

    # Pick a classifier

    # clf = LogisticRegression(solver='liblinear',penalty='l2',C=1.0)

    clf = QuadraticDiscriminantAnalysis(reg_param=reg, store_covariance=False, tol=1e-4)    

    # clf = GaussianMixture(n_components = 2,

    #                       covariance_type = 'full',#‘full’‘tied’‘diag’‘spherical’

    #                       max_iter = 100,

    #                       n_init = 100,

    #                       init_params = 'random', #‘kmeans’‘random’

    #                       tol = 1e-3,

    #                       reg_covar = 1e-6,

    #                       random_state = R,

    #                       warm_start = True,

    #                       verbose = 0,#0,1,2,

    #                       verbose_interval = 10,

    #                      )

    oof=pd.Series(0.5,index=X.index, dtype='float') 

    p_fold=[]

 

    i=0

    skf = StratifiedKFold(n_splits=k, random_state=R, shuffle=True)

    for tr_idx, te_idx in skf.split(X, y):

        i+=1

        clf.fit(X.iloc[tr_idx],y.iloc[tr_idx])

        oof[X.index[te_idx]]= clf.predict_proba(X.iloc[te_idx])[:,1]

        p_fold.append(pd.Series(  clf.predict_proba(T)[:,1], index=T.index, name='fold_'+str(i)  ))

    

    preds = pd.concat([fold for fold in p_fold],axis=1)

    # no k-fold training

#         clf.fit(X,y)

#         preds = pd.concat([preds,

#                            pd.Series(clf.predict_proba(T)[:,1],

#                                      index=T.index,

#                                      name='fold_'+str(i))], axis=1)





    #  Summary output for each of 512 models    

#     print(f"{len(X.columns)} features used\

#     \tIn-fold classifier score: {clf.score(X,y):.5f}\

#     \tOut-of-fold ROC AUC: {roc_auc_score(y,oof.values):.5f}")



    return oof, preds
def update_dataset(y, X, T, o=None, p=None, iter_n=0):

    y_, X_, T_=[],[],[]

    for i in tqdm_notebook(CATS):

        idx_x = (X[CAT_COL]==i)

        idx_t = (T[CAT_COL]==i)

        y_i = y[ idx_x ]

        X_i = X[ idx_x ].dropna(axis=1)

        T_i = T[ idx_t ].dropna(axis=1)



        if iter_n==1:

            y_i, X_i, T_i = first_pass(y_i, X_i, T_i)

        else:

            o_i = o[ idx_x ]

            p_i = p[ idx_t ]

            y_i, X_i, T_i = iterative_filter_augment(y_i, X_i, T_i, o_i, p_i, iter_n)

            

        X_i[CAT_COL]=i

        T_i[CAT_COL]=i

        y_.append(y_i)

        X_.append(X_i)

        T_.append(T_i)

    y_new = pd.concat([y for y in y_])

    X_new = pd.concat([x for x in X_])

    T_new = pd.concat([t for t in T_])

    return y_new, X_new, T_new
from sklearn import manifold



def first_pass(y, X, T):

    # AUGMENTATION

    # Add 1 synthetic feature from low variance features

    low_var_cols = X.loc[:,(X.var()<2)].columns

    # Sum of all the absolute values of the low variance features

    X['synthetic'] = X[low_var_cols].abs().sum(axis=1)

    T['synthetic'] = T[low_var_cols].abs().sum(axis=1)   

    

    # Remove features with low variance / low signal-to-noise

    X = X.drop(low_var_cols, axis=1)

    T = T.drop(low_var_cols, axis=1)  



    # FILTERING

    # Normalize all columns but WCTM

    scl = StandardScaler()

    scl.fit( pd.concat([X, T]) )

    X = pd.DataFrame(scl.transform(X),index=X.index, columns=X.columns)

    T = pd.DataFrame(scl.transform(T),index=T.index, columns=T.columns)



    # use generic column names, so the resulting dataframe is minimized

    # X.columns = np.arange(X.shape[1])

    # T.columns = np.arange(T.shape[1])

    return y, X, T
def iterative_filter_augment(y, X, T, o, p, iter_n): 

    # Remove samples with large distance between prediction and label

    drop_samples = ((y-o).abs().sort_values()[:int(.01*y.shape[0])]).index

    y = y.drop(drop_samples)

    X = X.drop(drop_samples)

    

    # Add samples from Test set which have confident out-of-fold predictions

    q = p.quantile(.9,axis=1)

    

    # get idx by threshold

    # idx_new = q[(q>.95) | (q<.05)].index

    # get idx by balanced %

    idx_new = pd.concat([ q.sort_values()[:int(iter_n*.11*q.shape[0])] , q.sort_values(ascending=False)[:int(iter_n*.11*q.shape[0])] ]).index

    

    # idx_new = idx_new[~idx_new.isin(X.index)]

    X = pd.concat([X, T.loc[idx_new]],axis=0).loc[~X.index.duplicated(keep='first')]

    y = pd.concat([y, q.loc[idx_new].round().astype(int)],axis=0).loc[~y.index.duplicated(keep='first')]

    return y, X, T
import matplotlib.pyplot as plt

def show_preds(p:pd.DataFrame):

    fig = plt.figure(figsize = (20,5))

    p.quantile(.1,axis=1).hist(bins=20,alpha=.3,color='green')

    p.quantile(.9,axis=1).hist(bins=20,alpha=.3,color='red')

    p.quantile(.5,axis=1).hist(bins=20,alpha=.3,color='yellow')

    display(f'p: {p.shape}    (green=10th, yellow=50th, red=90th percentile)')

    return
from sklearn import metrics

def compare_oof_target(o:pd.Series):  

    fig = plt.figure(figsize = (20,5))

    train['target'].hist(bins=20,alpha=1,color='red')

    o.hist(bins=20,alpha=.5,color='green')

    display(f'oof: {o.shape}    train:{train.shape}    (red=original train target, green=oof predictions)')

    

    fig = plt.figure(figsize = (20,10))

    fpr, tpr, thresholds = metrics.roc_curve(y, o)

    auc = metrics.auc(fpr, tpr)



    plt.plot(fpr, tpr, label='ROC curve (area = %.3f)'%auc)

    plt.legend()

    plt.title('ROC curve')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.grid(True)
y=train['target']

X=train.drop('target',axis=1)

T=test
%%time

y,X,T=update_dataset(y,X,T,iter_n=1)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
%%time

y, X, T, o, p = fit_set(y,X,T,k=15, reg=.5)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
show_preds(p)
compare_oof_target(o)
%%time

y,X,T=update_dataset(y,X,T,o,p,iter_n=2)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
%%time

y, X, T, o, p = fit_set(y,X,T,k=13,reg=.3)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
show_preds(p)
compare_oof_target(o)
%%time

y,X,T=update_dataset(y,X,T,o,p,iter_n=3)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
%%time

y, X, T, o, p = fit_set(y,X,T,k=11,reg=.1)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
show_preds(p)
compare_oof_target(o)
%%time

y,X,T=update_dataset(y,X,T,o,p,iter_n=4)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
%%time

y, X, T, o, p = fit_set(y,X,T,k=11,reg=.03)

display(f'y:{str(y.shape)}    X:{str(X.shape)}    T:{T.shape}')
show_preds(p)
compare_oof_target(o)
preds=p.quantile(.5,axis=1).dropna()

preds.name='target'

preds.hist(bins=3)

preds.shape
sub = pd.read_csv(FILES['sample'],index_col='id')

sub.update(preds)

sub.to_csv('submission.csv',index=True)