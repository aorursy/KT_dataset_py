# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
np.random.seed(42)
# !pip install catboost featuretools hyperopt colorama
df = pd.read_csv("../input/train.csv", index_col="ID_code")
df.info()
df.sample(5)
df.target.value_counts()
from sklearn.model_selection import train_test_split
X = df.drop("target", axis="columns")

y = df['target']
from scipy.stats import norm, rankdata

def add_features(df):  

    columns = df.columns

    

    df['sum']  = df[columns].sum(axis=1)  

    df['min']  = df[columns].min(axis=1)

    df['max']  = df[columns].max(axis=1)

    df['mean'] = df[columns].mean(axis=1)

    df['std']  = df[columns].std(axis=1)

    df['skew'] = df[columns].skew(axis=1)

    df['kurt'] = df[columns].kurtosis(axis=1)

    df['med']  = df[columns].median(axis=1)

    df['pos']  = df[columns].aggregate(lambda x: np.sum(x >= 0), axis=1)

    df['neg']  = df[columns].aggregate(lambda x: np.sum(x < 0), axis=1)

    

    # Add more features

    for col in df.columns:

        # Normalize the data, so that it can be used in norm.cdf(), as though it is a standard normal variable

        df[col] = ((df[col] - df[col].mean()) / df[col].std()).astype('float32')



        # Square root

        df[col+'_r1'] = np.round(df[col], 1)

        

        # Square root

        df[col+'_r2'] = np.round(df[col], 2)



        # Square

        df[col+'_s'] = np.power(df[col], 2)



        # Cube

        df[col+'_c'] = np.power(df[col], 3)



        # 4th power

        df[col+'_q'] = np.power(df[col], 4)

        

        # Normalize the data, so that it can be used in norm.cdf(), as though it is a standard normal variable

        # df[col] = ((df[col] - df[col].mean()) / df[col].std()).astype('float32')



        # Cumulative percentile (not normalized)

        # df[col+'_r'] = rankdata(df[col]).astype('float32')



        # Cumulative normal percentile

        # df[col+'_n'] = norm.cdf(df[col]).astype('float32')



    return df
X = add_features(X)
X.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

X_train.head()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, QuantileTransformer

from sklearn.linear_model import LogisticRegression
preprocessor = Pipeline([

    ('scaler', MaxAbsScaler())

])
import sklearn

import catboost

import hyperopt

from hyperopt import hp, fmin, tpe, STATUS_OK, Trials

import colorama



N_HYPEROPT_PROBES = 60

HYPEROPT_ALGO = tpe.suggest

colorama.init()
#preprocessor.fit(X_train)
#X_train = preprocessor.transform(X_train)

#X_test = preprocessor.transform(X_test)
D_train = catboost.Pool(X_train, y_train)

D_test = catboost.Pool(X_test, y_test)
def get_catboost_params(space):

    params = dict()

    params['learning_rate'] = space['learning_rate']

    params['depth'] = int(space['depth'])

    params['l2_leaf_reg'] = space['l2_leaf_reg']

    params['border_count'] = space['border_count']

    #params['rsm'] = space['rsm']

    return params
obj_call_count = 0

cur_best_loss = np.inf

log_writer = open( 'catboost-hyperopt-log.txt', 'w' )
def objective(space):

    global obj_call_count, cur_best_loss



    obj_call_count += 1



    print('\nCatBoost objective call #{} cur_best_loss={:7.5f}'.format(obj_call_count,cur_best_loss) )



    params = get_catboost_params(space)



    sorted_params = sorted(space.items(), key=lambda z: z[0])

    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])

    print('Params: {}'.format(params_str) )



    model = catboost.CatBoostClassifier(iterations=100000,

                                        learning_rate=params['learning_rate'],

                                        depth=int(params['depth']),

                                        loss_function='Logloss',

                                        use_best_model=True,

                                        task_type="GPU",

                                        eval_metric='AUC',

                                        l2_leaf_reg=params['l2_leaf_reg'],

                                        early_stopping_rounds=3000,

                                        od_type="Iter",

                                        border_count=int(params['border_count']),

                                        verbose=False

                                        )

    

    model.fit(D_train, eval_set=D_test, verbose=False)

    nb_trees = model.tree_count_



    print('nb_trees={}'.format(nb_trees))



    y_pred = model.predict_proba(D_test.get_features())

    test_loss = sklearn.metrics.log_loss(D_test.get_label(), y_pred, labels=[0, 1])

    acc = sklearn.metrics.accuracy_score(D_test.get_label(), np.argmax(y_pred, axis=1))

    auc = sklearn.metrics.roc_auc_score(D_test.get_label(), y_pred[:,1])



    log_writer.write('loss={:<7.5f} acc={} auc={} Params:{} nb_trees={}\n'.format(test_loss, acc, auc, params_str, nb_trees ))

    log_writer.flush()



    if test_loss<cur_best_loss:

        cur_best_loss = test_loss

        print(colorama.Fore.GREEN + 'NEW BEST LOSS={}'.format(cur_best_loss) + colorama.Fore.RESET)





    return{'loss':test_loss, 'status': STATUS_OK }
space = {

        'depth': hp.quniform("depth", 1, 6, 1),

        'border_count': hp.uniform ('border_count', 32, 255),

        'learning_rate': hp.loguniform('learning_rate', -5.0, -2),

        'l2_leaf_reg': hp.uniform('l2_leaf_reg', 3, 8),

       }



trials = Trials()

best = hyperopt.fmin(fn=objective,

                     space=space,

                     algo=HYPEROPT_ALGO,

                     max_evals=N_HYPEROPT_PROBES,

                     trials=trials,

                     verbose=True)



print('-'*50)

print('The best params:')

print( best )

print('\n\n')
print(best)
best.update({'border_count': int(best['border_count'])})
model = catboost.CatBoostClassifier(iterations=20000,

                                    loss_function='Logloss',

                                    use_best_model=True,

                                    task_type="GPU",

                                    eval_metric='AUC',

                                    early_stopping_rounds=500,

                                    od_type="Iter",

                                    verbose=2000,

                                    **best

                                    )



model.fit(D_train, eval_set=D_test, verbose=2000)
pred = model.predict_proba(D_test.get_features())

print("auc = ", sklearn.metrics.roc_auc_score(D_test.get_label(), pred[:,1]))

print("acc = ", sklearn.metrics.accuracy_score(D_test.get_label(), np.argmax(pred, axis=1)))

print("loss = ", sklearn.metrics.log_loss(D_test.get_label(), pred, labels=[0, 1]))
df_test = pd.read_csv("../input/test.csv", index_col="ID_code")

df_test.info()
df_test = add_features(df_test)

arr_test = df_test.values
D_train = catboost.Pool(arr_test)

test_preds = model.predict_proba(arr_test)[:,1]
test_preds[:10]
print("Saving submission file")

sample = pd.read_csv('../input/sample_submission.csv')

sample.target = test_preds.astype(float)

sample.ID_code = df_test.index

sample.to_csv('submission.csv', index=False)