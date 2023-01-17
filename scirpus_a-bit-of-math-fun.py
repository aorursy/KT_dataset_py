import numpy as np 

import pandas as pd

from sklearn.model_selection import KFold

from numba import jit
@jit

def eval_gini(y_true, y_prob):

    y_true = np.asarray(y_true)

    y_true = y_true[np.argsort(y_prob)]

    ntrue = 0

    gini = 0

    delta = 0

    n = len(y_true)

    for i in range(n-1, -1, -1):

        y_i = y_true[i]

        ntrue += y_i

        gini += y_i * delta

        delta += 1 - y_i

    gini = 1 - 2 * gini / (ntrue * (n - ntrue))

    return gini



def ProjectOnMean(data1, data2, columnName):

    grpOutcomes = data1.groupby(list([columnName]))['target'].mean().reset_index()

    x = pd.merge(data2[[columnName, 'target']], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=list([columnName]),

                 left_index=True)['target']

    return x.fillna(x.mean()).values



def GetData(strdirectory, folds, myseeds):

    # Project Categorical inputs to Target

    train = pd.read_csv(strdirectory+'train.csv')

    test = pd.read_csv(strdirectory+'test.csv')

 

    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

    train.drop(unwanted,inplace=True,axis=1)

    test.drop(unwanted,inplace=True,axis=1)



    test['target'] = np.nan

    feats = list(set(train.columns).difference(set(['id','target'])))

    feats = list(['id'])+feats +list(['target'])

    train = train[feats]

    test = test[feats]

    

    highcardinality = ['ps_car_02_cat',

                       'ps_car_09_cat',

                       'ps_ind_04_cat',

                       'ps_ind_05_cat',

                       'ps_car_03_cat',

                       'ps_ind_08_bin',

                       'ps_car_05_cat',

                       'ps_car_08_cat',

                       'ps_ind_06_bin',

                       'ps_ind_07_bin',

                       'ps_ind_12_bin',

                       'ps_ind_18_bin',

                       'ps_ind_17_bin',

                       'ps_car_07_cat',

                       'ps_car_11_cat',

                       'ps_ind_09_bin',

                       'ps_car_10_cat',

                       'ps_car_04_cat',

                       'ps_car_01_cat',

                       'ps_ind_02_cat',

                       'ps_ind_10_bin',

                       'ps_ind_11_bin',

                       'ps_car_06_cat',

                       'ps_ind_13_bin',

                       'ps_ind_16_bin']

    

    pomcolumns = []

    for c in highcardinality:

        train['pom_'+c] = 0

        test['pom_'+c] = 0

        pomcolumns.append('pom_'+c)    

    

    for sd in myseeds:

        print('Seed',sd)

        kf = KFold(n_splits=folds,shuffle=True,random_state=sd)

        for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]))):

            print('Fold:',i)

            blindtrain = train.iloc[test_index].copy() 

            vistrain = train.iloc[train_index].copy()



            for c in highcardinality:

                train.loc[test_index,'pom_'+c] += ProjectOnMean(vistrain,

                                                                blindtrain,c)

                test.loc[:,'pom_'+c] += ProjectOnMean(vistrain,

                                                      test,c)

   

    for c in highcardinality:

        train['pom_'+c] /= len(myseeds) 

        test['pom_'+c] /= len(myseeds) 

    

    

    features = list(set(train.columns).difference(set(['id','target'])))

    features = list(['id'])+features+list(['target'])

    train = train[features]

    test = test[features]



    return pomcolumns, highcardinality, train, test
folds = 5

np.random.seed(42)

myseeds = np.random.randint(0,99999,size=20)
strdirectory = '../input/'

pomcolumns, highcardinality, train, test = GetData(strdirectory,folds,myseeds)
train.head()
test.head()
eval_gini(train.target,train[pomcolumns].mean(axis=1))
x = train[list(pomcolumns)+list(['target'])].corr().target[:-1]
correlatedoutput = np.dot(train[pomcolumns].values,x)
eval_gini(train.target,correlatedoutput)
sub = pd.read_csv(strdirectory+'sample_submission.csv')

sub.target = np.dot(test[pomcolumns].values,x)

sub.to_csv('correlatedpom.csv',index=False)