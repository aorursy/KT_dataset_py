import numpy as np

import pandas as pd

from numba import jit

import xgboost as xgb

from sklearn.metrics import log_loss

from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.preprocessing import StandardScaler
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



def gini_xgb(preds, dtrain):

    labels = dtrain.get_label()

    preds -= preds.min()

    preds / preds.max()

    gini_score = -eval_gini(labels, preds)

    return [('gini', gini_score)]
def ProjectOnMean(data1, data2, columnName):

    grpOutcomes = data1.groupby(list([columnName]))['target'].mean().reset_index()

    grpCount = data1.groupby(list([columnName]))['target'].count().reset_index()

    grpOutcomes['cnt'] = grpCount.target

    grpOutcomes.drop('cnt', inplace=True, axis=1)

    outcomes = data2['target'].values

    x = pd.merge(data2[[columnName, 'target']], grpOutcomes,

                 suffixes=('x_', ''),

                 how='left',

                 on=list([columnName]),

                 left_index=True)['target']



    

    return x.fillna(x.mean()).values



def GetData(strdirectory):

    # Project Categorical inputs to Target

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



    train = pd.read_csv(strdirectory+'train.csv')

    test = pd.read_csv(strdirectory+'test.csv')



    train['missing'] = (train==-1).sum(axis=1).astype(float)

    test['missing'] = (test==-1).sum(axis=1).astype(float)



    unwanted = train.columns[train.columns.str.startswith('ps_calc_')]

    train.drop(unwanted,inplace=True,axis=1)

    test.drop(unwanted,inplace=True,axis=1)



    test['target'] = np.nan

    feats = list(set(train.columns).difference(set(['id','target'])))

    feats = list(['id'])+feats +list(['target'])

    train = train[feats]

    test = test[feats]

    

    blindloodata = None

    folds = 5

    kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=2017)

    for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]),train.target)):

        print('Fold:',i)

        blindtrain = train.iloc[test_index].copy() 

        vistrain = train.iloc[train_index].copy()



        for c in highcardinality:

            blindtrain.insert(1,'loo_'+c, ProjectOnMean(vistrain,

                                                       blindtrain,c))

        if(blindloodata is None):

            blindloodata = blindtrain.copy()

        else:

            blindloodata = pd.concat([blindloodata,blindtrain])



    for c in highcardinality:

        test.insert(1,'loo_'+c, ProjectOnMean(train,

                                              test,c))

    #test.drop(highcardinality,inplace=True,axis=1)

    # test.drop(['loo_ps_car_11_cat','loo_ps_car_09_cat'],inplace=True,axis=1)

    train = blindloodata

    #train.drop(highcardinality,inplace=True,axis=1)

    #train.drop(['loo_ps_car_11_cat','loo_ps_car_09_cat'],inplace=True,axis=1)

    

    print('Scale values')

    ss = StandardScaler()

    features = train.columns[1:-1]

    ss.fit(pd.concat([train[features],test[features]]))

    train[features] = ss.transform(train[features] )

    test[features] = ss.transform(test[features] )

    train[features] = np.round(train[features], 6)

    test[features] = np.round(test[features], 6)

    return highcardinality, train, test
strdirectory = '../input/'

highcardinality, train, test = GetData(strdirectory)
def GP1(data):

    v = pd.DataFrame()

    v["i1"] = 0.020000*np.tanh((((data["ps_car_12"] + ((data["ps_car_15"] + data["ps_reg_01"]) + data["loo_ps_ind_16_bin"])) * 2.0) * 2.0))

    v["i2"] = 0.019910*np.tanh((((data["loo_ps_car_04_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]))) * 2.0) * 2.0))

    v["i3"] = 0.020000*np.tanh((((data["loo_ps_car_05_cat"] + data["loo_ps_ind_16_bin"]) + (data["loo_ps_car_09_cat"] + (data["loo_ps_car_11_cat"] * 2.0))) * 2.0))

    v["i4"] = 0.020000*np.tanh((data["ps_car_15"] + (((data["ps_reg_02"] + data["loo_ps_ind_06_bin"]) + data["ps_reg_03"]) + data["loo_ps_car_11_cat"])))

    v["i5"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]) + data["loo_ps_car_02_cat"]) + data["loo_ps_car_04_cat"]) * 2.0))

    v["i6"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_05_cat"]) + (data["ps_car_13"] + (data["ps_car_13"] + data["ps_reg_02"]))))

    v["i7"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((data["loo_ps_car_03_cat"] + data["ps_car_13"]) * 2.0)) * 2.0))

    v["i8"] = 0.019945*np.tanh(((data["loo_ps_car_03_cat"] + data["loo_ps_car_04_cat"]) + ((data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]) + data["loo_ps_car_01_cat"])))

    v["i9"] = 0.020000*np.tanh((((data["ps_car_13"] + data["loo_ps_car_07_cat"]) * 2.0) + (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_16_bin"])))

    v["i10"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_06_bin"] + data["loo_ps_car_07_cat"]))) * 2.0) * 2.0))

    v["i11"] = 0.020000*np.tanh(((data["loo_ps_car_11_cat"] + ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_06_bin"]) + data["loo_ps_car_07_cat"])) * 2.0))

    v["i12"] = 0.020000*np.tanh((((((data["ps_car_13"] * 2.0) + data["loo_ps_car_03_cat"])/2.0) + (data["ps_reg_01"] + data["loo_ps_ind_05_cat"])) * 2.0))

    v["i13"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((data["ps_car_13"] + (data["loo_ps_car_01_cat"] + data["loo_ps_car_07_cat"])) * 2.0)) * 2.0))

    v["i14"] = 0.020000*np.tanh((((((data["loo_ps_car_02_cat"] + data["loo_ps_ind_07_bin"]) + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_07_bin"]) * 2.0) * 2.0))

    v["i15"] = 0.020000*np.tanh((data["ps_car_13"] + (data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_06_bin"] + data["loo_ps_car_01_cat"]) + data["loo_ps_car_09_cat"]))))

    v["i16"] = 0.020000*np.tanh(((((data["loo_ps_ind_06_bin"] + data["ps_reg_02"]) + data["ps_car_13"]) + data["loo_ps_car_01_cat"]) + data["ps_car_13"]))

    v["i17"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_17_bin"]) + (data["ps_car_12"] + data["ps_reg_01"])) - data["ps_ind_15"]))

    v["i18"] = 0.020000*np.tanh((((data["ps_car_15"] + ((data["ps_reg_03"] + data["loo_ps_car_09_cat"]) * 2.0)) + data["ps_reg_02"]) * 2.0))

    v["i19"] = 0.020000*np.tanh((data["ps_car_13"] + ((data["ps_reg_03"] + data["loo_ps_car_07_cat"]) - (data["ps_ind_15"] - data["loo_ps_car_03_cat"]))))

    v["i20"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_car_07_cat"] + (data["loo_ps_car_03_cat"] + data["loo_ps_car_09_cat"]))) * 2.0) * 2.0))

    v["i21"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_06_bin"] - data["ps_ind_15"])) + data["ps_reg_02"]) * 2.0))

    v["i22"] = 0.020000*np.tanh(((((((data["loo_ps_ind_17_bin"] + data["ps_reg_03"]) * 2.0) * 2.0) * 2.0) + data["loo_ps_car_11_cat"]) * 2.0))

    v["i23"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] + data["loo_ps_ind_07_bin"]) + (data["loo_ps_ind_05_cat"] + (data["ps_reg_03"] + data["loo_ps_ind_09_bin"]))))

    v["i24"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) + data["ps_ind_03"]) + data["loo_ps_car_07_cat"]) + data["loo_ps_ind_05_cat"]))

    v["i25"] = 0.020000*np.tanh(((data["ps_reg_02"] - ((data["loo_ps_ind_02_cat"] * 2.0) * data["ps_ind_03"])) + data["ps_car_13"]))

    v["i26"] = 0.020000*np.tanh((((data["ps_car_13"] + data["loo_ps_car_09_cat"]) + (data["loo_ps_car_07_cat"] + data["ps_ind_03"])) - data["ps_ind_15"]))

    v["i27"] = 0.020000*np.tanh(((data["loo_ps_car_07_cat"] + data["loo_ps_ind_06_bin"]) + (data["loo_ps_car_01_cat"] + (data["loo_ps_ind_16_bin"] - data["ps_ind_15"]))))

    v["i28"] = 0.020000*np.tanh(((((data["ps_reg_03"] + (data["ps_reg_03"] * 2.0)) + data["loo_ps_ind_04_cat"]) + data["ps_reg_02"]) * 2.0))

    v["i29"] = 0.020000*np.tanh(((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_17_bin"] + (data["ps_car_13"] + data["ps_ind_01"]))) + data["loo_ps_car_09_cat"]))

    v["i30"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] + (((data["loo_ps_ind_02_cat"] - data["ps_ind_15"]) + data["loo_ps_ind_09_bin"]) + data["ps_reg_01"])))

    v["i31"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + ((data["loo_ps_car_01_cat"] + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_02_cat"])) + data["loo_ps_ind_07_bin"]))

    v["i32"] = 0.020000*np.tanh((((((data["ps_ind_03"] + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i33"] = 0.020000*np.tanh((((data["loo_ps_car_01_cat"] + (data["loo_ps_car_01_cat"] - data["ps_ind_15"])) - data["ps_ind_15"]) + data["loo_ps_car_07_cat"]))

    v["i34"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + (data["loo_ps_ind_16_bin"] - data["loo_ps_ind_18_bin"])) * 2.0) * 2.0))

    v["i35"] = 0.020000*np.tanh((((((data["loo_ps_ind_05_cat"] * 2.0) + (data["loo_ps_ind_07_bin"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0))

    v["i36"] = 0.020000*np.tanh(((((data["ps_ind_03"] * data["ps_ind_03"]) + data["ps_ind_03"]) + (data["loo_ps_ind_05_cat"] * 2.0)) * 2.0))

    v["i37"] = 0.020000*np.tanh((((((data["loo_ps_ind_17_bin"] * 2.0) + (data["loo_ps_ind_05_cat"] + data["loo_ps_car_11_cat"])) * 2.0) * 2.0) * 2.0))

    v["i38"] = 0.020000*np.tanh(((data["ps_car_13"] + (data["loo_ps_car_08_cat"] + (data["ps_ind_03"] * data["ps_ind_03"]))) + data["loo_ps_car_09_cat"]))

    v["i39"] = 0.019996*np.tanh((((((data["ps_car_15"] + data["loo_ps_car_07_cat"]) + (data["ps_reg_03"] * 2.0)) * 2.0) * 2.0) * 2.0))

    v["i40"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_02_cat"]) * 2.0) * 2.0) + data["ps_car_15"]))

    v["i41"] = 0.020000*np.tanh((1.0 + ((data["loo_ps_ind_04_cat"] - data["ps_ind_15"]) + (data["loo_ps_car_03_cat"] - data["ps_ind_15"]))))

    v["i42"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_05_cat"] * 2.0)) + (data["ps_ind_03"] * data["ps_ind_03"])) * 2.0))

    v["i43"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] + (((data["loo_ps_ind_06_bin"] + data["loo_ps_ind_09_bin"]) * data["ps_ind_03"]) * 2.0)) * 2.0))

    v["i44"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] + data["loo_ps_ind_09_bin"]) + (data["ps_ind_01"] + data["loo_ps_ind_09_bin"])) * 2.0) * 2.0))

    v["i45"] = 0.020000*np.tanh((((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) + ((data["loo_ps_car_07_cat"] + data["loo_ps_ind_05_cat"]) * 2.0)) * 2.0))

    v["i46"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_04_cat"] + data["ps_ind_01"])) + data["loo_ps_car_09_cat"]) + data["ps_ind_03"]))

    v["i47"] = 0.020000*np.tanh((data["ps_ind_01"] + ((data["loo_ps_car_05_cat"] * (data["ps_ind_01"] * 2.0)) - data["ps_ind_15"])))

    v["i48"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_16_bin"] * data["ps_car_12"]) + data["loo_ps_ind_05_cat"])))

    v["i49"] = 0.020000*np.tanh(((8.70196723937988281) * ((8.70196723937988281) * (data["ps_ind_03"] * ((0.0) - data["loo_ps_ind_02_cat"])))))

    v["i50"] = 0.020000*np.tanh((data["ps_car_15"] + ((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_17_bin"]) - (data["loo_ps_car_04_cat"] * data["ps_car_11"]))))

    v["i51"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] - data["ps_reg_01"]) * ((7.16651058197021484) * data["loo_ps_car_01_cat"])))

    v["i52"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] - data["ps_car_11"]) - data["ps_ind_15"]) + (data["loo_ps_ind_05_cat"] - data["ps_ind_15"])))

    v["i53"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_06_bin"] * data["loo_ps_ind_05_cat"])))

    v["i54"] = 0.020000*np.tanh(((data["loo_ps_car_06_cat"] + (data["loo_ps_ind_17_bin"] + (data["ps_ind_01"] + data["missing"]))) * data["loo_ps_car_05_cat"]))

    v["i55"] = 0.020000*np.tanh(((data["ps_reg_03"] + ((data["ps_reg_03"] + data["loo_ps_ind_04_cat"]) * 2.0)) * 2.0))

    v["i56"] = 0.020000*np.tanh(((data["loo_ps_car_01_cat"] + (data["loo_ps_car_01_cat"] - data["loo_ps_car_02_cat"])) * 2.0))

    v["i57"] = 0.020000*np.tanh((-((data["ps_ind_03"] * ((((data["loo_ps_ind_02_cat"] * 2.0) * 2.0) * 2.0) * 2.0)))))

    v["i58"] = 0.019996*np.tanh((data["loo_ps_car_07_cat"] + (data["ps_ind_03"] * (data["ps_ind_03"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_11_cat"])))))

    v["i59"] = 0.020000*np.tanh((((-2.0 + data["loo_ps_ind_05_cat"]) - data["ps_car_11"]) * 2.0))

    v["i60"] = 0.020000*np.tanh((data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"]))

    v["i61"] = 0.020000*np.tanh((((((data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_08_bin"] * data["ps_ind_03"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i62"] = 0.020000*np.tanh((data["ps_car_15"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_03_cat"])))

    v["i63"] = 0.020000*np.tanh(((data["loo_ps_ind_06_bin"] * ((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"]) - data["loo_ps_car_11_cat"])) * 2.0))

    v["i64"] = 0.019992*np.tanh((data["loo_ps_car_05_cat"] * (data["ps_car_12"] + (data["loo_ps_ind_17_bin"] + data["loo_ps_car_08_cat"]))))

    v["i65"] = 0.020000*np.tanh(((((data["missing"] * 2.0) * data["loo_ps_ind_02_cat"]) * 2.0) + data["loo_ps_ind_09_bin"]))

    v["i66"] = 0.020000*np.tanh((((((data["missing"] / 2.0) + (data["loo_ps_car_08_cat"] * 2.0)) * 2.0) * 2.0) * data["loo_ps_ind_02_cat"]))

    v["i67"] = 0.020000*np.tanh(((-(((data["ps_reg_02"] * data["loo_ps_car_06_cat"]) + (data["ps_reg_03"] * data["ps_ind_01"])))) * 2.0))

    v["i68"] = 0.020000*np.tanh(((((data["ps_ind_01"] + data["loo_ps_car_09_cat"]) + ((data["loo_ps_car_11_cat"] + data["loo_ps_car_09_cat"])/2.0))/2.0) * data["loo_ps_ind_09_bin"]))

    v["i69"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * (((data["loo_ps_ind_06_bin"] - data["ps_reg_01"]) - data["ps_car_15"]) + data["loo_ps_ind_05_cat"])))

    v["i70"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] + ((((data["ps_ind_15"] * data["loo_ps_ind_18_bin"]) * 2.0) * 2.0) * 2.0)))

    v["i71"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] - (data["ps_car_11"] * data["ps_ind_01"])) - data["ps_car_11"]))

    v["i72"] = 0.019984*np.tanh((((-2.0 - ((data["ps_ind_03"] * 2.0) * 2.0)) * data["loo_ps_ind_02_cat"]) * 2.0))

    v["i73"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["ps_reg_03"]) + (data["loo_ps_ind_16_bin"] * data["loo_ps_car_04_cat"])) * 2.0))

    v["i74"] = 0.020000*np.tanh(((data["loo_ps_ind_05_cat"] - data["ps_car_13"]) * ((data["ps_reg_02"] * 2.0) * 2.0)))

    v["i75"] = 0.019992*np.tanh((((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_07_bin"]) + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"])) - data["loo_ps_car_04_cat"]))

    v["i76"] = 0.020000*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_car_03_cat"]) * data["loo_ps_ind_09_bin"]) + (data["loo_ps_car_05_cat"] * data["ps_ind_03"])))

    v["i77"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + data["ps_ind_01"]) - (data["ps_ind_01"] * data["ps_ind_15"])))

    v["i78"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * ((data["ps_ind_01"] - data["ps_reg_01"]) + (data["loo_ps_ind_07_bin"] - data["ps_reg_01"]))))

    v["i79"] = 0.020000*np.tanh((((data["loo_ps_ind_07_bin"] * data["ps_ind_15"]) + data["loo_ps_ind_07_bin"]) * (data["ps_ind_15"] * 2.0)))

    v["i80"] = 0.019988*np.tanh((((data["loo_ps_car_11_cat"] + data["ps_car_15"])/2.0) * data["missing"]))

    v["i81"] = 0.020000*np.tanh((((data["ps_ind_03"] * (data["loo_ps_ind_04_cat"] + (data["loo_ps_ind_04_cat"] * 2.0))) + data["loo_ps_ind_04_cat"]) * 2.0))

    v["i82"] = 0.020000*np.tanh((data["ps_car_12"] * (((-(data["ps_car_11"])) * 2.0) * 2.0)))

    v["i83"] = 0.020000*np.tanh((data["loo_ps_car_01_cat"] * (((data["loo_ps_car_01_cat"] - data["ps_reg_03"]) - data["ps_reg_03"]) - data["ps_ind_14"])))

    v["i84"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] - data["ps_ind_15"]) * data["loo_ps_car_04_cat"]) + (data["ps_ind_15"] * data["loo_ps_ind_07_bin"])))

    v["i85"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] + (data["loo_ps_ind_02_cat"] + (data["ps_ind_03"] * (data["loo_ps_car_11_cat"] - data["ps_reg_01"])))))

    v["i86"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_ind_05_cat"])/2.0) - (data["ps_reg_01"] * (data["ps_ind_03"] + data["loo_ps_ind_08_bin"]))))

    v["i87"] = 0.020000*np.tanh(((((data["loo_ps_car_01_cat"] * data["loo_ps_ind_16_bin"]) + data["loo_ps_ind_16_bin"])/2.0) + (data["loo_ps_car_04_cat"] * data["ps_ind_01"])))

    v["i88"] = 0.020000*np.tanh(((data["loo_ps_ind_02_cat"] + data["ps_ind_03"]) * (data["ps_car_12"] - ((data["loo_ps_ind_02_cat"] * 2.0) * 2.0))))

    v["i89"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * (data["ps_reg_02"] + ((data["missing"] + data["loo_ps_car_06_cat"]) + data["loo_ps_car_09_cat"]))))

    v["i90"] = 0.020000*np.tanh((data["loo_ps_ind_06_bin"] * ((data["loo_ps_car_09_cat"] * ((data["loo_ps_ind_06_bin"] + data["loo_ps_car_09_cat"])/2.0)) - data["ps_car_11"])))

    v["i91"] = 0.019984*np.tanh((((data["loo_ps_car_04_cat"] + data["ps_ind_01"]) + data["loo_ps_car_04_cat"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_03"])))

    v["i92"] = 0.020000*np.tanh((data["ps_reg_03"] * (data["loo_ps_ind_05_cat"] - data["loo_ps_car_01_cat"])))

    v["i93"] = 0.019988*np.tanh(((((data["ps_car_13"] + data["loo_ps_car_03_cat"])/2.0) + ((data["ps_car_15"] + data["loo_ps_ind_08_bin"])/2.0))/2.0))

    v["i94"] = 0.020000*np.tanh((((data["loo_ps_ind_05_cat"] * data["ps_reg_03"]) - data["ps_reg_01"]) * (data["loo_ps_ind_18_bin"] + data["ps_reg_03"])))

    v["i95"] = 0.020000*np.tanh(((data["loo_ps_ind_04_cat"] + (data["ps_ind_15"] * (data["loo_ps_ind_18_bin"] - data["loo_ps_car_06_cat"]))) + data["ps_reg_01"]))

    v["i96"] = 0.020000*np.tanh(((data["ps_ind_03"] * ((data["loo_ps_car_05_cat"] + (data["loo_ps_ind_05_cat"] + (data["loo_ps_ind_11_bin"] / 2.0)))/2.0)) / 2.0))

    v["i97"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * (data["loo_ps_car_09_cat"] / 2.0)))

    v["i98"] = 0.020000*np.tanh((data["loo_ps_car_03_cat"] * ((data["loo_ps_car_09_cat"] + data["ps_ind_01"]) * ((data["loo_ps_car_04_cat"] + data["ps_ind_01"])/2.0))))

    v["i99"] = 0.020000*np.tanh((data["loo_ps_ind_17_bin"] * (((data["ps_reg_03"] + data["loo_ps_ind_05_cat"]) + data["loo_ps_ind_05_cat"]) * 2.0)))

    v["i100"] = 0.019988*np.tanh(((data["ps_car_11"] * (np.tanh(data["loo_ps_ind_04_cat"]) * 2.0)) + data["loo_ps_ind_04_cat"]))

    v["i101"] = 0.020000*np.tanh(((((data["loo_ps_ind_05_cat"] * data["ps_ind_01"]) - data["loo_ps_ind_04_cat"]) + data["loo_ps_ind_18_bin"]) * data["ps_ind_01"]))

    v["i102"] = 0.020000*np.tanh(((((-1.0 - data["ps_ind_03"]) * 2.0) * np.tanh((data["loo_ps_ind_02_cat"] * 2.0))) * 2.0))

    v["i103"] = 0.020000*np.tanh((((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_07_bin"])/2.0) * data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_17_bin"] * data["loo_ps_car_09_cat"]))/2.0))

    v["i104"] = 0.020000*np.tanh(((((-2.0 + (data["ps_ind_03"] * data["ps_ind_03"])) - data["ps_ind_03"]) * 2.0) * 2.0))

    v["i105"] = 0.020000*np.tanh((((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] / 2.0)) * data["loo_ps_car_07_cat"]) - data["loo_ps_car_07_cat"]))

    v["i106"] = 0.020000*np.tanh(((((data["loo_ps_ind_17_bin"] + ((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_05_cat"] * data["loo_ps_ind_17_bin"]))/2.0))/2.0) + data["loo_ps_ind_12_bin"])/2.0))

    v["i107"] = 0.020000*np.tanh((((data["loo_ps_car_11_cat"] * data["loo_ps_car_03_cat"]) - data["loo_ps_car_03_cat"]) - (data["ps_car_15"] * data["loo_ps_car_03_cat"])))

    v["i108"] = 0.019996*np.tanh((((data["loo_ps_car_02_cat"] * data["ps_ind_03"]) + ((data["loo_ps_car_02_cat"] + (data["loo_ps_car_11_cat"] / 2.0))/2.0))/2.0))

    v["i109"] = 0.020000*np.tanh((((data["ps_reg_03"] + data["loo_ps_car_09_cat"]) + data["ps_ind_01"]) * (data["loo_ps_ind_05_cat"] - data["ps_reg_01"])))

    v["i110"] = 0.020000*np.tanh((np.tanh(((data["loo_ps_car_05_cat"] + data["ps_ind_01"])/2.0)) - (data["ps_car_11"] * data["ps_ind_01"])))

    v["i111"] = 0.020000*np.tanh((np.tanh(data["loo_ps_ind_04_cat"]) * (((-((data["loo_ps_car_03_cat"] / 2.0))) + data["loo_ps_ind_04_cat"])/2.0)))

    v["i112"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_02_cat"] + data["loo_ps_ind_08_bin"]) + data["loo_ps_ind_02_cat"])) * 2.0))

    v["i113"] = 0.020000*np.tanh((data["loo_ps_ind_16_bin"] * data["loo_ps_car_07_cat"]))

    v["i114"] = 0.020000*np.tanh(((data["loo_ps_ind_16_bin"] + ((((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_16_bin"])/2.0) + data["loo_ps_ind_05_cat"])/2.0)) * data["loo_ps_ind_09_bin"]))

    v["i115"] = 0.020000*np.tanh((((data["loo_ps_ind_04_cat"] / 2.0) * data["loo_ps_ind_04_cat"]) - data["loo_ps_ind_05_cat"]))

    v["i116"] = 0.019871*np.tanh(np.tanh((data["loo_ps_car_05_cat"] * (data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_02_cat"] * 2.0)))))

    v["i117"] = 0.019996*np.tanh((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_05_cat"]) * (data["loo_ps_ind_07_bin"] + data["loo_ps_car_04_cat"])) - data["loo_ps_ind_17_bin"]))

    v["i118"] = 0.019969*np.tanh((((data["loo_ps_ind_08_bin"] + ((data["ps_car_13"] + (data["loo_ps_ind_12_bin"] / 2.0))/2.0)) + data["loo_ps_ind_08_bin"])/2.0))

    v["i119"] = 0.020000*np.tanh(((data["loo_ps_car_09_cat"] - data["ps_car_15"]) * (data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] - data["ps_car_15"]))))

    v["i120"] = 0.020000*np.tanh((((data["loo_ps_ind_02_cat"] * (data["loo_ps_car_09_cat"] + data["ps_car_13"])) * 2.0) * 2.0))

    v["i121"] = 0.020000*np.tanh((data["loo_ps_ind_05_cat"] * ((data["ps_car_12"] + data["ps_car_12"]) * (data["loo_ps_car_07_cat"] * 2.0))))

    v["i122"] = 0.020000*np.tanh((((np.tanh(data["loo_ps_ind_07_bin"]) / 2.0) + np.tanh((data["loo_ps_ind_07_bin"] / 2.0)))/2.0))

    v["i123"] = 0.020000*np.tanh(((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_07_bin"] * (((data["loo_ps_car_08_cat"] + data["loo_ps_ind_17_bin"])/2.0) + data["loo_ps_ind_17_bin"])))/2.0))

    v["i124"] = 0.019906*np.tanh((((-(data["loo_ps_car_04_cat"])) * data["ps_car_11"]) - data["loo_ps_car_04_cat"]))

    v["i125"] = 0.019996*np.tanh(((((((data["loo_ps_car_06_cat"] * data["loo_ps_car_06_cat"]) + data["ps_car_11"])/2.0) * data["loo_ps_car_06_cat"]) + data["loo_ps_car_02_cat"])/2.0))

    v["i126"] = 0.020000*np.tanh((data["ps_reg_03"] * (data["ps_ind_15"] + ((data["loo_ps_ind_05_cat"] + data["ps_ind_15"]) + data["ps_ind_15"]))))

    v["i127"] = 0.020000*np.tanh(((((data["loo_ps_car_09_cat"] - data["ps_ind_01"]) - data["loo_ps_car_06_cat"]) - data["ps_ind_01"]) * data["loo_ps_ind_04_cat"]))

    v["i128"] = 0.020000*np.tanh(((((data["ps_reg_03"] + data["loo_ps_ind_05_cat"])/2.0) - data["ps_reg_01"]) * (data["ps_reg_03"] + data["ps_ind_03"])))

    v["i129"] = 0.020000*np.tanh(((data["loo_ps_car_03_cat"] * (-(data["loo_ps_ind_18_bin"]))) + (data["loo_ps_ind_09_bin"] * data["loo_ps_car_09_cat"])))

    v["i130"] = 0.020000*np.tanh(((((data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"]) - (data["loo_ps_ind_04_cat"] * 2.0)) * data["loo_ps_ind_02_cat"]) * 2.0))

    v["i131"] = 0.019941*np.tanh(((((data["ps_reg_03"] * 2.0) * (data["ps_reg_03"] * data["ps_ind_03"])) * 2.0) - data["ps_ind_03"]))

    v["i132"] = 0.020000*np.tanh(((((data["loo_ps_ind_07_bin"] - data["loo_ps_ind_07_bin"]) - data["loo_ps_ind_06_bin"]) - data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_04_cat"]))

    v["i133"] = 0.019996*np.tanh((data["ps_reg_01"] * ((((data["ps_reg_01"] + data["missing"])/2.0) - data["loo_ps_ind_05_cat"]) - data["loo_ps_car_09_cat"])))

    v["i134"] = 0.020000*np.tanh((((data["ps_ind_15"] / 2.0) - data["ps_ind_01"]) * ((data["ps_ind_03"] + data["ps_ind_15"])/2.0)))

    v["i135"] = 0.019996*np.tanh(((-(data["ps_car_13"])) - ((((data["ps_car_13"] * data["loo_ps_car_01_cat"]) * 2.0) * 2.0) * 2.0)))

    v["i136"] = 0.020000*np.tanh((((data["ps_reg_03"] + (data["ps_reg_03"] * data["loo_ps_ind_07_bin"])) + data["ps_reg_03"]) * data["loo_ps_ind_07_bin"]))

    v["i137"] = 0.019988*np.tanh((data["loo_ps_car_07_cat"] * (data["ps_reg_02"] / 2.0)))

    v["i138"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * ((data["missing"] + (data["loo_ps_car_09_cat"] * data["loo_ps_ind_02_cat"]))/2.0)))

    v["i139"] = 0.019996*np.tanh(((data["ps_reg_02"] + (data["loo_ps_car_06_cat"] / 2.0))/2.0))

    v["i140"] = 0.020000*np.tanh(((((data["loo_ps_ind_06_bin"] * data["ps_ind_15"]) - data["loo_ps_car_06_cat"]) * 2.0) * 2.0))

    v["i141"] = 0.020000*np.tanh((data["loo_ps_car_07_cat"] * ((data["loo_ps_ind_05_cat"] * data["ps_car_13"]) * data["loo_ps_car_07_cat"])))

    v["i142"] = 0.019980*np.tanh(((data["ps_ind_15"] * (-(data["ps_ind_03"]))) - np.tanh((data["ps_ind_15"] - data["ps_ind_03"]))))

    v["i143"] = 0.020000*np.tanh((data["loo_ps_ind_02_cat"] * (data["ps_car_15"] * data["loo_ps_ind_02_cat"])))

    v["i144"] = 0.019996*np.tanh((((np.tanh(data["loo_ps_ind_10_bin"]) / 2.0) - data["ps_reg_01"]) / 2.0))

    v["i145"] = 0.020000*np.tanh((((data["missing"] + (data["loo_ps_ind_02_cat"] * data["loo_ps_ind_02_cat"])) + data["loo_ps_car_08_cat"]) * data["loo_ps_ind_02_cat"]))

    v["i146"] = 0.019984*np.tanh((((data["ps_reg_02"] - data["loo_ps_car_02_cat"]) * (data["ps_reg_01"] * data["loo_ps_car_09_cat"])) / 2.0))

    v["i147"] = 0.020000*np.tanh((data["loo_ps_car_09_cat"] * ((data["loo_ps_car_09_cat"] * 2.0) + (data["loo_ps_car_09_cat"] * 2.0))))

    v["i148"] = 0.019977*np.tanh(((data["ps_reg_02"] * ((data["ps_car_13"] - data["ps_reg_03"]) * data["loo_ps_ind_06_bin"])) - data["loo_ps_ind_06_bin"]))

    v["i149"] = 0.019996*np.tanh((data["ps_car_11"] * (((data["loo_ps_car_06_cat"] + data["ps_car_11"])/2.0) - data["loo_ps_ind_16_bin"])))

    v["i150"] = 0.020000*np.tanh((((data["ps_ind_14"] / 2.0) + (data["ps_ind_15"] * data["ps_ind_14"]))/2.0))

    return v
#This is where you could add any original features or any other goodness

traintargets = train.target.values

train = GP1(train)

train['target'] = traintargets

test = GP1(test)
features = train.columns[:-1]
xgbtestpreds = np.zeros(test.shape[0])

scores = 0

myfeatures = list(features)

kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=2017)

for i, (train_index, test_index) in enumerate(kf.split(list(train.index),train.target)):

    print('Fold: ',i)

    blindtrain = train.iloc[test_index].copy()

    vistrain = train.iloc[train_index].copy()



    clf = xgb.XGBClassifier(n_estimators=2000,

                        learning_rate = 0.04,

                        max_depth = 5,

                        min_child_weight = 9,

                        subsample = 0.8,

                        colsample_bytree = 0.8,

                        reg_alpha = 10.4,

                        reg_lambda = 0.59,

                        seed = 2017,

                        nthread = 8,

                        silent = 1)



    eval_set=[(blindtrain[myfeatures],blindtrain.target)]

    model = clf.fit(vistrain[myfeatures], vistrain.target, 

                    eval_set=eval_set,

                    eval_metric=gini_xgb,

                    early_stopping_rounds=50,

                    verbose=False)



    print( "  Best N trees = ", model.best_ntree_limit )

    print( "  Best gini = ", model.best_score )

    trainpreds = model.predict_proba(blindtrain[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

    scores += 0.2*eval_gini(blindtrain.target,trainpreds)

    print( "  Best gini = ", eval_gini(blindtrain.target,trainpreds))

    xgbtestpreds += model.predict_proba(test[myfeatures],ntree_limit=model.best_ntree_limit)[:,1]

xgbtestpreds /= 5

print(scores)
sub = pd.read_csv(strdirectory+'sample_submission.csv')

sub.target = xgbtestpreds

sub.to_csv('gpxgbhybrid.csv',index=False)