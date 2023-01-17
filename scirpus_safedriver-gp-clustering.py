import numpy as np 

import pandas as pd

from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import StandardScaler

import seaborn as sns

import matplotlib.pyplot as plt

import scipy.stats as st

%matplotlib inline
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



    

    return x.values





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

    kf = StratifiedKFold(n_splits=folds,shuffle=True,random_state=42)

    for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]),train.target)):

        print('Fold:',i)

        blindtrain = train.loc[test_index].copy() 

        vistrain = train.loc[train_index].copy()



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

    test.drop(highcardinality,inplace=True,axis=1)



    train = blindloodata

    train.drop(highcardinality,inplace=True,axis=1)

    train = train.fillna(train.mean())

    test = test.fillna(train.mean())



    print('Scale values')

    ss = StandardScaler()

    features = train.columns[1:-1]

    ss.fit(pd.concat([train[features],test[features]]))

    train[features] = ss.transform(train[features] )

    test[features] = ss.transform(test[features] )

    train[features] = np.round(train[features], 6)

    test[features] = np.round(test[features], 6)

    return train, test
strdirectory = '../input/'

gptrain, gptest = GetData(strdirectory)
def GPX(data):

    v = pd.DataFrame()

    v["i0"] = 0.250000*np.tanh(((((((data["loo_ps_car_05_cat"] - data["loo_ps_car_02_cat"]) + (data["ps_reg_03"] * data["loo_ps_car_05_cat"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i1"] = 0.250000*np.tanh((((data["loo_ps_ind_08_bin"] + ((data["loo_ps_ind_08_bin"] - data["loo_ps_ind_18_bin"]) + data["loo_ps_ind_07_bin"])) * 2.0) * 2.0))

    v["i2"] = 0.250000*np.tanh((data["loo_ps_car_05_cat"] + ((((data["loo_ps_car_05_cat"] - data["loo_ps_ind_06_bin"]) * 2.0) - (data["loo_ps_ind_16_bin"] * 2.0)) * 2.0)))

    v["i3"] = 0.250000*np.tanh(((((data["loo_ps_ind_09_bin"] * 2.0) * 2.0) - data["missing"]) * 2.0))

    v["i4"] = 0.250000*np.tanh((data["loo_ps_car_05_cat"] + (data["loo_ps_ind_07_bin"] * (data["loo_ps_car_05_cat"] - ((data["loo_ps_car_08_cat"] - data["loo_ps_car_02_cat"]) * 2.0)))))

    v["i5"] = 0.250000*np.tanh((((data["loo_ps_ind_08_bin"] + (data["loo_ps_ind_09_bin"] + (data["missing"] * data["loo_ps_ind_08_bin"]))) * 2.0) * 2.0))

    v["i6"] = 0.250000*np.tanh(((-(data["loo_ps_ind_06_bin"])) - ((11.91003704071044922) * (data["loo_ps_ind_06_bin"] + (data["loo_ps_car_09_cat"] * 2.0)))))

    v["i7"] = 0.250000*np.tanh((((((data["missing"] * (data["loo_ps_car_04_cat"] + data["loo_ps_car_02_cat"])) - data["missing"]) * 2.0) - data["loo_ps_car_02_cat"]) * 2.0))

    v["i8"] = 0.250000*np.tanh((data["ps_ind_14"] + ((((data["ps_ind_14"] - data["loo_ps_car_10_cat"]) * 2.0) - data["loo_ps_ind_18_bin"]) - data["loo_ps_car_07_cat"])))

    v["i9"] = 0.250000*np.tanh(((data["loo_ps_ind_07_bin"] - (data["loo_ps_ind_18_bin"] * (data["loo_ps_car_04_cat"] + data["loo_ps_ind_07_bin"]))) - (data["loo_ps_car_09_cat"] * data["loo_ps_car_09_cat"])))

    v["i10"] = 0.250000*np.tanh((((data["loo_ps_ind_08_bin"] + (data["loo_ps_ind_08_bin"] * (data["ps_reg_03"] - data["loo_ps_ind_17_bin"]))) - data["loo_ps_ind_06_bin"]) - data["loo_ps_ind_06_bin"]))

    v["i11"] = 0.250000*np.tanh(((data["loo_ps_car_02_cat"] * (data["loo_ps_ind_06_bin"] - ((data["loo_ps_ind_08_bin"] - data["loo_ps_ind_06_bin"]) * 2.0))) * 2.0))

    v["i12"] = 0.250000*np.tanh((((((data["missing"] * data["missing"]) - 0.764706) * 2.0) * 2.0) - (data["ps_car_14"] * data["ps_car_12"])))

    v["i13"] = 0.250000*np.tanh(((((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_09_bin"] * 2.0)) + (data["loo_ps_car_05_cat"] + data["loo_ps_car_01_cat"])) * 2.0) + data["ps_car_15"]))

    v["i14"] = 0.250000*np.tanh((((data["loo_ps_ind_07_bin"] * 2.0) * (-((data["loo_ps_car_08_cat"] - (data["loo_ps_car_05_cat"] + data["loo_ps_car_02_cat"]))))) * 2.0))

    v["i15"] = 0.250000*np.tanh(((data["loo_ps_ind_09_bin"] + ((data["loo_ps_ind_09_bin"] * ((1.189660 - data["missing"]) - data["loo_ps_ind_17_bin"])) * 2.0)) * 2.0))

    v["i16"] = 0.250000*np.tanh(((((data["loo_ps_car_05_cat"] * 2.0) * 2.0) * (data["loo_ps_ind_07_bin"] - data["loo_ps_ind_06_bin"])) - data["loo_ps_car_10_cat"]))

    v["i17"] = 0.250000*np.tanh((data["loo_ps_ind_12_bin"] - ((data["loo_ps_car_10_cat"] + (data["loo_ps_ind_16_bin"] * (data["loo_ps_ind_06_bin"] + data["loo_ps_ind_06_bin"]))) + data["loo_ps_ind_06_bin"])))

    v["i18"] = 0.250000*np.tanh((((data["loo_ps_ind_06_bin"] + (data["loo_ps_ind_09_bin"] * 2.0)) + (data["loo_ps_ind_09_bin"] - data["ps_car_14"])) - data["ps_car_13"]))

    v["i19"] = 0.250000*np.tanh((((data["loo_ps_ind_12_bin"] - (data["loo_ps_car_10_cat"] + 1.647060)) - data["missing"]) - data["loo_ps_car_04_cat"]))

    v["i20"] = 0.250000*np.tanh(((data["ps_ind_14"] * data["ps_ind_14"]) - (data["loo_ps_car_09_cat"] + (data["loo_ps_car_03_cat"] - (data["loo_ps_car_03_cat"] * data["loo_ps_ind_17_bin"])))))

    v["i21"] = 0.250000*np.tanh(((data["ps_car_15"] - data["loo_ps_car_09_cat"]) + ((data["loo_ps_car_05_cat"] - data["loo_ps_car_08_cat"]) * (data["loo_ps_ind_18_bin"] + data["ps_reg_03"]))))

    v["i22"] = 0.250000*np.tanh(((data["loo_ps_ind_07_bin"] + (data["loo_ps_ind_07_bin"] - ((data["loo_ps_car_08_cat"] * data["loo_ps_ind_07_bin"]) + data["ps_car_12"]))) - data["loo_ps_ind_06_bin"]))

    v["i23"] = 0.250000*np.tanh(((data["ps_ind_14"] - data["loo_ps_car_10_cat"]) + ((data["loo_ps_ind_12_bin"] - data["loo_ps_car_09_cat"]) + (data["loo_ps_car_04_cat"] * data["loo_ps_ind_08_bin"]))))

    v["i24"] = 0.250000*np.tanh((data["loo_ps_ind_07_bin"] * ((data["loo_ps_car_05_cat"] * (-(data["loo_ps_car_04_cat"]))) + np.tanh((data["loo_ps_car_05_cat"] - data["loo_ps_car_08_cat"])))))

    v["i25"] = 0.250000*np.tanh(((data["missing"] * (data["loo_ps_car_02_cat"] + (data["ps_car_13"] + (data["loo_ps_car_02_cat"] + data["loo_ps_car_09_cat"])))) - data["loo_ps_car_09_cat"]))

    v["i26"] = 0.250000*np.tanh((data["ps_ind_14"] + ((data["loo_ps_ind_09_bin"] + (data["loo_ps_ind_16_bin"] - data["loo_ps_car_10_cat"])) - data["loo_ps_car_08_cat"])))

    v["i27"] = 0.250000*np.tanh((((data["loo_ps_ind_06_bin"] + data["loo_ps_ind_06_bin"]) * (data["loo_ps_car_03_cat"] * data["missing"])) + (data["loo_ps_ind_06_bin"] - data["loo_ps_car_10_cat"])))

    v["i28"] = 0.250000*np.tanh((data["loo_ps_ind_12_bin"] - ((data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_17_bin"] + data["loo_ps_car_09_cat"]) + data["loo_ps_ind_17_bin"])) * data["loo_ps_ind_09_bin"])))

    v["i29"] = 0.250000*np.tanh((data["ps_ind_14"] - (data["loo_ps_car_10_cat"] - (((data["loo_ps_car_10_cat"] - data["loo_ps_car_10_cat"]) + np.tanh((data["missing"] * 2.0)))/2.0))))

    v["i30"] = 0.250000*np.tanh(((((data["loo_ps_ind_12_bin"] - (data["missing"] * (data["loo_ps_ind_07_bin"] - (data["missing"] / 2.0)))) * 2.0) * 2.0) * 2.0))

    v["i31"] = 0.250000*np.tanh(((data["loo_ps_ind_08_bin"] - (data["loo_ps_ind_07_bin"] * data["loo_ps_car_08_cat"])) + ((data["loo_ps_ind_07_bin"] * 2.0) + data["loo_ps_ind_05_cat"])))

    v["i32"] = 0.250000*np.tanh((((data["ps_ind_14"] - data["loo_ps_car_10_cat"]) - data["loo_ps_car_10_cat"]) - ((data["ps_car_11"] + (data["ps_car_14"] * data["loo_ps_ind_18_bin"]))/2.0)))

    v["i33"] = 0.250000*np.tanh((((data["loo_ps_car_02_cat"] * (data["loo_ps_ind_17_bin"] * data["loo_ps_car_02_cat"])) + (data["loo_ps_ind_07_bin"] * data["loo_ps_car_02_cat"])) * 2.0))

    v["i34"] = 0.250000*np.tanh((((data["loo_ps_ind_09_bin"] - data["loo_ps_ind_16_bin"]) * data["loo_ps_car_01_cat"]) + ((data["loo_ps_ind_07_bin"] * (-(data["loo_ps_ind_18_bin"]))) * 2.0)))

    v["i35"] = 0.250000*np.tanh((((data["ps_reg_01"] * (-(data["loo_ps_ind_18_bin"]))) + ((data["loo_ps_car_07_cat"] * 2.0) * data["loo_ps_car_08_cat"])) + data["ps_ind_14"]))

    v["i36"] = 0.250000*np.tanh(((data["loo_ps_ind_12_bin"] + ((data["loo_ps_ind_08_bin"] * data["loo_ps_car_04_cat"]) * 2.0)) * data["loo_ps_car_05_cat"]))

    v["i37"] = 0.250000*np.tanh(((data["loo_ps_ind_18_bin"] + data["loo_ps_ind_18_bin"]) * ((-(data["loo_ps_car_05_cat"])) * data["loo_ps_ind_07_bin"])))

    v["i38"] = 0.250000*np.tanh((data["ps_ind_14"] - (((((data["loo_ps_car_10_cat"] - (data["loo_ps_ind_16_bin"] - data["loo_ps_car_10_cat"])) + data["ps_ind_14"])/2.0) + data["loo_ps_ind_07_bin"])/2.0)))

    v["i39"] = 0.250000*np.tanh(((data["loo_ps_car_02_cat"] * ((data["loo_ps_car_08_cat"] - data["loo_ps_ind_08_bin"]) + (data["loo_ps_car_08_cat"] - data["ps_reg_03"]))) + data["loo_ps_ind_12_bin"]))

    v["i40"] = 0.250000*np.tanh((data["loo_ps_ind_16_bin"] * (data["ps_car_12"] - ((data["ps_car_14"] + (data["loo_ps_car_04_cat"] + data["ps_car_15"])) + data["loo_ps_car_04_cat"]))))

    v["i41"] = 0.250000*np.tanh((data["loo_ps_ind_09_bin"] * (data["loo_ps_car_02_cat"] - ((data["loo_ps_ind_05_cat"] + ((data["loo_ps_ind_09_bin"] * data["ps_reg_02"]) / 2.0))/2.0))))

    v["i42"] = 0.250000*np.tanh(((data["loo_ps_ind_16_bin"] * (data["loo_ps_car_03_cat"] * (data["loo_ps_ind_06_bin"] * 1.800000))) - (data["loo_ps_ind_16_bin"] * data["loo_ps_ind_06_bin"])))

    v["i43"] = 0.250000*np.tanh((((data["loo_ps_ind_16_bin"] * data["loo_ps_car_05_cat"]) - data["loo_ps_car_06_cat"]) - (data["loo_ps_car_11_cat"] * (data["loo_ps_ind_16_bin"] * data["loo_ps_ind_09_bin"]))))

    v["i44"] = 0.250000*np.tanh((((data["loo_ps_ind_12_bin"] - data["ps_car_14"]) - data["loo_ps_ind_17_bin"]) + (data["loo_ps_car_02_cat"] * ((data["loo_ps_ind_17_bin"] + data["loo_ps_car_04_cat"])/2.0))))

    v["i45"] = 0.250000*np.tanh(((data["ps_reg_03"] - ((data["loo_ps_ind_18_bin"] + data["ps_reg_03"]) * data["loo_ps_ind_08_bin"])) * data["loo_ps_car_04_cat"]))

    v["i46"] = 0.250000*np.tanh((((data["loo_ps_ind_16_bin"] * data["loo_ps_ind_06_bin"]) - data["loo_ps_ind_06_bin"]) * (data["loo_ps_car_01_cat"] - (data["loo_ps_car_01_cat"] * data["loo_ps_car_02_cat"]))))

    v["i47"] = 0.250000*np.tanh((((((data["missing"] * data["loo_ps_car_09_cat"]) * data["loo_ps_car_09_cat"]) - data["loo_ps_car_10_cat"]) + (data["loo_ps_car_10_cat"] - data["loo_ps_car_10_cat"]))/2.0))

    v["i48"] = 0.250000*np.tanh((data["ps_ind_03"] * (((np.tanh((0.764706 / 2.0)) + ((0.253731 + data["ps_ind_03"])/2.0))/2.0) / 2.0)))

    v["i49"] = 0.250000*np.tanh(((((data["loo_ps_ind_12_bin"] + data["loo_ps_car_10_cat"]) * data["loo_ps_car_10_cat"]) + ((data["loo_ps_car_10_cat"] + (data["loo_ps_ind_12_bin"] + data["loo_ps_car_10_cat"]))/2.0))/2.0))

    v["i50"] = 0.250000*np.tanh((((data["loo_ps_ind_05_cat"] + data["loo_ps_car_10_cat"])/2.0) * data["loo_ps_car_10_cat"]))

    v["i51"] = 0.250000*np.tanh((data["loo_ps_car_09_cat"] * (((data["loo_ps_ind_07_bin"] + data["loo_ps_car_09_cat"])/2.0) * data["loo_ps_ind_18_bin"])))

    v["i52"] = 0.250000*np.tanh((((data["loo_ps_car_10_cat"] + np.tanh(data["loo_ps_car_10_cat"])) * data["loo_ps_car_10_cat"]) / 2.0))

    v["i53"] = 0.250000*np.tanh((((((data["loo_ps_car_10_cat"] * data["loo_ps_car_10_cat"]) - (data["loo_ps_ind_06_bin"] * data["loo_ps_car_03_cat"])) + data["loo_ps_ind_11_bin"]) + data["loo_ps_ind_12_bin"])/2.0))

    v["i54"] = 0.250000*np.tanh((-2.0 * (data["loo_ps_ind_17_bin"] + (data["loo_ps_car_05_cat"] * np.tanh(((data["loo_ps_ind_16_bin"] * 2.0) * data["loo_ps_ind_07_bin"]))))))

    v["i55"] = 0.250000*np.tanh(((data["loo_ps_car_09_cat"] * 2.0) + (0.765432 - ((data["loo_ps_car_09_cat"] + (data["ps_car_14"] * 2.0)) * data["loo_ps_car_09_cat"]))))

    v["i56"] = 0.250000*np.tanh(((data["missing"] * data["loo_ps_car_08_cat"]) - (((data["missing"] + data["loo_ps_car_10_cat"])/2.0) - ((data["loo_ps_ind_17_bin"] + data["loo_ps_car_08_cat"])/2.0))))

    v["i57"] = 0.250000*np.tanh(((-((data["ps_car_14"] + data["ps_car_14"]))) * data["loo_ps_ind_07_bin"]))

    v["i58"] = 0.250000*np.tanh(((data["ps_car_14"] * data["missing"]) + (((data["loo_ps_ind_06_bin"] + data["missing"]) * data["loo_ps_ind_06_bin"]) * data["loo_ps_car_08_cat"])))

    v["i59"] = 0.250000*np.tanh(((data["ps_car_14"] - (np.tanh(0.253731) * data["ps_reg_01"])) - (data["ps_car_14"] * data["ps_car_13"])))

    v["i60"] = 0.250000*np.tanh(((((data["loo_ps_ind_18_bin"] + (data["ps_car_13"] - data["missing"]))/2.0) + data["ps_car_13"]) * (data["missing"] * data["loo_ps_ind_07_bin"])))

    v["i61"] = 0.250000*np.tanh((((((data["loo_ps_ind_07_bin"] + (data["loo_ps_car_09_cat"] / 2.0))/2.0) + (data["missing"] * data["loo_ps_car_09_cat"]))/2.0) - data["loo_ps_car_10_cat"]))

    v["i62"] = 0.250000*np.tanh((((data["ps_car_15"] * (data["ps_car_15"] * data["loo_ps_car_02_cat"])) + ((data["loo_ps_ind_16_bin"] + (data["ps_car_12"] * data["loo_ps_ind_16_bin"]))/2.0))/2.0))

    v["i63"] = 0.250000*np.tanh((((data["ps_car_12"] * data["loo_ps_ind_09_bin"]) + (((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_09_bin"])/2.0) * (data["loo_ps_ind_18_bin"] - data["loo_ps_ind_05_cat"])))/2.0))

    v["i64"] = 0.250000*np.tanh((((data["ps_car_14"] + ((data["ps_reg_01"] + ((((data["ps_reg_01"] + data["ps_car_14"])/2.0) + data["ps_reg_01"])/2.0))/2.0))/2.0) * data["loo_ps_ind_05_cat"]))

    v["i65"] = 0.250000*np.tanh((-((data["loo_ps_ind_09_bin"] * ((data["loo_ps_ind_17_bin"] - data["loo_ps_car_01_cat"]) - (data["loo_ps_ind_08_bin"] * data["ps_reg_03"]))))))

    v["i66"] = 0.250000*np.tanh((((data["ps_reg_03"] * data["loo_ps_car_01_cat"]) - data["loo_ps_car_01_cat"]) * (data["loo_ps_ind_08_bin"] * data["ps_reg_03"])))

    v["i67"] = 0.250000*np.tanh((((data["loo_ps_car_10_cat"] * data["loo_ps_car_08_cat"]) + (data["ps_car_14"] * data["loo_ps_car_08_cat"]))/2.0))

    v["i68"] = 0.250000*np.tanh((0.363636 - ((data["ps_car_15"] + (((data["ps_car_15"] * data["loo_ps_ind_06_bin"]) + data["loo_ps_ind_06_bin"])/2.0))/2.0)))

    v["i69"] = 0.250000*np.tanh((((data["loo_ps_car_06_cat"] * data["loo_ps_ind_09_bin"]) + ((data["loo_ps_ind_09_bin"] * 2.0) * (data["loo_ps_ind_16_bin"] / 2.0)))/2.0))

    v["i70"] = 0.250000*np.tanh(((data["loo_ps_car_02_cat"] * np.tanh(np.tanh((data["loo_ps_ind_17_bin"] + (data["loo_ps_ind_17_bin"] - data["loo_ps_car_01_cat"]))))) - data["loo_ps_car_10_cat"]))

    v["i71"] = 0.250000*np.tanh((data["loo_ps_car_10_cat"] * ((data["ps_car_14"] / 2.0) * ((0.081633 * data["loo_ps_car_10_cat"]) / 2.0))))

    v["i72"] = 0.250000*np.tanh((data["loo_ps_car_08_cat"] * ((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_06_bin"])) + (data["loo_ps_ind_07_bin"] * data["loo_ps_ind_18_bin"]))))

    v["i73"] = 0.250000*np.tanh(((data["loo_ps_ind_08_bin"] * data["loo_ps_ind_18_bin"]) * (((-(data["loo_ps_car_11_cat"])) + data["ps_car_11"]) + (-(data["loo_ps_car_11_cat"])))))

    v["i74"] = 0.250000*np.tanh((-((-((-((((data["loo_ps_car_02_cat"] + (-((data["loo_ps_car_05_cat"] * data["loo_ps_car_02_cat"]))))/2.0) * data["loo_ps_ind_08_bin"]))))))))

    v["i75"] = 0.250000*np.tanh((data["loo_ps_ind_12_bin"] + ((data["loo_ps_ind_05_cat"] * data["loo_ps_ind_08_bin"]) * data["loo_ps_car_09_cat"])))

    v["i76"] = 0.250000*np.tanh(((data["loo_ps_car_05_cat"] * (data["loo_ps_ind_17_bin"] * data["loo_ps_car_08_cat"])) * data["loo_ps_car_08_cat"]))

    v["i77"] = 0.250000*np.tanh((0.081633 * ((data["loo_ps_car_10_cat"] * data["loo_ps_car_10_cat"]) - (0.0 * data["loo_ps_ind_10_bin"]))))

    v["i78"] = 0.250000*np.tanh((((((data["loo_ps_ind_12_bin"] + 0.0)/2.0) + (((-(data["loo_ps_ind_09_bin"])) + 0.0)/2.0))/2.0) * data["loo_ps_ind_17_bin"]))

    v["i79"] = 0.250000*np.tanh((((-((-((data["loo_ps_car_09_cat"] * data["loo_ps_ind_16_bin"]))))) - data["loo_ps_ind_06_bin"]) * data["loo_ps_car_10_cat"]))

    v["i80"] = 0.250000*np.tanh((data["ps_reg_03"] * (data["loo_ps_ind_07_bin"] * data["loo_ps_ind_16_bin"])))

    v["i81"] = 0.250000*np.tanh(((((data["loo_ps_car_08_cat"] * (data["loo_ps_car_09_cat"] * data["loo_ps_car_08_cat"])) + data["loo_ps_car_08_cat"]) + data["ps_car_14"]) * data["loo_ps_car_08_cat"]))

    v["i82"] = 0.250000*np.tanh((((data["loo_ps_ind_07_bin"] * data["loo_ps_ind_18_bin"]) * data["ps_reg_01"]) + ((data["loo_ps_ind_07_bin"] * data["loo_ps_ind_18_bin"]) * data["loo_ps_car_08_cat"])))

    v["i83"] = 0.250000*np.tanh((((data["loo_ps_car_04_cat"] - (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_08_bin"])) * 2.0) * (data["loo_ps_car_08_cat"] * data["loo_ps_car_05_cat"])))

    v["i84"] = 0.250000*np.tanh((-((((-((data["loo_ps_ind_17_bin"] * np.tanh((-(data["loo_ps_ind_09_bin"])))))) + ((data["loo_ps_car_10_cat"] + data["loo_ps_ind_07_bin"])/2.0))/2.0))))

    v["i85"] = 0.250000*np.tanh((((data["loo_ps_ind_07_bin"] * data["ps_car_12"]) + (data["loo_ps_ind_07_bin"] * (np.tanh(data["loo_ps_car_11_cat"]) * data["ps_car_11"])))/2.0))

    v["i86"] = 0.250000*np.tanh(((data["loo_ps_car_09_cat"] + (data["loo_ps_ind_18_bin"] / 2.0)) * (-((data["loo_ps_car_09_cat"] * (data["loo_ps_ind_09_bin"] + data["loo_ps_ind_09_bin"]))))))

    v["i87"] = 0.250000*np.tanh(((data["loo_ps_ind_07_bin"] * data["loo_ps_ind_18_bin"]) * (data["loo_ps_car_09_cat"] + (np.tanh((data["missing"] * 2.0)) + data["loo_ps_ind_12_bin"]))))

    v["i88"] = 0.250000*np.tanh(((((data["loo_ps_ind_09_bin"] + data["loo_ps_ind_16_bin"])/2.0) + (np.tanh((data["loo_ps_ind_16_bin"] + data["loo_ps_car_02_cat"])) * data["loo_ps_ind_09_bin"]))/2.0))

    v["i89"] = 0.250000*np.tanh((((np.tanh(np.tanh(data["ps_ind_01"])) / 2.0) + (-(0.253731)))/2.0))

    v["i90"] = 0.250000*np.tanh((((((data["loo_ps_car_09_cat"] * data["loo_ps_ind_06_bin"]) / 2.0) + (data["loo_ps_car_04_cat"] * (-(data["loo_ps_ind_09_bin"]))))/2.0) * data["loo_ps_ind_16_bin"]))

    v["i91"] = 0.250000*np.tanh((data["ps_reg_03"] * (data["loo_ps_car_05_cat"] * (((-((data["loo_ps_car_02_cat"] - data["ps_car_11"]))) / 2.0) - data["loo_ps_car_08_cat"]))))

    v["i92"] = 0.250000*np.tanh((-(((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_18_bin"])/2.0) * ((data["missing"] + data["loo_ps_car_08_cat"])/2.0)) * (data["ps_car_14"] * 2.0)))))

    v["i93"] = 0.250000*np.tanh((((data["loo_ps_car_05_cat"] + data["loo_ps_car_04_cat"]) * ((data["loo_ps_car_08_cat"] + data["loo_ps_car_08_cat"]) * data["loo_ps_ind_08_bin"])) * data["ps_reg_03"]))

    v["i94"] = 0.250000*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_car_08_cat"] + (data["loo_ps_car_02_cat"] * (-(data["loo_ps_car_08_cat"]))))/2.0)))

    v["i95"] = 0.250000*np.tanh((data["loo_ps_ind_06_bin"] * (((data["loo_ps_car_08_cat"] + data["loo_ps_car_07_cat"])/2.0) - (data["loo_ps_car_08_cat"] * (data["loo_ps_ind_09_bin"] * data["loo_ps_car_03_cat"])))))

    v["i96"] = 0.250000*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_car_08_cat"] * (data["loo_ps_ind_06_bin"] * data["loo_ps_car_03_cat"])) + (data["loo_ps_car_02_cat"] * data["loo_ps_ind_08_bin"]))))

    v["i97"] = 0.250000*np.tanh((((data["loo_ps_car_03_cat"] * data["loo_ps_car_02_cat"]) - ((data["loo_ps_car_03_cat"] * data["loo_ps_car_04_cat"]) * data["loo_ps_car_08_cat"])) * data["loo_ps_ind_08_bin"]))

    v["i98"] = 0.250000*np.tanh(((data["loo_ps_ind_16_bin"] * (data["loo_ps_car_08_cat"] * data["loo_ps_car_05_cat"])) + ((data["loo_ps_car_03_cat"] * data["ps_car_12"]) / 2.0)))

    v["i99"] = 0.250000*np.tanh((((data["loo_ps_car_09_cat"] * (-(data["loo_ps_car_05_cat"]))) + (np.tanh(data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_05_cat"]))/2.0))

    return v.sum(axis=1)





def GPY(data):

    v = pd.DataFrame()

    v["i0"] = 0.250000*np.tanh(((((((data["loo_ps_car_07_cat"] - data["loo_ps_car_05_cat"]) + (data["loo_ps_ind_17_bin"] - data["loo_ps_ind_16_bin"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["i1"] = 0.250000*np.tanh(((((((data["ps_reg_03"] * data["loo_ps_ind_07_bin"]) + data["loo_ps_car_08_cat"]) * 2.0) - data["loo_ps_ind_08_bin"]) * 2.0) + data["loo_ps_ind_07_bin"]))

    v["i2"] = 0.250000*np.tanh(((data["loo_ps_car_08_cat"] + (((data["loo_ps_ind_06_bin"] * data["loo_ps_ind_16_bin"]) - data["ps_car_14"]) - data["loo_ps_ind_06_bin"])) * 2.0))

    v["i3"] = 0.250000*np.tanh(((((data["missing"] - data["ps_car_14"]) + ((data["ps_reg_03"] + data["ps_car_14"]) * data["loo_ps_ind_09_bin"])) * 2.0) * 2.0))

    v["i4"] = 0.250000*np.tanh((data["loo_ps_car_07_cat"] + (((data["loo_ps_car_05_cat"] * (data["loo_ps_car_08_cat"] + data["loo_ps_car_08_cat"])) - data["ps_car_14"]) * 2.0)))

    v["i5"] = 0.250000*np.tanh(((data["loo_ps_car_04_cat"] + (data["loo_ps_car_08_cat"] - data["loo_ps_ind_12_bin"])) + (data["loo_ps_car_08_cat"] - data["ps_car_14"])))

    v["i6"] = 0.250000*np.tanh(((((data["loo_ps_car_09_cat"] + data["loo_ps_ind_18_bin"]) * data["loo_ps_ind_06_bin"]) + (data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_06_bin"] * 2.0))) * 2.0))

    v["i7"] = 0.250000*np.tanh(((data["loo_ps_car_03_cat"] - data["loo_ps_ind_16_bin"]) + ((data["loo_ps_car_03_cat"] + (2.050000 + data["loo_ps_car_03_cat"])) - data["loo_ps_ind_16_bin"])))

    v["i8"] = 0.250000*np.tanh(((((data["loo_ps_ind_17_bin"] + data["loo_ps_car_07_cat"])/2.0) - data["ps_ind_14"]) + ((data["loo_ps_car_07_cat"] + 0.715909)/2.0)))

    v["i9"] = 0.250000*np.tanh(((-((data["loo_ps_ind_07_bin"] - (data["loo_ps_car_07_cat"] + (data["loo_ps_car_09_cat"] - ((data["ps_car_11"] + data["loo_ps_car_07_cat"])/2.0)))))) * 2.0))

    v["i10"] = 0.250000*np.tanh(((data["loo_ps_car_08_cat"] + (data["loo_ps_ind_06_bin"] * (data["loo_ps_car_08_cat"] + ((data["loo_ps_ind_16_bin"] * 2.0) * 2.0)))) * 2.0))

    v["i11"] = 0.250000*np.tanh(((data["loo_ps_car_07_cat"] - (data["loo_ps_ind_06_bin"] - ((data["loo_ps_car_08_cat"] - data["loo_ps_ind_06_bin"]) + data["loo_ps_car_07_cat"]))) - data["loo_ps_ind_18_bin"]))

    v["i12"] = 0.250000*np.tanh((((data["loo_ps_ind_17_bin"] + data["ps_car_14"]) + data["ps_car_14"]) * (data["missing"] + ((data["ps_car_14"] + data["ps_car_12"])/2.0))))

    v["i13"] = 0.250000*np.tanh(((data["loo_ps_car_05_cat"] - data["ps_car_11"]) * (-((3.0 * (data["loo_ps_car_01_cat"] * data["loo_ps_ind_09_bin"]))))))

    v["i14"] = 0.250000*np.tanh(((((data["loo_ps_car_05_cat"] + data["loo_ps_car_05_cat"])/2.0) + (data["loo_ps_car_05_cat"] - data["loo_ps_ind_07_bin"])) * ((data["loo_ps_car_08_cat"] + data["loo_ps_ind_16_bin"])/2.0)))

    v["i15"] = 0.250000*np.tanh(((((data["loo_ps_car_07_cat"] - data["ps_car_11"]) * data["missing"]) * 2.0) + (data["loo_ps_ind_09_bin"] * (-(data["loo_ps_ind_17_bin"])))))

    v["i16"] = 0.250000*np.tanh(((data["loo_ps_car_07_cat"] + (data["loo_ps_car_01_cat"] + (-(((data["loo_ps_ind_07_bin"] * data["loo_ps_car_04_cat"]) * data["loo_ps_car_04_cat"])))))/2.0))

    v["i17"] = 0.250000*np.tanh((data["loo_ps_ind_06_bin"] - (data["loo_ps_ind_06_bin"] * (((data["loo_ps_ind_16_bin"] + data["loo_ps_ind_16_bin"]) * (data["loo_ps_car_05_cat"] * 2.0)) * 2.0))))

    v["i18"] = 0.250000*np.tanh(((data["loo_ps_car_02_cat"] - data["ps_car_14"]) + (data["loo_ps_car_02_cat"] * ((data["ps_car_14"] * 2.0) * data["loo_ps_car_05_cat"]))))

    v["i19"] = 0.250000*np.tanh((((data["loo_ps_car_03_cat"] + data["loo_ps_car_03_cat"]) - data["loo_ps_ind_12_bin"]) + (0.818182 - data["loo_ps_ind_12_bin"])))

    v["i20"] = 0.250000*np.tanh((data["missing"] + ((((data["loo_ps_car_06_cat"] + data["loo_ps_ind_17_bin"]) + data["loo_ps_car_06_cat"]) * data["ps_car_14"]) - data["ps_car_14"])))

    v["i21"] = 0.250000*np.tanh((((np.tanh(0.764706) + (data["loo_ps_car_05_cat"] - data["loo_ps_ind_12_bin"])) + data["loo_ps_car_05_cat"])/2.0))

    v["i22"] = 0.250000*np.tanh(((((data["ps_reg_03"] - data["loo_ps_car_02_cat"]) - (data["ps_reg_03"] * data["loo_ps_car_02_cat"])) - data["loo_ps_car_08_cat"]) * data["loo_ps_ind_07_bin"]))

    v["i23"] = 0.250000*np.tanh(((((data["ps_reg_03"] - data["ps_car_11"]) - data["loo_ps_car_04_cat"]) - data["loo_ps_car_04_cat"]) * 2.0))

    v["i24"] = 0.250000*np.tanh((-((data["loo_ps_ind_07_bin"] * (data["loo_ps_car_08_cat"] + ((data["loo_ps_car_04_cat"] - data["ps_reg_03"]) - (-(data["loo_ps_car_04_cat"]))))))))

    v["i25"] = 0.250000*np.tanh((data["loo_ps_ind_17_bin"] + (np.tanh(1.647060) + (data["missing"] * (-(data["loo_ps_car_02_cat"]))))))

    v["i26"] = 0.250000*np.tanh((data["ps_reg_03"] - (data["loo_ps_ind_16_bin"] + (((data["loo_ps_ind_09_bin"] * (data["loo_ps_ind_16_bin"] - data["ps_reg_03"])) * 2.0) * 2.0))))

    v["i27"] = 0.250000*np.tanh((data["loo_ps_car_03_cat"] * (data["loo_ps_car_07_cat"] - (data["loo_ps_car_03_cat"] + (data["ps_car_14"] + (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_06_bin"]))))))

    v["i28"] = 0.250000*np.tanh((((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_06_bin"] - data["loo_ps_car_09_cat"])) - data["loo_ps_ind_10_bin"]) - data["loo_ps_ind_12_bin"]))

    v["i29"] = 0.250000*np.tanh((((-(data["loo_ps_car_06_cat"])) - data["loo_ps_ind_09_bin"]) - (((data["loo_ps_car_10_cat"] * (data["loo_ps_ind_09_bin"] * 2.0)) + data["missing"])/2.0)))

    v["i30"] = 0.250000*np.tanh(((((data["loo_ps_car_07_cat"] / 2.0) / 2.0) * data["loo_ps_car_07_cat"]) - data["loo_ps_car_07_cat"]))

    v["i31"] = 0.250000*np.tanh((data["loo_ps_ind_08_bin"] * ((np.tanh(data["loo_ps_car_11_cat"]) - (data["loo_ps_car_08_cat"] * data["loo_ps_car_03_cat"])) - data["loo_ps_car_03_cat"])))

    v["i32"] = 0.250000*np.tanh((((data["ps_car_14"] * ((data["loo_ps_ind_08_bin"] + data["loo_ps_ind_18_bin"]) + data["loo_ps_ind_17_bin"])) + data["loo_ps_ind_18_bin"]) * 2.0))

    v["i33"] = 0.250000*np.tanh((data["loo_ps_ind_07_bin"] * ((data["loo_ps_car_02_cat"] * (data["loo_ps_ind_17_bin"] - ((data["loo_ps_car_02_cat"] + data["loo_ps_ind_07_bin"])/2.0))) * 2.0)))

    v["i34"] = 0.250000*np.tanh((data["loo_ps_car_01_cat"] * (data["loo_ps_ind_09_bin"] - (data["loo_ps_car_03_cat"] - (data["loo_ps_car_01_cat"] - (data["loo_ps_car_03_cat"] - data["loo_ps_ind_09_bin"]))))))

    v["i35"] = 0.250000*np.tanh(((((data["ps_car_14"] - data["loo_ps_ind_08_bin"]) * data["loo_ps_ind_17_bin"]) + (data["loo_ps_car_08_cat"] * (data["ps_car_14"] - data["loo_ps_ind_18_bin"])))/2.0))

    v["i36"] = 0.250000*np.tanh((data["loo_ps_ind_09_bin"] * ((data["ps_reg_03"] - data["loo_ps_car_04_cat"]) - ((data["ps_reg_03"] - data["loo_ps_car_04_cat"]) * data["loo_ps_ind_18_bin"]))))

    v["i37"] = 0.250000*np.tanh((((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_16_bin"] * 2.0)) * (data["missing"] * 2.0)) + (data["ps_car_13"] * data["ps_car_13"])))

    v["i38"] = 0.250000*np.tanh((data["loo_ps_ind_07_bin"] * ((data["ps_reg_01"] - (data["loo_ps_ind_16_bin"] * (data["loo_ps_ind_06_bin"] - data["loo_ps_car_08_cat"]))) - data["loo_ps_car_08_cat"])))

    v["i39"] = 0.250000*np.tanh((-((data["loo_ps_ind_18_bin"] * ((-(data["loo_ps_car_09_cat"])) + (((data["ps_reg_03"] + data["loo_ps_ind_18_bin"])/2.0) + data["ps_reg_03"]))))))

    v["i40"] = 0.250000*np.tanh((data["loo_ps_car_04_cat"] * ((data["ps_car_14"] + data["loo_ps_car_03_cat"]) + data["loo_ps_ind_16_bin"])))

    v["i41"] = 0.250000*np.tanh((((-(data["loo_ps_ind_09_bin"])) + ((data["loo_ps_car_07_cat"] * (data["loo_ps_car_07_cat"] * data["loo_ps_car_05_cat"])) / 2.0))/2.0))

    v["i42"] = 0.250000*np.tanh((((data["loo_ps_ind_12_bin"] + (0.081633 / 2.0)) - data["loo_ps_car_03_cat"]) * (data["loo_ps_ind_08_bin"] * 2.0)))

    v["i43"] = 0.250000*np.tanh((data["loo_ps_ind_07_bin"] * ((data["missing"] - ((data["missing"] * data["loo_ps_ind_08_bin"]) / 2.0)) * (data["loo_ps_ind_16_bin"] * 2.0))))

    v["i44"] = 0.250000*np.tanh(((((data["loo_ps_car_02_cat"] * 2.0) + (data["loo_ps_car_08_cat"] * 2.0)) * data["loo_ps_ind_07_bin"]) * (data["loo_ps_ind_16_bin"] - data["ps_reg_03"])))

    v["i45"] = 0.250000*np.tanh((((data["loo_ps_ind_08_bin"] - data["loo_ps_car_04_cat"]) - (data["loo_ps_ind_07_bin"] * data["loo_ps_car_02_cat"])) * data["ps_reg_03"]))

    v["i46"] = 0.250000*np.tanh((((data["loo_ps_car_04_cat"] + data["loo_ps_ind_16_bin"])/2.0) * data["loo_ps_ind_06_bin"]))

    v["i47"] = 0.250000*np.tanh(((data["loo_ps_car_03_cat"] - (data["loo_ps_car_08_cat"] - data["loo_ps_car_02_cat"])) * (((data["loo_ps_ind_16_bin"] + data["loo_ps_car_08_cat"])/2.0) + data["loo_ps_ind_09_bin"])))

    v["i48"] = 0.250000*np.tanh((((data["loo_ps_car_03_cat"] * ((data["loo_ps_ind_17_bin"] + data["loo_ps_car_08_cat"]) * data["loo_ps_ind_06_bin"])) - data["loo_ps_ind_06_bin"]) * data["loo_ps_car_02_cat"]))

    v["i49"] = 0.250000*np.tanh((data["ps_reg_03"] * (-((data["ps_car_13"] - ((data["loo_ps_ind_09_bin"] / 2.0) - data["loo_ps_ind_18_bin"]))))))

    v["i50"] = 0.250000*np.tanh((data["ps_reg_03"] * ((-(data["loo_ps_ind_07_bin"])) * (data["loo_ps_car_02_cat"] * (-(((data["loo_ps_ind_17_bin"] * 2.0) * 2.0)))))))

    v["i51"] = 0.250000*np.tanh((data["loo_ps_ind_17_bin"] * ((data["loo_ps_car_02_cat"] + data["loo_ps_car_02_cat"]) * (data["loo_ps_ind_07_bin"] * (data["ps_car_14"] + data["ps_reg_03"])))))

    v["i52"] = 0.250000*np.tanh((data["loo_ps_car_02_cat"] * (((data["loo_ps_car_08_cat"] + ((data["loo_ps_ind_17_bin"] + data["ps_car_13"])/2.0)) + data["loo_ps_car_10_cat"]) - data["loo_ps_ind_06_bin"])))

    v["i53"] = 0.250000*np.tanh((data["loo_ps_car_05_cat"] * (data["loo_ps_car_08_cat"] + ((data["loo_ps_car_04_cat"] + data["loo_ps_car_04_cat"]) * data["loo_ps_car_04_cat"]))))

    v["i54"] = 0.250000*np.tanh((data["loo_ps_ind_17_bin"] * (data["loo_ps_ind_17_bin"] * ((data["loo_ps_ind_17_bin"] * (data["loo_ps_car_06_cat"] - data["loo_ps_ind_09_bin"])) - data["loo_ps_ind_17_bin"]))))

    v["i55"] = 0.250000*np.tanh(((((data["ps_car_14"] + (data["loo_ps_car_04_cat"] + (data["ps_car_14"] - 0.253731))) + data["loo_ps_car_09_cat"])/2.0) * data["loo_ps_ind_06_bin"]))

    v["i56"] = 0.250000*np.tanh(((data["loo_ps_car_07_cat"] * data["loo_ps_ind_05_cat"]) + (data["loo_ps_ind_08_bin"] * ((data["loo_ps_car_08_cat"] * data["loo_ps_ind_07_bin"]) * data["loo_ps_ind_07_bin"]))))

    v["i57"] = 0.250000*np.tanh((data["ps_car_14"] * ((data["loo_ps_ind_07_bin"] + (((data["loo_ps_ind_09_bin"] * data["ps_car_11"]) - 0.764706) - data["loo_ps_ind_09_bin"]))/2.0)))

    v["i58"] = 0.250000*np.tanh((data["ps_car_14"] * (data["loo_ps_car_08_cat"] + (data["loo_ps_car_06_cat"] - (data["loo_ps_car_08_cat"] * np.tanh(data["ps_car_14"]))))))

    v["i59"] = 0.250000*np.tanh((((data["ps_car_13"] + (data["loo_ps_ind_12_bin"] / 2.0))/2.0) * (-((1.0 + data["ps_car_13"])))))

    v["i60"] = 0.250000*np.tanh((((data["loo_ps_ind_11_bin"] - (((data["loo_ps_car_09_cat"] + data["loo_ps_car_09_cat"]) + data["loo_ps_ind_07_bin"])/2.0)) / 2.0) - data["loo_ps_ind_12_bin"]))

    v["i61"] = 0.250000*np.tanh(((data["missing"] * (data["loo_ps_ind_17_bin"] - ((((data["missing"] + data["loo_ps_ind_06_bin"])/2.0) + data["ps_car_14"])/2.0))) * data["loo_ps_ind_06_bin"]))

    v["i62"] = 0.250000*np.tanh(((((data["ps_car_12"] + (data["loo_ps_car_07_cat"] * (data["loo_ps_car_07_cat"] - 2.840000)))/2.0) / 2.0) - 0.363636))

    v["i63"] = 0.250000*np.tanh(((data["loo_ps_ind_18_bin"] * (data["loo_ps_ind_09_bin"] - (data["ps_reg_03"] / 2.0))) * (data["loo_ps_car_09_cat"] - data["ps_reg_03"])))

    v["i64"] = 0.250000*np.tanh(((data["loo_ps_ind_06_bin"] * 2.0) * ((-((data["loo_ps_car_02_cat"] * ((data["ps_reg_01"] + data["ps_reg_01"])/2.0)))) - data["loo_ps_car_02_cat"])))

    v["i65"] = 0.250000*np.tanh((-(((data["loo_ps_car_02_cat"] * data["loo_ps_ind_18_bin"]) - ((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_09_bin"]) * data["loo_ps_car_02_cat"])))))

    v["i66"] = 0.250000*np.tanh((data["loo_ps_car_02_cat"] * (data["loo_ps_ind_08_bin"] - (data["loo_ps_ind_07_bin"] * data["loo_ps_car_02_cat"]))))

    v["i67"] = 0.250000*np.tanh((((-(3.736840)) * data["loo_ps_car_08_cat"]) * (data["missing"] * (data["loo_ps_ind_17_bin"] * data["loo_ps_ind_06_bin"]))))

    v["i68"] = 0.250000*np.tanh((data["loo_ps_car_02_cat"] * (((((data["ps_car_14"] + data["loo_ps_ind_06_bin"])/2.0) * 2.0) * 2.0) * (data["loo_ps_car_03_cat"] + data["loo_ps_car_05_cat"]))))

    v["i69"] = 0.250000*np.tanh(((data["loo_ps_car_05_cat"] * (data["loo_ps_ind_09_bin"] - data["loo_ps_ind_16_bin"])) * data["loo_ps_car_08_cat"]))

    v["i70"] = 0.250000*np.tanh((((data["loo_ps_car_10_cat"] / 2.0) * data["loo_ps_car_10_cat"]) - (data["loo_ps_car_10_cat"] * 2.0)))

    v["i71"] = 0.250000*np.tanh((data["loo_ps_ind_06_bin"] * (((np.tanh(data["ps_car_11"]) + data["loo_ps_ind_16_bin"]) + (data["loo_ps_car_01_cat"] * data["loo_ps_car_05_cat"])) * 2.0)))

    v["i72"] = 0.250000*np.tanh((-((data["loo_ps_ind_09_bin"] * (((((data["loo_ps_ind_17_bin"] * data["loo_ps_ind_17_bin"]) + data["loo_ps_car_09_cat"])/2.0) + data["loo_ps_car_10_cat"])/2.0)))))

    v["i73"] = 0.250000*np.tanh((data["loo_ps_car_02_cat"] * ((data["loo_ps_ind_07_bin"] * data["loo_ps_ind_18_bin"]) - ((data["loo_ps_ind_07_bin"] * data["loo_ps_car_08_cat"]) * data["loo_ps_ind_17_bin"]))))

    v["i74"] = 0.250000*np.tanh((((data["loo_ps_car_02_cat"] * 2.0) + (data["loo_ps_ind_16_bin"] * 2.0)) * (-((data["loo_ps_car_05_cat"] * data["loo_ps_ind_08_bin"])))))

    v["i75"] = 0.250000*np.tanh(((data["loo_ps_ind_17_bin"] * data["ps_car_13"]) * (((data["loo_ps_car_07_cat"] - data["loo_ps_ind_08_bin"]) - data["loo_ps_ind_08_bin"]) - data["loo_ps_car_09_cat"])))

    v["i76"] = 0.250000*np.tanh(((data["loo_ps_car_05_cat"] * data["loo_ps_car_08_cat"]) * (data["loo_ps_car_01_cat"] + ((0.683333 + data["loo_ps_ind_07_bin"])/2.0))))

    v["i77"] = 0.250000*np.tanh(((data["loo_ps_car_04_cat"] + (data["loo_ps_car_02_cat"] * data["loo_ps_ind_06_bin"])) * (data["loo_ps_ind_16_bin"] + (data["loo_ps_ind_16_bin"] * data["ps_reg_03"]))))

    v["i78"] = 0.250000*np.tanh(((data["loo_ps_car_04_cat"] * 2.0) * (((data["ps_reg_01"] - data["loo_ps_car_09_cat"]) - data["loo_ps_ind_17_bin"]) * (-(data["loo_ps_ind_09_bin"])))))

    v["i79"] = 0.250000*np.tanh((data["loo_ps_ind_07_bin"] * ((data["loo_ps_ind_07_bin"] * (data["missing"] * (data["loo_ps_ind_07_bin"] * data["loo_ps_ind_16_bin"]))) + data["loo_ps_ind_16_bin"])))

    v["i80"] = 0.250000*np.tanh((((data["loo_ps_ind_09_bin"] - (data["loo_ps_ind_06_bin"] * data["ps_car_11"])) * data["loo_ps_car_01_cat"]) / 2.0))

    v["i81"] = 0.250000*np.tanh(((((data["ps_car_11"] + data["ps_car_11"]) + data["loo_ps_ind_09_bin"])/2.0) * ((data["loo_ps_car_08_cat"] - data["loo_ps_car_01_cat"]) * data["loo_ps_car_05_cat"])))

    v["i82"] = 0.250000*np.tanh((((data["loo_ps_car_07_cat"] * (data["loo_ps_ind_05_cat"] * data["loo_ps_car_07_cat"])) + (((data["loo_ps_ind_18_bin"] - data["loo_ps_car_08_cat"]) + data["loo_ps_ind_10_bin"])/2.0))/2.0))

    v["i83"] = 0.250000*np.tanh(((data["loo_ps_car_10_cat"] * data["loo_ps_car_10_cat"]) / 2.0))

    v["i84"] = 0.250000*np.tanh(((((((data["loo_ps_car_07_cat"] / 2.0) * data["loo_ps_ind_05_cat"]) / 2.0) - (data["loo_ps_ind_09_bin"] + data["loo_ps_car_01_cat"])) / 2.0) / 2.0))

    v["i85"] = 0.250000*np.tanh((data["loo_ps_ind_05_cat"] * (data["loo_ps_ind_05_cat"] * ((data["ps_car_11"] + ((data["loo_ps_ind_07_bin"] + data["loo_ps_ind_07_bin"])/2.0))/2.0))))

    v["i86"] = 0.250000*np.tanh((data["loo_ps_ind_06_bin"] * (data["loo_ps_ind_08_bin"] * ((-(((data["loo_ps_car_09_cat"] + (data["loo_ps_car_09_cat"] / 2.0))/2.0))) * data["loo_ps_car_09_cat"]))))

    v["i87"] = 0.250000*np.tanh((data["loo_ps_ind_18_bin"] * (((-(data["loo_ps_car_04_cat"])) + ((data["missing"] / 2.0) + ((data["loo_ps_ind_11_bin"] + data["missing"])/2.0)))/2.0)))

    v["i88"] = 0.250000*np.tanh(((((data["loo_ps_ind_09_bin"] * data["loo_ps_car_02_cat"]) * (data["loo_ps_car_05_cat"] * data["loo_ps_car_02_cat"])) - data["loo_ps_ind_09_bin"]) * data["loo_ps_car_04_cat"]))

    v["i89"] = 0.250000*np.tanh((data["loo_ps_ind_09_bin"] * ((-((((-(data["loo_ps_car_02_cat"])) * data["loo_ps_ind_09_bin"]) + data["loo_ps_car_02_cat"]))) * data["loo_ps_ind_18_bin"])))

    v["i90"] = 0.250000*np.tanh((-(((data["loo_ps_car_10_cat"] + ((((data["ps_reg_02"] + (data["missing"] / 2.0))/2.0) + 0.363636)/2.0))/2.0))))

    v["i91"] = 0.250000*np.tanh(((((-(((data["loo_ps_car_02_cat"] + 0.081633)/2.0))) + (data["loo_ps_ind_11_bin"] / 2.0))/2.0) / 2.0))

    v["i92"] = 0.250000*np.tanh((((data["loo_ps_ind_18_bin"] * data["ps_car_14"]) + (data["ps_car_14"] * np.tanh(data["loo_ps_car_08_cat"])))/2.0))

    v["i93"] = 0.250000*np.tanh(np.tanh((data["loo_ps_ind_08_bin"] * ((data["loo_ps_car_04_cat"] + (data["ps_car_14"] * data["loo_ps_car_04_cat"]))/2.0))))

    v["i94"] = 0.250000*np.tanh(((((data["loo_ps_car_05_cat"] * (data["ps_car_13"] - (data["ps_car_13"] * data["loo_ps_ind_17_bin"]))) + data["ps_car_13"])/2.0) / 2.0))

    v["i95"] = 0.250000*np.tanh(((data["loo_ps_ind_11_bin"] + data["loo_ps_ind_11_bin"])/2.0))

    v["i96"] = 0.250000*np.tanh(((data["loo_ps_ind_08_bin"] + (data["loo_ps_ind_16_bin"] * (data["missing"] * ((-(data["loo_ps_ind_08_bin"])) * (-(data["loo_ps_ind_08_bin"]))))))/2.0))

    v["i97"] = 0.250000*np.tanh(((0.173913 + 0.173913)/2.0))

    v["i98"] = 0.250000*np.tanh(((((((data["loo_ps_ind_17_bin"] + data["loo_ps_ind_17_bin"])/2.0) + (data["loo_ps_car_05_cat"] * data["ps_ind_03"]))/2.0) / 2.0) / 2.0))

    v["i99"] = 0.250000*np.tanh(((((data["loo_ps_ind_16_bin"] * (-(data["loo_ps_car_09_cat"]))) + data["loo_ps_ind_07_bin"])/2.0) * data["loo_ps_car_05_cat"]))

    return v.sum(axis=1)
colors = ['red','blue']
plt.figure(figsize=(15,15))

plt.scatter(GPX(gptrain),

            GPY(gptrain),

            s=[x for x in gptrain.target],

            color=[colors[x] for x in gptrain.target])
plt.figure(figsize=(15,15))

plt.scatter(GPX(gptrain),

            GPY(gptrain),

            s=[1-x for x in gptrain.target],

            color=[colors[x] for x in gptrain.target])
x = GPX(gptrain[::500]) # This is really slow

y = GPY(gptrain[::500])

print(x.shape)

xmin, xmax = x.min(), x.max()

ymin, ymax = y.min(), y.max()



# Peform the kernel density estimate

xx, yy = np.mgrid[xmin:xmax:x.shape[0]*1j, ymin:ymax:x.shape[0]*1j]

positions = np.vstack([xx.ravel(), yy.ravel()])

values = np.vstack([x, y])

kernel = st.gaussian_kde(values)

f = np.reshape(kernel(positions).T, xx.shape)



fig = plt.figure(figsize=(15,15))

ax = fig.gca()

ax.set_xlim(xmin, xmax)

ax.set_ylim(ymin, ymax)

# Contourf plot

fset = ax.contourf(xx, yy, f, cmap='Blues')

## Or kernel density estimate plot instead of the contourf plot

# Contour plot

cset = ax.contour(xx, yy, f, colors='k')

ax.clabel(cset, inline=1, fontsize=10)

#ax.imshow(np.rot90(f), cmap='Blues', extent=[xmin, xmax, ymin, ymax])



# Label plot



ax.set_xlabel('X')

ax.set_ylabel('Y')



plt.show()