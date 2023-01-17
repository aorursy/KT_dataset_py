import gc

import numpy as np

import pandas as pd

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score
directory = '../input/'

train = pd.read_csv(directory+'train.csv')

test = pd.read_csv(directory+'test.csv')

test.insert(1,'target',np.nan)
train['missing'] = (train==-1).sum(axis=1).astype(float)

test['missing'] = (test==-1).sum(axis=1).astype(float)
feats = list(set(train.columns).difference(set(['id','target'])))

feats = list(['id'])+feats +list(['target'])

train = train[feats]

test = test[feats]
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
highcardinality =[]

for i in train.columns[1:-1]:

    if((train[i].dtype!='float64')&((i.find('bin')!=-1) or (i.find('cat')!=-1))):

        highcardinality.append(i)

blindloodata = None

folds = 10

kf = KFold(n_splits=folds,shuffle=True,random_state=42)

for i, (train_index, test_index) in enumerate(kf.split(range(train.shape[0]))):

    print('Fold:',i)

    blindtrain = train.loc[test_index].copy() 

    vistrain = train.loc[train_index].copy()







    for c in highcardinality:

        blindtrain.insert(1,'loo'+c, ProjectOnMean(vistrain,

                                                   blindtrain,c))

    if(blindloodata is None):

        blindloodata = blindtrain.copy()

    else:

        blindloodata = pd.concat([blindloodata,blindtrain])



for c in highcardinality:

    test.insert(1,'loo'+c, ProjectOnMean(train,

                                           test,c))

test.drop(highcardinality,inplace=True,axis=1)



train = blindloodata

train.drop(highcardinality,inplace=True,axis=1)

train = train.fillna(train.mean())

test = test.fillna(train.mean())
ss = StandardScaler()

features = train.columns[1:-1]

ss.fit(pd.concat([train[features],test[features]]))

train[features] = ss.transform(train[features] )

test[features] = ss.transform(test[features] )
def GiniScore(y_actual, y_pred):

  return 2*roc_auc_score(y_actual, y_pred)-1



def Outputs(p):

    return 1./(1.+np.exp(-p))



def GP(data):

    p = (1.000000*np.tanh(((((data["loops_ind_04_cat"] * (-(data["ps_calc_02"]))) - (12.36611652374267578)) - (12.36611270904541016)) * (12.36611270904541016))) +

        1.000000*np.tanh((data["ps_car_14"] - (66.0 + ((((data["loops_ind_16_bin"] + data["loops_ind_08_bin"])/2.0) - data["loops_car_08_cat"]) * (-(data["loops_ind_04_cat"])))))) +

        1.000000*np.tanh(((data["loops_car_07_cat"] - np.tanh((np.tanh((data["loops_car_07_cat"] + data["ps_reg_03"])) + np.tanh(np.tanh((-(data["ps_reg_03"]))))))) - 6.083330)) +

        1.000000*np.tanh((((((((data["loops_ind_06_bin"] + data["loops_ind_05_cat"]) + data["loops_ind_17_bin"])/2.0) + ((-3.0 + data["loops_car_01_cat"])/2.0))/2.0) + np.tanh(((data["ps_reg_03"] + data["ps_car_13"])/2.0)))/2.0)) +

        1.000000*np.tanh((0.076923 * ((data["ps_ind_03"] + (data["loops_ind_09_bin"] + ((data["loops_car_09_cat"] + data["loops_car_11_cat"])/2.0))) - (0.729412 - (-(data["ps_ind_15"])))))) +

        1.000000*np.tanh(((data["loops_ind_05_cat"] - ((((data["loops_calc_17_bin"] + data["missing"])/2.0) + (data["loops_car_07_cat"] * ((data["ps_car_13"] + data["loops_ind_05_cat"])/2.0)))/2.0)) * 0.076923)) +

        1.000000*np.tanh((0.123077 * np.tanh((data["loops_ind_02_cat"] + ((data["loops_car_07_cat"] * data["loops_car_07_cat"]) + ((((data["loops_ind_16_bin"] + data["ps_car_15"])/2.0) + data["loops_ind_04_cat"])/2.0)))))) +

        0.999805*np.tanh((np.tanh(data["loops_ind_02_cat"]) * ((0.057692 * (data["ps_ind_03"] * data["ps_ind_03"])) - ((0.076923 + 0.123077) * data["ps_ind_03"])))) +

        1.000000*np.tanh((-((0.076923 * (((((data["ps_car_15"] - data["ps_ind_03"]) + data["ps_reg_03"]) * data["loops_ind_06_bin"]) + ((data["ps_car_11"] + data["ps_reg_03"])/2.0))/2.0))))) +

        0.935144*np.tanh((0.057692 * (((data["ps_reg_01"] - data["loops_ind_05_cat"]) + ((data["ps_car_11"] * (-(data["loops_car_04_cat"]))) + (data["ps_ind_01"] * data["loops_ind_05_cat"])))/2.0))) +

        1.000000*np.tanh((0.076923 * ((((-1.0 + (-(data["ps_ind_15"])))/2.0) + (((data["loops_ind_09_bin"] * (data["loops_car_03_cat"] + data["ps_ind_01"])) + data["loops_car_03_cat"])/2.0))/2.0))) +

        1.000000*np.tanh((-((0.057692 * ((data["loops_car_10_cat"] + ((((data["loops_car_10_cat"] + (data["ps_reg_03"] + data["loops_ind_17_bin"])) * data["ps_car_13"]) + data["loops_car_01_cat"])/2.0))/2.0))))) +

        1.000000*np.tanh((0.076923 * np.tanh(((data["ps_ind_01"] * data["loops_car_05_cat"]) + (((data["ps_ind_01"] + data["loops_car_05_cat"]) + (data["ps_ind_03"] * data["loops_car_05_cat"]))/2.0))))) +

        0.999805*np.tanh((0.123077 * np.tanh(((((data["loops_ind_05_cat"] * data["loops_ind_05_cat"]) * ((data["loops_ind_07_bin"] + data["ps_reg_02"])/2.0)) + ((data["loops_car_09_cat"] + data["loops_ind_05_cat"])/2.0))/2.0)))) +

        1.000000*np.tanh((0.123077 * np.tanh((data["ps_reg_03"] * (((-(data["loops_car_01_cat"])) + (((-(data["ps_ind_01"])) + ((data["ps_reg_03"] + data["ps_calc_14"])/2.0))/2.0))/2.0))))) +

        0.939051*np.tanh((0.123077 * np.tanh((data["loops_ind_02_cat"] * (data["ps_car_15"] - (((0.987805 + data["ps_calc_10"]) + data["loops_ind_04_cat"]) + data["loops_ind_04_cat"])))))) +

        0.999805*np.tanh((0.076923 * ((np.tanh(data["loops_car_07_cat"]) + np.tanh((((data["loops_car_07_cat"] * data["loops_ind_17_bin"]) + data["ps_car_15"]) - data["loops_ind_17_bin"])))/2.0))) +

        0.954679*np.tanh(((0.057692 * ((data["ps_ind_03"] + (data["loops_ind_09_bin"] - ((((0.057692 + data["ps_reg_01"])/2.0) + data["ps_reg_01"])/2.0)))/2.0)) * data["ps_ind_03"])) +

        0.999805*np.tanh((0.057692 * (((data["ps_ind_01"] * (-(((data["ps_ind_01"] + data["ps_ind_15"])/2.0)))) + (-((((-(data["ps_ind_01"])) + data["loops_car_04_cat"])/2.0))))/2.0))) +

        1.000000*np.tanh((0.057692 * np.tanh(((data["ps_reg_01"] * (-(data["ps_reg_03"]))) + (((data["loops_car_06_cat"] + data["ps_reg_01"])/2.0) + np.tanh(data["loops_ind_04_cat"])))))))

    return Outputs(p)
print(GiniScore(train.target,GP(train)))
sub = pd.read_csv(directory+'sample_submission.csv')

sub.target = GP(test).values

sub.to_csv('gp_266.csv',index=False)