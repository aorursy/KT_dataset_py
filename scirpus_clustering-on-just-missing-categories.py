import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
def GiniScore(y_actual, y_pred):

    return 2*roc_auc_score(y_actual, y_pred)-1
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

test.insert(1,'target',np.nan)

features = list(['id'])+list(train.columns[2:-1])+list(['target'])

train = train[features].copy()

test = test[features].copy()
highcardinality =[]

for i in train.columns[1:-1]:

    if(((i.find('bin')!=-1) or (i.find('cat')!=-1))):

        highcardinality.append(i)



highcardinality
x = pd.concat([train[highcardinality],test[highcardinality]])

x = (x==-1).astype(int)
def GP1(data):

    v = pd.DataFrame()

    v["0"] = 1.000000*np.tanh(((data["ps_car_03_cat"] + ((7.50136470794677734) - ((data["ps_car_03_cat"] * 2.0) * (data["ps_car_03_cat"] * (9.0))))) * 2.0))

    v["1"] = 1.000000*np.tanh(((((0.51183354854583740) + (((data["ps_ind_05_cat"] - data["ps_car_09_cat"]) * 2.0) * 2.0))/2.0) + (data["ps_ind_05_cat"] * 2.0)))

    v["2"] = 0.971870*np.tanh((((((data["ps_ind_18_bin"] + (((2.02591228485107422) + data["ps_ind_18_bin"])/2.0))/2.0) + data["ps_car_02_cat"])/2.0) / 2.0))

    v["3"] = 1.000000*np.tanh((((14.54255104064941406) - (((8.0) * (14.54255104064941406)) * ((data["ps_car_03_cat"] + data["ps_car_09_cat"])/2.0))) * 2.0))

    v["4"] = 1.000000*np.tanh(((((((1.0) * 2.0) - ((data["ps_car_05_cat"] * 2.0) * 2.0)) - data["ps_car_05_cat"]) - data["ps_car_05_cat"]) * 2.0))

    v["5"] = 1.000000*np.tanh((((1.0 - ((data["ps_car_05_cat"] + data["ps_car_03_cat"]) + (data["ps_car_07_cat"] * 2.0))) * 2.0) * 2.0))

    v["6"] = 1.000000*np.tanh((((((((1.0 + 1.0)/2.0) + np.tanh(2.0))/2.0) + (1.15680480003356934))/2.0) - data["ps_car_05_cat"]))

    v["7"] = 1.000000*np.tanh((((((((data["ps_car_01_cat"] * 2.0) * 2.0) - data["ps_car_07_cat"]) * 2.0) - data["ps_car_07_cat"]) * 2.0) - data["ps_car_07_cat"]))

    v["8"] = 1.000000*np.tanh(((((data["ps_ind_05_cat"] - (data["ps_car_05_cat"] * (data["ps_ind_05_cat"] * 2.0))) * 2.0) * 2.0) * 2.0))

    v["9"] = 1.000000*np.tanh(((((((data["ps_ind_05_cat"] * 2.0) - data["ps_car_09_cat"]) * 2.0) * 2.0) * 2.0) - data["ps_car_09_cat"]))

    v["10"] = 1.000000*np.tanh(((((((data["ps_car_07_cat"] * (-(data["ps_car_07_cat"]))) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0))

    v["11"] = 1.000000*np.tanh((((((data["ps_ind_02_cat"] + ((data["ps_ind_05_cat"] * 2.0) - data["ps_car_09_cat"])) * 2.0) * 2.0) * 2.0) * 2.0))

    v["12"] = 1.000000*np.tanh((((-(((data["ps_car_03_cat"] + ((data["ps_car_03_cat"] + -2.0)/2.0))/2.0))) / 2.0) - ((data["ps_car_09_cat"] * 2.0) * 2.0)))

    v["13"] = 1.000000*np.tanh((data["ps_car_07_cat"] * (((((data["ps_ind_05_cat"] * 2.0) - data["ps_car_07_cat"]) * 2.0) * 2.0) * 2.0)))

    v["14"] = 1.000000*np.tanh(((10.0) * (((10.0) * ((data["ps_ind_05_cat"] * 2.0) - data["ps_car_09_cat"])) * 2.0)))

    v["15"] = 1.000000*np.tanh(((6.0) * (((((((data["ps_ind_02_cat"] * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0) * 2.0)))

    v["16"] = 1.000000*np.tanh(((((data["ps_car_07_cat"] * (((data["ps_ind_05_cat"] * 2.0) - data["ps_car_07_cat"]) * 2.0)) * 2.0) - data["ps_ind_05_cat"]) * 2.0))

    v["17"] = 0.999609*np.tanh((((6.63926267623901367) * data["ps_ind_02_cat"]) * data["ps_ind_02_cat"]))

    v["18"] = 1.000000*np.tanh((((((((data["ps_ind_02_cat"] * 2.0) - data["ps_car_07_cat"]) * 2.0) * 2.0) * 2.0) * 2.0) - np.tanh(data["ps_car_01_cat"])))

    v["19"] = 1.000000*np.tanh(((((data["ps_car_09_cat"] * 2.0) * (data["ps_car_07_cat"] * 2.0)) - (data["ps_car_09_cat"] + (data["ps_car_07_cat"] * 2.0))) * 2.0))

    v["20"] = 0.999414*np.tanh(((((((0.01044631283730268) * 2.0) + ((0.01044631283730268) - ((data["ps_car_03_cat"] + -1.0)/2.0)))/2.0) + (0.01044631283730268))/2.0))

    v["21"] = 0.999023*np.tanh(((((np.tanh(-1.0) / 2.0) + ((data["ps_car_05_cat"] + ((data["ps_car_05_cat"] + data["ps_ind_18_bin"])/2.0))/2.0))/2.0) / 2.0))

    v["22"] = 1.000000*np.tanh(((7.22747135162353516) * (((-((data["ps_car_03_cat"] * (data["ps_ind_05_cat"] * 2.0)))) + data["ps_ind_05_cat"]) * (7.22747135162353516))))

    v["23"] = 0.988474*np.tanh(((data["ps_ind_17_bin"] * (0.0 / 2.0)) * data["ps_ind_09_bin"]))

    v["24"] = 1.000000*np.tanh(((data["ps_car_07_cat"] * data["ps_car_01_cat"]) - ((data["ps_car_07_cat"] + ((data["ps_car_07_cat"] / 2.0) + data["ps_car_07_cat"]))/2.0)))

    v["25"] = 1.000000*np.tanh((((((data["ps_car_07_cat"] * (data["ps_car_09_cat"] * 2.0)) - data["ps_car_09_cat"]) * 2.0) - data["ps_car_09_cat"]) * 2.0))

    v["26"] = 1.000000*np.tanh((data["ps_car_09_cat"] * ((((data["ps_car_07_cat"] - data["ps_car_09_cat"]) * 2.0) * 2.0) - (data["ps_car_09_cat"] * 2.0))))

    v["27"] = 0.999609*np.tanh(((((data["ps_car_05_cat"] + (0.01999140344560146)) + data["ps_car_05_cat"]) * (-(data["ps_car_07_cat"]))) + (0.01999140344560146)))

    v["28"] = 0.965618*np.tanh((-(np.tanh(data["ps_ind_13_bin"]))))

    v["29"] = 1.000000*np.tanh((data["ps_car_09_cat"] * ((((data["ps_car_09_cat"] * ((data["ps_car_07_cat"] - data["ps_car_09_cat"]) * 2.0)) * 2.0) * 2.0) * 2.0)))

    v["30"] = 1.000000*np.tanh(((data["ps_car_09_cat"] - data["ps_car_01_cat"]) * (-((data["ps_car_09_cat"] - data["ps_car_01_cat"])))))

    v["31"] = 0.809728*np.tanh(((data["ps_car_05_cat"] * ((0.00353694055229425) * 2.0)) + (((data["ps_ind_04_cat"] / 2.0) + ((0.00353336427360773) * data["ps_car_05_cat"]))/2.0)))

    v["32"] = 1.000000*np.tanh(((((((data["ps_car_05_cat"] * 2.0) - data["ps_car_09_cat"]) * 2.0) * data["ps_car_09_cat"]) * 2.0) * 2.0))

    v["33"] = 1.000000*np.tanh(((((data["ps_car_09_cat"] * data["ps_car_03_cat"]) * 2.0) + (-(((np.tanh(data["ps_car_03_cat"]) + (-1.0 / 2.0))/2.0))))/2.0))

    v["34"] = 1.000000*np.tanh((((data["ps_car_03_cat"] * (data["ps_car_09_cat"] * 2.0)) + ((((-((data["ps_car_03_cat"] / 2.0))) + data["ps_car_05_cat"])/2.0) / 2.0))/2.0))

    v["35"] = 1.000000*np.tanh(((8.28782367706298828) * ((data["ps_ind_05_cat"] - ((data["ps_ind_05_cat"] * 2.0) * (data["ps_car_03_cat"] * data["ps_car_03_cat"]))) * 2.0)))

    v["36"] = 1.000000*np.tanh((-(((data["ps_car_09_cat"] / 2.0) - (data["ps_car_09_cat"] * (((data["ps_car_09_cat"] * 2.0) * data["ps_car_05_cat"]) * 2.0))))))

    v["37"] = 0.941395*np.tanh((((data["ps_car_09_cat"] * (data["ps_car_07_cat"] + data["ps_car_07_cat"])) * 2.0) - (data["ps_car_09_cat"] + data["ps_car_07_cat"])))

    v["38"] = 1.000000*np.tanh(((7.0) * (data["ps_car_09_cat"] * (data["ps_car_05_cat"] + (((data["ps_car_07_cat"] - data["ps_car_09_cat"]) + data["ps_car_07_cat"])/2.0)))))

    v["39"] = 1.000000*np.tanh(((-(((((((np.tanh(data["ps_ind_02_cat"]) + data["ps_ind_02_cat"])/2.0) / 2.0) / 2.0) + data["ps_ind_02_cat"])/2.0))) / 2.0))

    v["40"] = 0.726900*np.tanh((((data["ps_ind_02_cat"] + data["ps_ind_04_cat"])/2.0) + (data["ps_ind_04_cat"] * (-((data["ps_ind_02_cat"] + (data["ps_ind_04_cat"] * 2.0)))))))

    v["41"] = 1.000000*np.tanh(((((((data["ps_car_09_cat"] + data["ps_car_09_cat"])/2.0) * data["ps_car_07_cat"]) / 2.0) - data["ps_car_05_cat"]) * data["ps_car_07_cat"]))

    v["42"] = 0.700918*np.tanh(((data["ps_ind_04_cat"] + ((-(data["ps_ind_02_cat"])) + data["ps_ind_04_cat"])) * (-(data["ps_ind_04_cat"]))))

    v["43"] = 1.000000*np.tanh(((10.0) * ((((data["ps_car_07_cat"] * data["ps_ind_05_cat"]) * 2.0) - ((data["ps_car_07_cat"] + data["ps_ind_05_cat"])/2.0)) * 2.0)))

    v["44"] = 1.000000*np.tanh((((data["ps_car_09_cat"] + (((data["ps_car_07_cat"] / 2.0) / 2.0) / 2.0))/2.0) - ((data["ps_car_01_cat"] * data["ps_car_09_cat"]) * 2.0)))

    v["45"] = 1.000000*np.tanh((((data["ps_ind_04_cat"] + (data["ps_ind_04_cat"] + data["ps_car_01_cat"]))/2.0) - (data["ps_car_01_cat"] * (data["ps_ind_02_cat"] * 2.0))))

    v["46"] = 1.000000*np.tanh(((11.72634220123291016) * (data["ps_ind_05_cat"] - ((data["ps_ind_05_cat"] * 2.0) * data["ps_car_05_cat"]))))

    v["47"] = 1.000000*np.tanh(((7.77970266342163086) * (((((data["ps_ind_05_cat"] * 2.0) * data["ps_car_07_cat"]) * 2.0) - data["ps_ind_05_cat"]) - data["ps_car_07_cat"])))

    v["48"] = 0.988865*np.tanh((-2.0 - (np.tanh((np.tanh((-3.0 + -2.0)) + -2.0)) * 2.0)))

    v["49"] = 0.799180*np.tanh(((-(((data["ps_ind_11_bin"] - data["ps_car_11_cat"]) / 2.0))) * 2.0))

    return (v.sum(axis=1))



def GP2(data):

    v = pd.DataFrame()

    v["0"] = 1.000000*np.tanh((((data["ps_car_03_cat"] + (((data["ps_car_03_cat"] + np.tanh(data["ps_car_03_cat"])) + -3.0)/2.0)) * 2.0) * 2.0))

    v["1"] = 1.000000*np.tanh((((data["ps_car_09_cat"] + data["ps_ind_05_cat"]) * 2.0) + (((data["ps_ind_05_cat"] * 2.0) + np.tanh(-1.0))/2.0)))

    v["2"] = 0.971870*np.tanh(((((0.02136111818253994) + (0.02136469446122646)) + ((0.02136469446122646) + ((((0.02136469446122646) / 2.0) + (0.02136469446122646))/2.0)))/2.0))

    v["3"] = 1.000000*np.tanh(((((-1.0 + data["ps_car_03_cat"]) * 2.0) + (((data["ps_car_03_cat"] / 2.0) + data["ps_car_03_cat"])/2.0))/2.0))

    v["4"] = 1.000000*np.tanh(((((((data["ps_car_03_cat"] - 0.0) + data["ps_ind_05_cat"]) - (data["ps_car_05_cat"] * 2.0)) * 2.0) * 2.0) * 2.0))

    v["5"] = 1.000000*np.tanh((((data["ps_car_07_cat"] + ((data["ps_car_03_cat"] - (data["ps_car_05_cat"] * 2.0)) * 2.0)) * 2.0) * 2.0))

    v["6"] = 1.000000*np.tanh((((((2.0 - (data["ps_car_05_cat"] * 2.0)) - data["ps_car_05_cat"]) - data["ps_car_05_cat"]) * 2.0) * 2.0))

    v["7"] = 1.000000*np.tanh(((((data["ps_car_03_cat"] + (data["ps_car_07_cat"] - ((data["ps_car_05_cat"] + 1.0)/2.0))) * 2.0) * 2.0) * 2.0))

    v["8"] = 1.000000*np.tanh(((((data["ps_car_03_cat"] - data["ps_car_05_cat"]) * (data["ps_car_03_cat"] - data["ps_car_05_cat"])) - data["ps_ind_05_cat"]) * (9.0)))

    v["9"] = 1.000000*np.tanh((((data["ps_car_09_cat"] - data["ps_ind_05_cat"]) * 2.0) - (((np.tanh(-1.0) + data["ps_car_05_cat"])/2.0) / 2.0)))

    v["10"] = 1.000000*np.tanh(((data["ps_car_03_cat"] + ((-2.0 + ((data["ps_car_03_cat"] + ((9.0) * ((data["ps_car_07_cat"] * 2.0) * 2.0)))/2.0))/2.0))/2.0))

    v["11"] = 1.000000*np.tanh(((((data["ps_car_09_cat"] + ((data["ps_car_03_cat"] - data["ps_car_09_cat"]) * data["ps_ind_05_cat"])) * 2.0) - data["ps_ind_05_cat"]) * 2.0))

    v["12"] = 1.000000*np.tanh((((0.04997135326266289) / 2.0) + (data["ps_car_09_cat"] * 2.0)))

    v["13"] = 1.000000*np.tanh(((((data["ps_car_07_cat"] - data["ps_ind_05_cat"]) * 2.0) * ((data["ps_car_07_cat"] - data["ps_car_05_cat"]) + data["ps_car_07_cat"])) * 2.0))

    v["14"] = 1.000000*np.tanh((((data["ps_car_09_cat"] + ((data["ps_ind_05_cat"] * (-(data["ps_car_09_cat"]))) * 2.0)) * 2.0) - data["ps_car_04_cat"]))

    v["15"] = 1.000000*np.tanh((((-(data["ps_ind_02_cat"])) * data["ps_car_01_cat"]) - ((data["ps_car_01_cat"] + data["ps_ind_02_cat"])/2.0)))

    v["16"] = 1.000000*np.tanh(((((np.tanh((data["ps_car_07_cat"] + data["ps_ind_05_cat"])) - (data["ps_ind_05_cat"] * data["ps_car_07_cat"])) * 2.0) * 2.0) * 2.0))

    v["17"] = 0.999609*np.tanh((np.tanh((0.03399253636598587)) * (((-1.0 * (0.03399253636598587)) + np.tanh(-1.0))/2.0)))

    v["18"] = 1.000000*np.tanh((((((data["ps_car_07_cat"] - data["ps_ind_04_cat"]) * 2.0) * 2.0) * 2.0) * 2.0))

    v["19"] = 1.000000*np.tanh(((np.tanh(data["ps_car_07_cat"]) + ((((-(data["ps_car_07_cat"])) - (data["ps_car_09_cat"] / 2.0)) * 2.0) * data["ps_car_09_cat"]))/2.0))

    v["20"] = 0.999414*np.tanh((((0.03869176656007767) * data["ps_car_03_cat"]) + ((-(((data["ps_ind_17_bin"] + (0.03869176656007767))/2.0))) - (0.03869176656007767))))

    v["21"] = 0.999023*np.tanh((-(((((((((data["ps_car_05_cat"] + data["ps_ind_10_bin"])/2.0) / 2.0) + -1.0)/2.0) + data["ps_car_05_cat"])/2.0) / 2.0))))

    v["22"] = 1.000000*np.tanh((((((data["ps_ind_05_cat"] * (data["ps_car_05_cat"] * 2.0)) - data["ps_ind_05_cat"]) * 2.0) * 2.0) * 2.0))

    v["23"] = 0.988474*np.tanh(((data["ps_car_01_cat"] + ((((data["ps_car_03_cat"] + -1.0)/2.0) + (data["ps_car_03_cat"] - (data["ps_car_03_cat"] * data["ps_car_05_cat"])))/2.0))/2.0))

    v["24"] = 1.000000*np.tanh(((data["ps_car_07_cat"] + ((data["ps_car_07_cat"] * (data["ps_car_07_cat"] * 2.0)) * (data["ps_ind_04_cat"] * 2.0)))/2.0))

    v["25"] = 1.000000*np.tanh((-((((((data["ps_car_09_cat"] + data["ps_car_09_cat"]) + data["ps_car_09_cat"]) * data["ps_car_07_cat"]) * 2.0) - data["ps_car_09_cat"]))))

    v["26"] = 1.000000*np.tanh((-((((data["ps_car_09_cat"] * 2.0) * 2.0) * (((data["ps_car_07_cat"] * 2.0) * 2.0) * data["ps_car_07_cat"])))))

    v["27"] = 0.999609*np.tanh((((data["ps_car_07_cat"] * 2.0) * (data["ps_car_05_cat"] * 2.0)) * ((data["ps_car_07_cat"] * 2.0) * 2.0)))

    v["28"] = 0.965618*np.tanh((data["ps_car_07_cat"] + (((-1.0 + np.tanh(2.0))/2.0) - np.tanh((np.tanh(data["ps_car_07_cat"]) * 2.0)))))

    v["29"] = 1.000000*np.tanh(((((data["ps_car_07_cat"] * 2.0) * 2.0) * (((data["ps_car_07_cat"] * data["ps_ind_04_cat"]) * 2.0) - data["ps_car_09_cat"])) * 2.0))

    v["30"] = 1.000000*np.tanh((((((data["ps_car_09_cat"] * (data["ps_car_01_cat"] * 2.0)) * 2.0) * data["ps_car_09_cat"]) * 2.0) - data["ps_car_01_cat"]))

    v["31"] = 0.809728*np.tanh(((0.0 * 2.0) * 2.0))

    v["32"] = 1.000000*np.tanh((data["ps_car_09_cat"] + (((data["ps_car_09_cat"] * 2.0) * (data["ps_car_09_cat"] * (-(data["ps_car_05_cat"])))) * (10.99655914306640625))))

    v["33"] = 1.000000*np.tanh((-((((data["ps_car_03_cat"] + (data["ps_car_09_cat"] * (data["ps_car_07_cat"] * data["ps_car_09_cat"]))) * data["ps_car_09_cat"]) * 2.0))))

    v["34"] = 1.000000*np.tanh((((6.06971168518066406) * np.tanh((((-(data["ps_car_03_cat"])) * 2.0) * 2.0))) * np.tanh(data["ps_car_09_cat"])))

    v["35"] = 1.000000*np.tanh(((data["ps_ind_05_cat"] * (data["ps_car_03_cat"] * (data["ps_ind_12_bin"] + ((11.35804939270019531) / 2.0)))) - data["ps_ind_05_cat"]))

    v["36"] = 1.000000*np.tanh(((data["ps_car_09_cat"] / 2.0) + (((-(data["ps_car_05_cat"])) * (data["ps_car_09_cat"] * 2.0)) * 2.0)))

    v["37"] = 0.941395*np.tanh((((data["ps_car_07_cat"] * (-(data["ps_car_09_cat"]))) + ((((data["ps_ind_12_bin"] + data["ps_car_07_cat"])/2.0) + data["ps_car_07_cat"])/2.0))/2.0))

    v["38"] = 1.000000*np.tanh(((data["ps_car_05_cat"] * 2.0) * ((data["ps_car_05_cat"] * ((((data["ps_car_07_cat"] / 2.0) + data["ps_car_07_cat"])/2.0) * 2.0)) * 2.0)))

    v["39"] = 1.000000*np.tanh(((-((((-((data["ps_ind_11_bin"] / 2.0))) + data["ps_car_11_cat"])/2.0))) * data["ps_ind_11_bin"]))

    v["40"] = 0.726900*np.tanh((((data["ps_ind_04_cat"] * (data["ps_car_01_cat"] * data["ps_car_07_cat"])) * (data["ps_car_07_cat"] * 2.0)) - data["ps_ind_02_cat"]))

    v["41"] = 1.000000*np.tanh((((data["ps_car_07_cat"] * (data["ps_car_05_cat"] * 2.0)) * (data["ps_car_07_cat"] * 2.0)) - (data["ps_car_09_cat"] * data["ps_car_07_cat"])))

    v["42"] = 0.700918*np.tanh((((data["ps_ind_02_cat"] / 2.0) + ((data["ps_ind_04_cat"] * 2.0) * ((data["ps_ind_04_cat"] * data["ps_ind_02_cat"]) * data["ps_ind_02_cat"])))/2.0))

    v["43"] = 1.000000*np.tanh((np.tanh(np.tanh(((data["ps_car_07_cat"] / 2.0) / 2.0))) - (((data["ps_ind_05_cat"] * 2.0) - data["ps_ind_05_cat"]) / 2.0)))

    v["44"] = 1.000000*np.tanh((((data["ps_car_09_cat"] * data["ps_car_01_cat"]) * np.tanh(np.tanh(data["ps_car_01_cat"]))) * 2.0))

    v["45"] = 1.000000*np.tanh((((((data["ps_ind_07_bin"] - (np.tanh(0.0) / 2.0)) + data["ps_car_01_cat"])/2.0) - data["ps_ind_04_cat"]) / 2.0))

    v["46"] = 1.000000*np.tanh(((((data["ps_car_05_cat"] * ((data["ps_car_05_cat"] * 2.0) * data["ps_ind_05_cat"])) * 2.0) * (data["ps_car_05_cat"] * 2.0)) * 2.0))

    v["47"] = 1.000000*np.tanh((-(np.tanh(((data["ps_ind_18_bin"] + (((np.tanh(data["ps_car_03_cat"]) / 2.0) * data["ps_car_09_cat"]) * 2.0))/2.0)))))

    v["48"] = 0.988865*np.tanh((((np.tanh(((((0.30820375680923462) - data["ps_ind_06_bin"]) * np.tanh(data["ps_car_03_cat"])) / 2.0)) / 2.0) / 2.0) / 2.0))

    v["49"] = 0.799180*np.tanh(np.tanh(np.tanh(np.tanh(np.tanh(((data["ps_car_03_cat"] * (0.03774047642946243)) - (0.03774047642946243)))))))

    return (v.sum(axis=1))
x = train[highcardinality].copy()

x = (x==-1).astype(int)

x['x1'] = GP1(x)

x['x2'] = GP2(x)

x['target'] = train.target.ravel()

grpcounts = x.groupby(['x1','x2']).target.count().reset_index()

grpcounts.columns = ['x1','x2','grpcounts']

x = x.merge(grpcounts,on=['x1','x2'])

grpmeans = x.groupby(['x1','x2']).target.mean().reset_index()

grpmeans.columns = ['x1','x2','grpmeans']

x = x.merge(grpmeans,on=['x1','x2'])
x.head()
colors = ['red','blue']

plt.figure(figsize=(15,15))

plt.scatter(x.x1,x.x2,s=np.floor(x.grpcounts/1000).astype(int))
GiniScore(x.target,x.grpmeans)