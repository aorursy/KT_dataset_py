import pandas as pd

import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
train = pd.read_csv('../input/voice.csv')

mapping = {'male': 0, 'female': 1}

train.label = train.label.apply(lambda s: mapping.get(s))

pca = PCA(n_components=None, copy=True, whiten=False)

features = train.columns[:-1]

train[features] = pca.fit_transform(train[features])
def Outputs(p):

    return 1./(1.+np.exp(-p))





def GPClusterOne(data):

    p = (1.000000*np.tanh(((data["sp.ent"] - (((data["median"] / data["sp.ent"]) + data["sp.ent"])/2.0)) - ((0.890909 - data["median"]) / data["sp.ent"]))) +

        1.000000*np.tanh((((((-(((data["Q75"] / 0.061728) + data["Q75"]))) + ((0.058823 / 2.0) - (data["sp.ent"] / 0.058823)))/2.0) - (data["sp.ent"] / 0.058823)) - data["median"])) +

        1.000000*np.tanh((np.tanh(((((data["sd"] / 2.0) / 2.0) + (-(0.318310)))/2.0)) - np.tanh(data["median"]))) +

        1.000000*np.tanh((((((((data["sp.ent"] / 2.0) / 2.0) / 2.0) + (10.0)) * (((data["mode"] + data["sfm"]) - (data["sp.ent"] / 2.0)) - ((data["meanfun"] / 2.0) * 2.0))) * 2.0) * 2.0)) +

        1.000000*np.tanh(((data["centroid"] / ((0.058823 + ((data["minfun"] * ((9.0) * 4.333330)) * (data["dfrange"] * data["minfun"])))/2.0)) + (data["minfun"] * ((9.0) * 4.333330)))) +

        1.000000*np.tanh((((((data["mode"] + (np.tanh((0.061728 + (data["mode"] / 0.160000))) * 2.0))/2.0) / 2.0) + (data["mode"] / 0.061728)) + 0.160000)) +

        1.000000*np.tanh((((0.160000 - (np.tanh((((data["Q25"] * data["meanfun"]) * 2.0) + (data["Q75"] - data["IQR"]))) * 2.0)) - data["Q25"]) - (((14.10039424896240234) * data["meanfun"]) * 2.0))) +

        1.000000*np.tanh(((data["sfm"] * (6.0)) * 2.0)) +

        1.000000*np.tanh((((data["skew"] + (0.061728 + (data["centroid"] / 0.061728)))/2.0) * 2.0)) +

        0.394000*np.tanh((4.409090 * (data["minfun"] - ((10.0) * ((data["maxfun"] + (data["sp.ent"] * ((data["dfrange"] + data["minfun"])/2.0)))/2.0))))) +

        1.000000*np.tanh((np.tanh(data["mode"]) + ((((data["mode"] + data["maxfun"]) - data["mode"]) + ((data["mode"] + ((data["mode"] - data["sp.ent"]) * 2.0)) * 2.0)) * 2.0))) +

        1.000000*np.tanh((np.tanh((((0.058823 * 0.058823) + ((0.160000 * 0.061728) / data["sp.ent"]))/2.0)) - data["Q75"])) +

        1.000000*np.tanh((-((((data["sp.ent"] - data["sp.ent"]) * 2.0) + ((np.tanh((data["sp.ent"] - data["sfm"])) / 0.117647) - 0.117647))))) +

        1.000000*np.tanh((data["Q25"] * (data["Q25"] - ((1.0 + data["skew"])/2.0)))) +

        0.817000*np.tanh(((((-(0.318310)) + ((data["sd"] + (data["sd"] * (((0.890909 / 2.0) + ((data["dfrange"] + data["sd"])/2.0))/2.0)))/2.0))/2.0) / 2.0)) +

        1.000000*np.tanh((((data["meandom"] * 2.0) * (-(np.tanh(data["modindx"])))) - (0.275862 - ((data["minfun"] * 2.0) * 2.0)))) +

        1.000000*np.tanh((((data["IQR"] - data["kurt"]) - data["kurt"]) + (((data["mode"] + (((data["skew"] + data["kurt"])/2.0) - data["kurt"]))/2.0) - data["kurt"]))) +

        1.000000*np.tanh((data["maxfun"] * (data["meandom"] - (data["maxfun"] + ((((9.62289524078369141) * 2.0) + (((data["skew"] + ((9.62289524078369141) * 2.0)) + 0.319588)/2.0))/2.0))))) +

        1.000000*np.tanh((((-(data["Q75"])) / 2.0) / 2.0)) +

        1.000000*np.tanh(((data["mindom"] * (((0.061728 - ((data["minfun"] + 1.570796)/2.0)) * 2.0) / 2.0)) * 0.275862)) +

        1.000000*np.tanh(((data["mode"] + data["mode"]) + data["skew"])) +

        1.000000*np.tanh((((-(((0.061728 * (-(data["dfrange"]))) + ((0.160000 / 2.0) * ((((data["sp.ent"] + data["meanfreq"]) + data["IQR"])/2.0) * 0.160000))))) + 0.117647)/2.0)) +

        1.000000*np.tanh(((((0.916667 * data["sp.ent"]) * (data["mindom"] / 2.0)) * 0.058823) - data["sp.ent"])) +

        1.000000*np.tanh((data["median"] / ((((((((data["skew"] - 0.318310) * 2.0) + (0.65020340681076050))/2.0) - 3.141593) - data["median"]) - 3.141593) + (3.141593 - 3.141593)))) +

        1.000000*np.tanh((data["sp.ent"] - (((data["sp.ent"] * (4.333330 * 2.0)) + data["Q75"])/2.0))) +

        1.000000*np.tanh(((data["mode"] * 2.0) * 2.0)) +

        1.000000*np.tanh((data["meanfun"] * ((0.636620 - (((data["meanfun"] * (4.333330 + 4.333330)) + (data["meandom"] * data["minfun"])) + 4.333330)) - (9.27417945861816406)))) +

        0.777000*np.tanh((((-(data["sp.ent"])) * 0.636620) - ((data["Q75"] * 2.0) / 2.0))) +

        1.000000*np.tanh((data["minfun"] / (((np.tanh(0.319588) - (((-((data["minfun"] / 0.319588))) + np.tanh(data["Q75"]))/2.0)) / 2.0) / 2.0))) +

        1.000000*np.tanh(((-((data["kurt"] + ((data["sp.ent"] / np.tanh(((data["sp.ent"] + (data["kurt"] + (data["sp.ent"] + data["sp.ent"]))) + 2.0))) * 2.0)))) * 2.0)) +

        1.000000*np.tanh((((((np.tanh(data["skew"]) + np.tanh(data["dfrange"]))/2.0) + data["skew"]) + (data["centroid"] + ((0.117647 * 0.0) * 2.0)))/2.0)) +

        1.000000*np.tanh(((-(data["meanfun"])) * ((data["meanfun"] + ((data["meanfreq"] + ((data["meanfreq"] / 2.0) * ((data["sd"] / 2.0) * 2.0)))/2.0))/2.0))) +

        0.875000*np.tanh((((-(data["Q25"])) - (data["median"] / 2.0)) / (4.047620 + ((-(np.tanh(0.275862))) / (0.319588 / 2.0))))) +

        0.984000*np.tanh(((((data["sfm"] + data["centroid"]) * 2.0) * 2.0) + ((((data["centroid"] * 2.0) + ((data["sfm"] + data["sfm"])/2.0)) + (np.tanh(data["sfm"]) / 2.0)) - data["sp.ent"]))) +

        1.000000*np.tanh((data["minfun"] * (data["sd"] + (((data["median"] * data["minfun"]) + (data["median"] / 2.0)) * 2.0)))) +

        0.858000*np.tanh(np.tanh(np.tanh((0.058823 + (((data["IQR"] + np.tanh(((0.117647 + (0.058823 + data["centroid"]))/2.0))) * 2.0) - data["Q75"]))))) +

        1.000000*np.tanh(((-(data["skew"])) * ((data["sd"] + (((-(data["maxfun"])) / 2.0) * 0.0))/2.0))) +

        1.000000*np.tanh((((((data["skew"] + np.tanh(((-(data["median"])) * (((-((-(data["median"])))) / 2.0) / 2.0))))/2.0) - (((data["median"] + (data["modindx"] / 2.0))/2.0) / 2.0)) + data["IQR"])/2.0)) +

        1.000000*np.tanh((data["sp.ent"] / ((((((0.319588 + data["meanfun"])/2.0) * data["maxdom"]) / 2.0) * (-(data["sp.ent"]))) - (0.319588 + data["centroid"])))) +

        0.887000*np.tanh(((((2.0 * (np.tanh(((data["centroid"] / 2.0) * data["Q75"])) + data["sfm"])) / 2.0) * (-(data["Q75"]))) + (data["Q25"] * data["Q75"]))) +

        1.000000*np.tanh((data["mode"] * (data["mode"] / 0.058823))) +

        1.000000*np.tanh(((((np.tanh(((4.91747379302978516) * data["dfrange"])) * np.tanh(((2.814810 + (np.tanh(0.319588) - (2.52346229553222656)))/2.0))) / 2.0) * 2.0) * 2.0)) +

        1.000000*np.tanh((data["sp.ent"] / ((4.047620 * (2.814810 * 2.0)) / data["meanfreq"]))) +

        0.845000*np.tanh((0.583333 * (data["centroid"] / np.tanh(np.tanh((0.061728 / (data["mode"] * (((30.0 - 0.160000) * 2.0) * 2.0)))))))) +

        1.000000*np.tanh(np.tanh((1.411760 / (-(((((((1.33079445362091064) * 2.0) + 1.411760) + (1.33079445362091064)) / (30.0 - (1.33079445362091064))) + ((data["sd"] + (1.33079445362091064)) * 2.0))))))) +

        1.000000*np.tanh((((data["modindx"] + data["skew"]) + data["skew"])/2.0)) +

        0.695000*np.tanh(((data["Q75"] * (-(((data["median"] + (((-((((data["median"] * np.tanh((data["Q75"] * 2.0))) + np.tanh(1.570796))/2.0))) + 1.411760)/2.0))/2.0)))) - data["sp.ent"])) +

        0.938000*np.tanh(((data["sp.ent"] / 2.0) * 2.0)) +

        0.501000*np.tanh((data["sp.ent"] / ((data["sfm"] + ((data["sp.ent"] + ((data["mode"] + np.tanh((0.583333 / (((data["sfm"] + data["mode"])/2.0) + data["sfm"]))))/2.0))/2.0))/2.0))) +

        1.000000*np.tanh(((((0.160000 * 0.636620) * (((data["meandom"] * (np.tanh(data["dfrange"]) * 2.0)) / 2.0) / 2.0)) / 2.0) * (data["skew"] * 0.689655))))

    return Outputs(p)





def GPClusterTwo(data):

    p = (1.000000*np.tanh(((((data["Q75"] + ((1.411760 / ((0.636620 / data["sp.ent"]) - (1.411760 * 2.0))) * (0.636620 + 3.141593))) * 2.0) * (3.76630878448486328)) * 2.0)) +

        1.000000*np.tanh(((data["median"] + (-((-((-((data["sfm"] * 30.0)))))))) + (((data["median"] + ((30.0 * data["sp.ent"]) * 2.0))/2.0) + data["sp.ent"]))) +

        1.000000*np.tanh((((np.tanh((data["minfun"] - ((((data["centroid"] * 2.0) * 2.0) + (-(((data["centroid"] * 2.0) * 3.141593))))/2.0))) * (0.275862 - 30.0)) * 2.0) / 2.0)) +

        1.000000*np.tanh((data["meanfun"] * (((((((data["meanfun"] * ((8.18931198120117188) + (((8.18931198120117188) * 2.0) - (8.18931198120117188)))) * 2.0) * 2.0) - (data["sd"] - (8.18930816650390625))) * 2.0) * 2.0) * 2.0))) +

        1.000000*np.tanh((data["Q25"] + (-((((0.160000 + ((((0.160000 + (data["sfm"] * 2.0)) - data["sfm"]) + (0.689655 + (data["sfm"] * 2.0)))/2.0))/2.0) - data["sp.ent"]))))) +

        1.000000*np.tanh(((((9.0) + 0.058823)/2.0) * ((((0.318310 * 2.0) / (((-(data["mode"])) + (0.058823 / (-(data["mode"])))) + data["mode"])) / 2.0) + data["sp.ent"]))) +

        1.000000*np.tanh(((data["maxfun"] + ((data["maxfun"] + ((data["sp.ent"] + (data["maxfun"] * 2.0)) * 2.0)) + np.tanh((np.tanh(data["Q75"]) + np.tanh((data["maxfun"] * 2.0)))))) * 2.0)) +

        1.000000*np.tanh((((-((data["sd"] - (1.411760 - ((-(np.tanh((data["sd"] - 2.814810)))) * (-(data["sd"]))))))) / 2.0) / 2.0)) +

        1.000000*np.tanh((data["sfm"] * (-(((30.0 + ((((-(((0.319588 + ((0.319588 + (0.319588 + data["meanfreq"]))/2.0))/2.0))) / 2.0) + 30.0) + 0.319588))/2.0))))) +

        0.394000*np.tanh(((-(data["skew"])) * ((data["centroid"] / 2.0) + (((data["centroid"] * data["skew"]) + (0.636620 + ((-(data["skew"])) * data["maxfun"])))/2.0)))) +

        1.000000*np.tanh((data["mode"] * ((-((((-(data["dfrange"])) + 30.0)/2.0))) - (0.0 * ((-((((30.0 * 1.0) + 30.0)/2.0))) / data["mode"]))))) +

        1.000000*np.tanh(np.tanh((data["sp.ent"] * (4.333330 - (((data["dfrange"] * (data["skew"] * 2.0)) * (((0.0 / 2.0) + (-(1.428570)))/2.0)) / 3.141593))))) +

        1.000000*np.tanh((data["kurt"] / ((0.808081 + ((data["meanfun"] * (-((np.tanh(0.583333) * 2.0)))) + ((data["maxfun"] / 2.0) * (data["modindx"] / np.tanh(data["meanfun"])))))/2.0))) +

        1.000000*np.tanh(((data["Q75"] + (np.tanh(np.tanh((data["Q75"] * np.tanh(((-((data["Q75"] / 2.0))) * data["Q75"]))))) * (np.tanh((0.0 / data["mode"])) / 2.0)))/2.0)) +

        0.817000*np.tanh((((data["mode"] * (((0.160000 + ((-((data["modindx"] / 2.0))) * 2.0))/2.0) * (data["sp.ent"] / 2.0))) + (data["modindx"] / data["sd"]))/2.0)) +

        1.000000*np.tanh((data["mindom"] * (((0.583333 + (0.0 * (0.689655 * (data["meanfun"] - (np.tanh((np.tanh(4.409090) * data["maxfun"])) / 0.583333))))) / 2.0) / 2.0))) +

        1.000000*np.tanh((-((data["skew"] - (-((4.333330 * (data["centroid"] * ((4.409090 + ((data["dfrange"] / (np.tanh(1.570796) * 0.318310)) * 2.0))/2.0))))))))) +

        1.000000*np.tanh((data["sp.ent"] / (((0.636620 / 2.0) + (0.583333 * ((((data["sp.ent"] + data["sp.ent"])/2.0) + np.tanh(((data["median"] + (data["dfrange"] / data["meanfun"]))/2.0)))/2.0)))/2.0))) +

        1.000000*np.tanh(((data["median"] + ((0.061728 * 0.636620) * ((data["meanfreq"] + ((0.061728 * data["median"]) / (((0.061728 / data["maxdom"]) * (11.56621742248535156)) * 2.0)))/2.0)))/2.0)) +

        1.000000*np.tanh((((((-(data["IQR"])) / 2.0) + np.tanh(data["Q75"])) + ((-(data["IQR"])) + (-(data["IQR"])))) + (data["Q75"] * (data["maxdom"] / 0.160000)))) +

        1.000000*np.tanh(((-(((data["minfun"] + data["mode"]) * (3.141593 + (3.141593 + (data["minfun"] * (data["mode"] * (data["maxdom"] * (data["maxdom"] / 2.0))))))))) * 2.0)) +

        1.000000*np.tanh(((((data["centroid"] - data["Q25"]) * np.tanh(data["sp.ent"])) / 2.0) * ((((0.0 * 2.0) * data["dfrange"]) / ((data["median"] + (3.141593 / 2.0))/2.0)) / data["Q75"]))) +

        1.000000*np.tanh((((0.642857 / ((data["dfrange"] - (data["Q25"] * 2.0)) + ((data["Q25"] * 2.0) - 1.0))) / data["sp.ent"]) * ((data["maxdom"] / 2.0) / np.tanh(data["kurt"])))) +

        1.000000*np.tanh(((((0.117647 + data["Q75"]) + ((data["Q25"] + (((data["Q25"] / 2.0) + (data["Q25"] - data["Q25"]))/2.0))/2.0))/2.0) - ((data["skew"] - data["sp.ent"]) * 2.0))) +

        1.000000*np.tanh((data["kurt"] - (0.061728 - (data["sfm"] * (data["dfrange"] - (1.0 - (data["dfrange"] / (((data["centroid"] * 0.061728) * (0.43061622977256775)) * 2.0)))))))) +

        1.000000*np.tanh((data["sp.ent"] + (((data["kurt"] * (data["Q25"] / 0.061728)) + (30.0 / (0.058823 - (2.869570 / (data["maxdom"] / 2.869570)))))/2.0))) +

        1.000000*np.tanh((0.583333 * ((((data["sp.ent"] * (((data["sp.ent"] + data["meandom"]) - (data["meanfreq"] / 2.0)) / 2.0)) + data["sp.ent"]) + data["sp.ent"]) + data["Q75"]))) +

        0.777000*np.tanh((data["sp.ent"] - ((0.160000 + ((((np.tanh((data["dfrange"] * 2.0)) + data["dfrange"])/2.0) - data["dfrange"]) * (data["dfrange"] * (data["dfrange"] / (data["sp.ent"] / 2.0)))))/2.0))) +

        1.000000*np.tanh((data["centroid"] * ((-(2.814810)) - ((data["centroid"] - data["modindx"]) + (data["kurt"] - ((data["mode"] / 2.0) + (data["modindx"] / data["median"]))))))) +

        1.000000*np.tanh(((data["meandom"] * (data["dfrange"] * (0.636620 / np.tanh(0.058823)))) + (np.tanh(data["maxfun"]) + ((((data["mode"] / 2.0) / 2.0) + (0.0 / 0.117647))/2.0)))) +

        1.000000*np.tanh(((data["kurt"] - np.tanh((data["minfun"] * (2.869570 - (((((data["minfun"] * data["modindx"]) / data["kurt"]) / (data["kurt"] * 2.0)) / data["kurt"]) * 2.0))))) * 2.0)) +

        1.000000*np.tanh(((0.0 + ((0.160000 / 2.0) * (data["dfrange"] + (1.428570 * ((((-(data["centroid"])) * 2.0) + data["dfrange"])/2.0))))) * (data["dfrange"] / data["mode"]))) +

        0.875000*np.tanh(((np.tanh(data["modindx"]) * (np.tanh(data["sp.ent"]) / np.tanh((data["modindx"] - (data["sp.ent"] - ((data["modindx"] * data["sp.ent"]) / 2.0)))))) / ((0.916667 * 2.0) * 2.0))) +

        0.984000*np.tanh(((data["sp.ent"] + ((0.275862 * (0.808081 / ((1.72122752666473389) * 2.0))) * (((30.0 * data["minfun"]) * (np.tanh(data["modindx"]) / 0.058823)) * 2.0))) * 2.0)) +

        1.000000*np.tanh((-((data["mode"] * ((0.0 - (data["sd"] * 2.0)) - np.tanh((1.0 * (data["sd"] * (data["skew"] + (np.tanh(0.0) / data["skew"])))))))))) +

        0.858000*np.tanh((1.428570 * ((data["mode"] - (data["modindx"] / data["meanfun"])) * ((np.tanh(0.318310) * ((data["modindx"] * 0.061728) / np.tanh(data["centroid"]))) / data["centroid"])))) +

        1.000000*np.tanh((np.tanh((data["median"] / (((2.814810 + (-((data["dfrange"] * np.tanh((data["dfrange"] * (np.tanh(np.tanh(data["Q75"])) / np.tanh(0.160000))))))))/2.0) * 2.0))) / 2.0)) +

        1.000000*np.tanh(((data["sp.ent"] - data["skew"]) - (0.0 * (((data["skew"] * 0.916667) + 1.0) - ((0.0 * 2.0) + (0.583333 / data["meanfun"])))))) +

        1.000000*np.tanh(((((data["Q25"] + ((0.117647 / 2.0) * ((data["mindom"] / (data["sp.ent"] / data["kurt"])) - (data["sd"] * 2.0))))/2.0) + (data["Q75"] * (data["dfrange"] / data["Q25"])))/2.0)) +

        0.887000*np.tanh((data["meanfun"] / (0.061728 + (((((data["dfrange"] / (0.061728 + 30.0)) * ((1.411760 * 2.0) * data["meandom"])) / 2.0) * data["meanfun"]) / data["meanfreq"])))) +

        1.000000*np.tanh((((data["maxdom"] + (data["maxdom"] / 4.047620))/2.0) / 2.0)) +

        1.000000*np.tanh((data["sp.ent"] + (data["centroid"] * ((4.047620 / (data["sd"] / 2.0)) - (((1.428570 / 2.0) + 4.409090) + (4.047620 - (data["Q75"] / data["centroid"]))))))) +

        1.000000*np.tanh((((data["maxdom"] + (data["sp.ent"] * 2.0))/2.0) * 2.0)) +

        0.845000*np.tanh((((data["sfm"] * 2.0) + (data["kurt"] + (-((-((np.tanh(data["sp.ent"]) * (data["meandom"] / 2.0)))))))) + 0.0)) +

        1.000000*np.tanh((((-((data["sfm"] * 2.0))) + (((data["sp.ent"] * (((data["dfrange"] + (-(data["sfm"])))/2.0) * 2.0)) + (data["sp.ent"] / 2.0))/2.0))/2.0)) +

        1.000000*np.tanh((-((((((6.0) + (np.tanh((((6.0) + data["sfm"]) * data["sfm"])) + data["sfm"])) * data["sfm"]) + data["minfun"]) - data["sfm"])))) +

        0.695000*np.tanh(((2.869570 + (data["mindom"] + (2.869570 / ((2.869570 + np.tanh(np.tanh((data["sp.ent"] + ((data["dfrange"] + data["sp.ent"])/2.0)))))/2.0)))) * data["sp.ent"])) +

        0.938000*np.tanh(((data["median"] / 0.642857) * ((-(data["mode"])) + ((data["sfm"] * 2.0) + (((((-(data["mode"])) * 2.0) * 2.0) * 2.0) + data["skew"]))))) +

        0.501000*np.tanh((0.061728 * (data["median"] - (9.0)))) +

        1.000000*np.tanh(((((np.tanh(((data["meandom"] / 2.0) / 2.0)) + (((-(data["mode"])) + np.tanh((0.160000 / (((2.869570 * 1.0) + 0.117647)/2.0))))/2.0))/2.0) + data["Q75"])/2.0)))

    return Outputs(p)
colors = {0:'red',1:'green'}

plt.figure(figsize=(12,12))

_ = plt.scatter(GPClusterOne(train),GPClusterTwo(train),color=[colors[x] for x in train.label])
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC

x = pd.DataFrame()

x['cl1'] = GPClusterOne(train)

x['cl2'] = GPClusterTwo(train)

x['label'] = train.label

svc = SVC(kernel='linear',probability=True)

svc.fit(x[x.columns[:-1]],x.label)

predictions = svc.predict_proba(x[x.columns[:-1]])[:,1]

print(accuracy_score(x.label,(predictions>.5)))