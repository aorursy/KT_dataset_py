import pandas as pd

import numpy as np

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

    p = (0.406000*np.tanh((((data["sd"] + (3.141593 * np.tanh(data["dfrange"]))) - ((data["sd"] - ((3.141593 + 0.318310) * 0.318310)) * 0.318310)) * 0.318310)) +

        1.000000*np.tanh((-((data["Q25"] - np.tanh((data["Q75"] * (data["sd"] - (3.141593 + ((10.0) - (data["meanfreq"] * (data["dfrange"] - data["Q75"]))))))))))) +

        1.000000*np.tanh(((2.0 * data["sfm"]) - (data["sp.ent"] * ((3.141593 + (data["sp.ent"] + (3.77971982955932617))) + (-((data["sp.ent"] * ((3.77971982955932617) + data["sp.ent"])))))))) +

        0.940000*np.tanh(np.tanh((0.318310 * (data["sd"] * np.tanh((0.318310 + (0.318310 * np.tanh(((0.318310 * 0.318310) * data["sd"]))))))))) +

        1.000000*np.tanh((data["skew"] - (((data["Q75"] + data["kurt"]) + (data["kurt"] + data["kurt"])) + (((data["kurt"] - (data["Q75"] * data["kurt"])) + data["kurt"]) - data["kurt"])))) +

        1.000000*np.tanh((data["sp.ent"] * (-((data["sp.ent"] * (-((data["meanfreq"] * np.tanh(((data["modindx"] - data["meanfreq"]) + (-((data["meanfreq"] * data["sp.ent"]))))))))))))) +

        1.000000*np.tanh(((-(data["sfm"])) - ((np.tanh(np.tanh(data["median"])) - data["median"]) * (-((data["sfm"] - (-(((0.318310 + data["IQR"]) + data["mode"]))))))))) +

        1.000000*np.tanh(((((data["sp.ent"] + (data["meanfun"] + data["meanfun"])) * data["median"]) + data["meanfun"]) + np.tanh(((data["sp.ent"] * data["median"]) - data["sp.ent"])))) +

        1.000000*np.tanh((((data["meandom"] * (-(data["meanfreq"]))) * data["meandom"]) * (((data["meandom"] + data["meanfreq"]) + data["meandom"]) + (data["meandom"] * (-(data["meandom"])))))) +

        1.000000*np.tanh((data["meanfun"] + ((0.318310 - ((data["median"] + data["sfm"]) * 0.318310)) * (0.318310 - (-((0.318310 * (-(0.318310))))))))) +

        1.000000*np.tanh((((data["IQR"] - (data["Q75"] - ((((data["Q75"] - data["Q75"]) - data["Q75"]) * data["Q75"]) * 1.570796))) * 2.0) * (data["Q75"] - data["skew"]))) +

        1.000000*np.tanh((-((data["sp.ent"] + (np.tanh((data["IQR"] + data["Q25"])) * np.tanh(np.tanh((0.318310 - (data["Q75"] + (data["Q75"] + (data["Q75"] + data["Q25"]))))))))))) +

        1.000000*np.tanh((((data["meandom"] * data["sd"]) - np.tanh((np.tanh(np.tanh((data["mode"] - (data["meandom"] * (data["sd"] - (8.0)))))) + data["sfm"]))) - data["minfun"])) +

        1.000000*np.tanh(((data["Q75"] + (np.tanh(data["Q75"]) + data["Q25"])) * (data["IQR"] + np.tanh((-(((data["Q75"] + data["IQR"]) * (data["Q75"] + data["Q75"])))))))) +

        1.000000*np.tanh(((data["Q75"] + data["Q75"]) * (0.318310 + (data["Q75"] * ((0.318310 * data["meanfun"]) - (data["Q75"] + data["Q75"])))))) +

        1.000000*np.tanh((-((data["skew"] * (-(((-(np.tanh(np.tanh(data["sd"])))) - ((np.tanh((np.tanh(data["sd"]) + data["sd"])) * data["skew"]) - data["skew"])))))))) +

        1.000000*np.tanh(((data["maxfun"] + data["meanfun"]) + (data["meanfun"] * np.tanh((data["sd"] + ((data["meanfun"] * data["maxfun"]) * data["meanfun"])))))) +

        0.981000*np.tanh(((-(data["Q75"])) * np.tanh((((data["sd"] - np.tanh((data["sd"] - (-(data["Q75"]))))) - np.tanh(data["Q75"])) * np.tanh(np.tanh(0.318310)))))) +

        1.000000*np.tanh(((data["skew"] + (data["sd"] * (data["mode"] + data["dfrange"]))) * np.tanh(data["Q75"]))) +

        1.000000*np.tanh(((data["IQR"] + (data["IQR"] + data["Q25"])) * (data["kurt"] + ((data["IQR"] + 0.0) + data["kurt"])))) +

        1.000000*np.tanh(np.tanh((data["mode"] * (-((data["meanfun"] + (data["mode"] * (data["mode"] + ((9.38472557067871094) + ((9.38472557067871094) + data["sd"])))))))))) +

        1.000000*np.tanh(((data["Q75"] - data["kurt"]) * ((data["kurt"] - data["Q75"]) * (data["Q75"] - (-(((data["Q75"] - (-(data["Q75"]))) + data["Q75"]))))))) +

        1.000000*np.tanh((-((data["kurt"] * ((data["Q25"] * ((1.570796 - data["kurt"]) - (data["kurt"] - data["Q25"]))) - data["kurt"]))))) +

        1.000000*np.tanh(((data["meanfun"] - data["centroid"]) * ((data["meanfreq"] * data["centroid"]) + (np.tanh((-((data["meanfun"] - data["centroid"])))) - (np.tanh(data["meanfreq"]) + data["meanfun"]))))) +

        1.000000*np.tanh((-(np.tanh((data["sp.ent"] * (0.318310 - np.tanh((np.tanh((-(np.tanh(data["sp.ent"])))) * data["sp.ent"])))))))) +

        1.000000*np.tanh((data["kurt"] * (data["skew"] + ((data["IQR"] + np.tanh(((data["IQR"] + (data["skew"] + data["IQR"])) + data["IQR"]))) + np.tanh(data["IQR"]))))) +

        1.000000*np.tanh((data["Q75"] * ((data["Q75"] * (-(data["Q75"]))) + ((data["sp.ent"] * data["sd"]) - (0.636620 + ((data["minfun"] * (12.81465339660644531)) - np.tanh(2.0))))))) +

        1.000000*np.tanh((-((-((np.tanh(((data["maxdom"] - (-(data["sp.ent"]))) * 2.0)) * data["sp.ent"])))))) +

        0.924000*np.tanh((-((data["sp.ent"] * np.tanh((((data["median"] + data["sp.ent"]) + data["modindx"]) + (((0.318310 + (data["sp.ent"] - data["sp.ent"])) + data["median"]) + data["median"]))))))) +

        1.000000*np.tanh((-((data["meanfun"] * (((data["median"] + data["meanfun"]) + data["median"]) - (data["meanfun"] * (data["meanfun"] * ((-(data["median"])) - (data["median"] - data["kurt"]))))))))) +

        1.000000*np.tanh(((data["Q75"] * np.tanh(data["Q75"])) * (-((np.tanh(data["Q75"]) - ((-(np.tanh(data["sd"]))) + ((data["Q75"] * data["Q75"]) * data["Q75"]))))))) +

        1.000000*np.tanh(((data["sp.ent"] * (-((-(data["modindx"]))))) * (((data["sp.ent"] - (-(data["skew"]))) * (-(data["meandom"]))) - 0.0))) +

        1.000000*np.tanh((-(((data["minfun"] * (data["minfun"] - (data["maxdom"] * ((data["minfun"] + (-((data["minfun"] * data["meanfreq"])))) * data["minfun"])))) * (data["meanfreq"] * 1.570796))))) +

        1.000000*np.tanh((data["minfun"] * ((data["maxfun"] - data["median"]) - (data["minfun"] * (((data["sp.ent"] - (data["sp.ent"] * data["dfrange"])) - data["sp.ent"]) + data["meanfun"]))))) +

        0.218000*np.tanh((data["sp.ent"] * np.tanh(data["Q25"]))) +

        1.000000*np.tanh((data["sfm"] * np.tanh((data["median"] - np.tanh((data["sfm"] * data["sfm"])))))) +

        0.839000*np.tanh(((data["maxdom"] - (data["sd"] + (0.0 * (data["Q25"] - (1.0 - (data["meandom"] + np.tanh((data["sd"] + 1.0)))))))) * data["meandom"])) +

        0.838000*np.tanh((data["centroid"] * (((data["meanfun"] + data["sfm"]) + (-(data["Q75"]))) + ((3.141593 * data["mindom"]) + data["sfm"])))) +

        0.980000*np.tanh((data["mode"] * (data["maxfun"] - (data["sfm"] + (data["Q75"] - data["dfrange"]))))) +

        1.000000*np.tanh((np.tanh(data["centroid"]) * (((data["Q25"] - data["sfm"]) - ((-(data["Q25"])) - data["sp.ent"])) - (-(data["sp.ent"]))))) +

        0.811000*np.tanh((data["kurt"] * np.tanh(np.tanh((-((data["sd"] + np.tanh(np.tanh((-(data["meanfreq"]))))))))))) +

        1.000000*np.tanh(((data["dfrange"] * np.tanh(0.636620)) * (data["median"] + data["IQR"]))) +

        1.000000*np.tanh(np.tanh((data["skew"] * (data["minfun"] - (((-((np.tanh(data["kurt"]) - data["skew"]))) - data["minfun"]) - data["kurt"]))))) +

        1.000000*np.tanh((data["mode"] * (data["Q25"] + (data["Q25"] + (data["meanfun"] * (data["meanfreq"] - (data["meandom"] * ((data["meanfun"] * (data["meanfun"] + data["mode"])) - data["meanfreq"])))))))) +

        1.000000*np.tanh((data["sp.ent"] * (-((data["Q25"] + np.tanh((data["Q75"] + ((-((0.318310 - (data["meanfreq"] * data["Q25"])))) + (data["sp.ent"] * np.tanh(0.318310)))))))))) +

        1.000000*np.tanh((data["centroid"] * (data["Q25"] * ((((2.0 + data["Q25"]) + 1.0) + data["Q25"]) + (data["Q25"] + (1.0 + data["Q25"])))))) +

        1.000000*np.tanh((data["mode"] * ((data["minfun"] * data["maxfun"]) * (-(data["minfun"]))))) +

        1.000000*np.tanh((-((data["centroid"] * ((data["kurt"] + data["skew"]) * ((data["sd"] + (6.0)) + (data["sd"] + ((data["maxfun"] + (7.0)) + data["maxfun"])))))))) +

        0.939000*np.tanh((data["Q75"] * np.tanh((((data["mode"] * data["sd"]) - data["mode"]) + (data["skew"] - (((3.58607149124145508) * data["sfm"]) - (data["meanfun"] * data["sd"]))))))) +

        1.000000*np.tanh((data["sp.ent"] * np.tanh(np.tanh((data["Q75"] * ((((data["sp.ent"] * (-(data["meanfreq"]))) * (-(data["meanfreq"]))) * (-(data["meanfreq"]))) * (-(data["meanfreq"])))))))))

    return Outputs(p)





def GPClusterTwo(data):

    p = (0.406000*np.tanh(((data["median"] + ((data["dfrange"] - data["skew"]) + (((6.03158855438232422) * (-(np.tanh(np.tanh(data["IQR"]))))) + ((6.03158855438232422) * data["mode"])))) - data["skew"])) +

        1.000000*np.tanh((data["Q75"] - (-((np.tanh(data["Q75"]) - (2.0 * (-(((((-(data["kurt"])) + 2.0) * data["mindom"]) - data["kurt"]))))))))) +

        1.000000*np.tanh((((-(3.141593)) * np.tanh((data["IQR"] + (data["skew"] - ((data["sp.ent"] * 3.141593) + data["mode"]))))) - data["skew"])) +

        0.940000*np.tanh((-((0.0 - ((0.318310 * data["sd"]) * np.tanh(np.tanh(0.318310))))))) +

        1.000000*np.tanh((0.318310 * np.tanh((data["median"] + np.tanh(np.tanh((data["sfm"] * ((-((10.0))) + (-((10.0))))))))))) +

        1.000000*np.tanh((data["meanfun"] * ((6.93633317947387695) - (((-(data["sd"])) + data["meanfun"]) + ((6.93633317947387695) * np.tanh(data["meanfun"])))))) +

        1.000000*np.tanh(((-(np.tanh(data["maxfun"]))) - ((data["sfm"] + np.tanh(data["sfm"])) + (data["maxfun"] * (2.81130266189575195))))) +

        1.000000*np.tanh((((data["centroid"] * (data["kurt"] - data["meanfreq"])) * data["centroid"]) + (data["mode"] + (-((data["kurt"] - ((data["meanfun"] * data["centroid"]) * data["centroid"]))))))) +

        1.000000*np.tanh((-(((data["skew"] * ((14.34124279022216797) - (-((14.34124279022216797))))) * (-((((-((14.34124279022216797))) * data["dfrange"]) - data["kurt"]))))))) +

        1.000000*np.tanh((data["meandom"] * ((10.52644634246826172) + ((data["centroid"] + data["meandom"]) - (data["centroid"] + np.tanh(np.tanh((data["centroid"] * (14.61073684692382812))))))))) +

        1.000000*np.tanh((data["IQR"] * ((data["Q25"] + np.tanh((data["Q75"] * (6.0)))) + (data["Q25"] + data["IQR"])))) +

        1.000000*np.tanh((np.tanh(((-(np.tanh(data["centroid"]))) - (data["modindx"] - data["mode"]))) - ((-((0.0))) * ((-((data["maxfun"] + (0.0)))) - data["centroid"])))) +

        1.000000*np.tanh((data["Q75"] * (data["mode"] - np.tanh(((data["sd"] * data["sfm"]) - (data["mode"] - (data["sfm"] + (data["maxdom"] + (data["sd"] * data["sfm"]))))))))) +

        1.000000*np.tanh((data["skew"] * (-(((data["skew"] + (-(((data["mindom"] - np.tanh(data["sd"])) - data["skew"])))) + ((data["kurt"] * data["sd"]) - data["mindom"])))))) +

        1.000000*np.tanh(((((9.0) - (np.tanh(data["median"]) * data["median"])) + data["meanfun"]) * np.tanh((np.tanh(data["meanfun"]) * (np.tanh(data["Q75"]) + np.tanh(data["median"])))))) +

        1.000000*np.tanh(((data["meanfun"] - data["centroid"]) - (np.tanh(data["sfm"]) + ((data["meanfun"] + (data["meanfun"] + (data["meanfun"] * data["meanfun"]))) * (data["meanfun"] + np.tanh(data["sfm"])))))) +

        1.000000*np.tanh((data["mode"] * (np.tanh(data["meanfreq"]) + (3.141593 * np.tanh(((3.141593 * np.tanh((data["median"] + (data["mode"] * np.tanh(data["kurt"]))))) - data["median"])))))) +

        0.981000*np.tanh((data["mode"] * (data["Q75"] - ((-((np.tanh(0.318310) * data["Q75"]))) + (((data["Q75"] - data["centroid"]) - data["kurt"]) * (data["sd"] - 0.0)))))) +

        1.000000*np.tanh((data["meanfun"] + (data["IQR"] * np.tanh((np.tanh(np.tanh((data["meanfun"] + (7.0)))) - ((-(data["meanfun"])) * (data["IQR"] - ((11.97325134277343750) * (11.97325134277343750))))))))) +

        1.000000*np.tanh((data["kurt"] * ((np.tanh(((data["median"] * (-(data["sd"]))) - ((-(data["sd"])) - (3.141593 * data["sp.ent"])))) * data["median"]) + np.tanh(data["median"])))) +

        1.000000*np.tanh((-((data["maxfun"] + np.tanh(((-(data["mindom"])) * ((data["maxfun"] - (-(data["maxfun"]))) + (3.141593 + (8.83117485046386719))))))))) +

        1.000000*np.tanh(((np.tanh(np.tanh(data["Q25"])) - 0.318310) * (np.tanh((data["Q75"] * (np.tanh(data["Q25"]) - 0.318310))) - np.tanh((data["Q25"] * 0.318310))))) +

        1.000000*np.tanh((-((data["Q25"] * np.tanh(((3.58818507194519043) * ((data["maxfun"] + data["centroid"]) - (-((data["kurt"] * data["Q25"])))))))))) +

        1.000000*np.tanh((data["skew"] * (np.tanh((-((-((data["mode"] * data["meanfreq"])))))) - np.tanh((data["centroid"] * data["minfun"]))))) +

        1.000000*np.tanh((data["sp.ent"] * (((((5.0) - 0.0) * data["minfun"]) * ((5.0) - (-((5.0))))) - (((5.0) - data["sp.ent"]) * (-(data["IQR"])))))) +

        1.000000*np.tanh((-((data["centroid"] * (data["IQR"] * (((data["IQR"] - (7.93866825103759766)) - (7.93866825103759766)) + ((7.93866825103759766) * np.tanh((data["sd"] - data["centroid"]))))))))) +

        1.000000*np.tanh(((data["Q25"] * ((data["Q25"] * data["minfun"]) * data["meandom"])) * ((data["minfun"] * data["meandom"]) + data["Q25"]))) +

        1.000000*np.tanh(((data["skew"] * (data["median"] + data["kurt"])) + (data["kurt"] * (data["skew"] + (data["skew"] + (data["skew"] + np.tanh((data["median"] * data["median"])))))))) +

        0.924000*np.tanh(((data["sfm"] * (data["sfm"] - np.tanh((-((data["sp.ent"] * data["meanfreq"])))))) + (data["sfm"] * (np.tanh((data["meanfun"] * data["meanfreq"])) * 3.141593)))) +

        1.000000*np.tanh((np.tanh((data["Q75"] - data["IQR"])) * np.tanh((data["meanfun"] * ((11.94026756286621094) - (data["Q75"] - data["Q75"])))))) +

        1.000000*np.tanh(((data["meanfun"] + data["meanfun"]) * (0.0 + ((data["meanfun"] - np.tanh((data["centroid"] * (-(data["meanfreq"]))))) - np.tanh((data["centroid"] * (-(data["meanfreq"])))))))) +

        1.000000*np.tanh((data["skew"] * np.tanh(((data["meanfun"] * data["meanfreq"]) - (data["sd"] * (data["kurt"] + ((data["kurt"] + (data["meanfun"] * data["sd"])) + data["sp.ent"]))))))) +

        1.000000*np.tanh(((data["skew"] + data["kurt"]) * (data["kurt"] - (-((((6.0) * data["kurt"]) - (data["minfun"] * ((6.0) + np.tanh(((2.62265038490295410) * 3.141593)))))))))) +

        1.000000*np.tanh((((-(data["Q25"])) + ((-(data["Q75"])) + 0.0)) * (((data["median"] * (data["median"] * data["Q75"])) + data["median"]) * data["Q75"]))) +

        0.218000*np.tanh((data["Q75"] * (data["skew"] + (data["skew"] + (0.318310 * ((0.318310 - data["Q75"]) + (data["skew"] + np.tanh((data["sp.ent"] * data["meanfreq"]))))))))) +

        1.000000*np.tanh((data["sfm"] * np.tanh(((data["meanfreq"] - (data["sd"] + (np.tanh(data["kurt"]) + ((data["sd"] - np.tanh(data["sfm"])) - np.tanh(data["modindx"]))))) * data["kurt"])))) +

        0.839000*np.tanh((((data["minfun"] - data["meandom"]) - (-(((data["minfun"] - data["sp.ent"]) - data["centroid"])))) * np.tanh(((data["centroid"] - data["sd"]) + (data["median"] * 1.0))))) +

        0.838000*np.tanh((data["Q75"] * ((3.0) * (data["minfun"] + ((data["minfun"] + data["minfun"]) + ((data["minfun"] + (data["skew"] - np.tanh(data["sp.ent"]))) - data["sp.ent"])))))) +

        0.980000*np.tanh((data["sfm"] * (((data["meanfun"] + data["meanfun"]) - (np.tanh((-(((7.0) * (7.0))))) * data["maxfun"])) * (-(((7.0) * (7.0))))))) +

        1.000000*np.tanh((data["meanfun"] - ((data["sp.ent"] - (data["sfm"] - (data["Q75"] * 0.318310))) * (data["Q25"] - (0.318310 - np.tanh(data["Q25"])))))) +

        0.811000*np.tanh((data["minfun"] * (((data["IQR"] + (data["IQR"] + data["median"])) + np.tanh((data["minfun"] - data["median"]))) + np.tanh(data["median"])))) +

        1.000000*np.tanh((-((data["sfm"] * ((-((np.tanh(data["median"]) * np.tanh((-((data["sd"] + (data["median"] + (3.141593 + 0.636620))))))))) + np.tanh(data["median"])))))) +

        1.000000*np.tanh(((np.tanh(data["skew"]) * (data["skew"] * (data["kurt"] + data["skew"]))) + (data["Q25"] * ((data["sd"] * np.tanh(data["meanfun"])) + (data["kurt"] + data["skew"]))))) +

        1.000000*np.tanh((-((data["mindom"] * (-((data["sd"] - (-((data["sd"] - np.tanh(np.tanh((-((-(data["sd"])))))))))))))))) +

        1.000000*np.tanh((data["Q75"] * ((data["Q75"] * data["Q25"]) + (-(data["mode"]))))) +

        1.000000*np.tanh((data["sfm"] * ((-(np.tanh((data["sp.ent"] * ((6.84499883651733398) * (6.84499883651733398)))))) + ((-(data["sfm"])) + np.tanh((data["sfm"] * (6.84499883651733398))))))) +

        1.000000*np.tanh((-(((data["kurt"] - data["skew"]) * (3.141593 * np.tanh((data["sfm"] * (data["sd"] - ((-(data["kurt"])) - (data["kurt"] + 3.141593)))))))))) +

        1.000000*np.tanh(((data["skew"] * ((5.82858467102050781) * np.tanh(data["meandom"]))) * (0.318310 - ((5.82858467102050781) + ((5.82858467102050781) + ((data["skew"] + data["meandom"]) + (5.82858467102050781))))))) +

        0.939000*np.tanh((data["Q75"] * (data["mode"] * ((((data["mode"] + 1.570796) + 1.570796) + (1.570796 + data["modindx"])) + (data["mode"] + data["mode"]))))) +

        1.000000*np.tanh((data["minfun"] * np.tanh((data["meanfreq"] + (np.tanh(data["meanfreq"]) + (np.tanh((data["minfun"] * data["minfun"])) * (-(data["meanfreq"])))))))))

    return Outputs(p)
colors = {0:'red',1:'green'}

plt.figure(figsize=(12,12))

_ = plt.scatter(GPClusterOne(train),GPClusterTwo(train),color=[colors[x] for x in train.label])