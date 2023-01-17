!pip install adtk ;
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from adtk.detector import GeneralizedESDTestAD

from adtk.visualization import plot

from sklearn.metrics import f1_score

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')

df = df.iloc[0:500000]

df.time = [np.datetime64('2020-05-12') + pd.Timedelta(x, unit='ms') for x in df.time.values]

df.index = df.time
from adtk.detector import GeneralizedESDTestAD

esd_ad = GeneralizedESDTestAD(alpha=75)

anomalies = esd_ad.fit_detect(df.signal)
plot(df.signal, anomaly=anomalies, ts_linewidth=1, ts_markersize=3, anomaly_markersize=5, anomaly_color='red', anomaly_tag="marker");
f1_score(df.open_channels, anomalies, average='macro')
def change(val, b, r):

    t = val.copy()

    for i in range(0,len(t),b):

        if np.sum(t[i:i+b])==b-1:

            t[i:i+b]=1

    for i in range(0,len(t),r//2):

        if np.sum(t[i:i+r])<2:

            t[i:i+r]=0

    return t
f1_score(df.open_channels, change(anomalies.values,5,250), average='macro')