import datetime as dt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
!pip install changefinder
import changefinder
import matplotlib.pyplot as plt
from matplotlib import pyplot
# Any results you write to the current directory are saved as output.
for dirname, _, filenames in os.walk('/kaggle/input/nab/realTraffic/realTraffic/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# -*- coding: utf-
import matplotlib.pyplot as plt
import changefinder


def changefinderTS(data, r, order, smooth):
    cf = changefinder.ChangeFinder(r=r, order=order, smooth=smooth)

    ret = []
    for i in data:
        score = cf.update(i)
        ret.append(score)
    return ret
def plotTSScore(data, ret):
    fig = plt.figure(figsize=(17, 8))
    ax = fig.add_subplot(111)
    ax.plot(ret, label='anomaly score', color = 'red', alpha=0.9)
    ax2 = ax.twinx()
    ax2.plot(data, label='real', color = 'blue', alpha=0.9)
    plt.title('Anomaly Score time series')
    plt.ylabel('anomaly score')
    plt.xlabel('data count')
    plt.legend()
    plt.show()


# generic data example
data=np.concatenate([np.random.normal(0.7, 0.05, 300),
np.random.normal(1.5, 0.05, 300),
np.random.normal(0.6, 0.05, 300),
np.random.normal(1.3, 0.05, 300)])

ret = changefinderTS(data, r=0.03, order=1, smooth=5)

plotTSScore(data, ret)
x = np.arange(200)
normal_data = np.sin(x*0.5).reshape(200, -1)
abnormal_data = (np.sin(x*0.5)**2).reshape(200, -1)

whole_data = np.vstack((normal_data, abnormal_data))
whole_data = np.vstack((whole_data, normal_data))

ret_data2 = changefinderTS(whole_data, r=0.03, order=1, smooth=5)

plotTSScore(whole_data, ret_data2)
fpath = "../input/nab/realAWSCloudwatch/realAWSCloudwatch//"
fname = "grok_asg_anomaly.csv"

fullPath = fpath + fname

def parser(x):
	return dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
 
data_ = pd.read_csv(fullPath, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
data_.head()
ret = changefinderTS(data_.values, r=0.01, order=1, smooth=3)

plotTSScore(data_.values, ret)

