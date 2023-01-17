import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import log1p

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import time,pickle

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

DATA_DIR = '../input/'

# Any results you write to the current directory are saved as output.
def plot_bs(x,name):

    sns.set(style="white", palette="muted", color_codes=True)     #set( )设置主题，调色板更常用

    pyplot.plot(x)

    #

#     pyplot.savefig(name+'.png',dpi=520)

    pyplot.show()

    plot_acf(x, lags=50)

#     pyplot.savefig(name + 'acf.png',dpi=520)

    pyplot.show()

#     plot_pacf(x, lags=50)

#     # pyplot.show()

#     pyplot.savefig(name + 'pacf.png',dpi=520)

#     pyplot.show()
def test_stationarity(timeseries):

    # Perform Dickey-Fuller test:

    print('Results of Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:2], index=['Test Statistic', 'p-value'])



    print(dfoutput)
def run_main(x,filename):

    sns.set(style="white", palette="muted", color_codes=True)



    plot_bs(x,filename+'timeSorted')

    test_stationarity(x)
fr = open(DATA_DIR+'result.pickle', 'rb')

result = pickle.load(fr)

result=np.log1p(result)

#流量大小图例

for index in range(12):

    run_main(result[:,index,-1],str(index))

#总连接数

for index in range(12):

    run_main(result[:,index,-2],str(index))
category = ["P2P", "VoIP",  "财经", "导航", '股票', "即时通信", "其他", "社交网络", '视频', "网页浏览",

            "文件传输", "邮件", '游戏']



# 子类流量大小

for class_index in range(13):

    print('category traffic:',category[class_index])

    for index in range(3):

        run_main(result[:,index,class_index],str(index))
# 子类连接数

for class_index in range(13):

    print('category connect number:',category[class_index])

    for index in range(3):

        run_main(result[:,index,class_index+13],str(index))