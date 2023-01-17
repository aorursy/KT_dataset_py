import numpy as np  # linear algebra

import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from math import log1p

from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic

import os



DATA_DIR = '../input/'
def combine_data(traffic_df):

    name=np.asarray(traffic_df['小区CGI'])

    # traffic=np.asarray(traffic_df[['时间','空口业务总字节数(KByte)']])

    traffic = np.asarray(traffic_df['空口业务总字节数(KByte)'])

    assert name.shape[0] == traffic.shape[0]

    name_list = []



    traffic_list = []

    temp = 0

    for i in range(1, traffic.shape[0]):

        if name[i] == name[i-1]:

            pass

        else:

            name_list.append(name[i-1])

            traffic_list.append(np.log1p(traffic[temp:i]))

            temp=i

    name_list.append(name[i])

    traffic_list.append(np.log1p(traffic[temp:i+1]))

    # traffic = np.asarray(traffic_list)



    name_list = np.asarray(name_list, dtype=str)



    return name_list, traffic_list
def clean_nan(traffic_list):

    for i in range(len(traffic_list)):

        temp=traffic_list[i]

        for j in range(len(temp)):

            if temp[j]==0:

                mean_list=fullwithneigh(temp,j)

                res=sum(mean_list)/len(mean_list)

                temp[j]=res

        traffic_list[i]=temp

    return traffic_list

def fullwithneigh(temp,j,n=7):

    mean_list=[]

    for k in range(1,1+n):

        if j-k*24>=0 and temp[j-k*24]>0:

            mean_list.append(temp[j-k*24])

        if j-k*24<0:

            break

    for k in range(1,1+n):

        if j+k*24<len(temp) and temp[j+k*24]>0:

            mean_list.append(temp[j+k*24])

        if j+k*24>=len(temp):

            break

    if len(mean_list)==0:

        mean_list=fullwithneigh(temp,j,n+7)

    

    return mean_list

                
def plot_waveform(name_list, traffic_list):

    for i in range(len(traffic_list)):

        p_value=test_stationarity(traffic_list[i])

#         if p_value<0.01:

#             continue

        pyplot.title(name_list[i]+'waveform')

        pyplot.plot(traffic_list[i])

        pyplot.savefig(name_list[i]+'waveform.jpg')#保存图片

        pyplot.show()



    # for i in range(len(traffic_list)):

        pyplot.title(name_list[i]+'distribution')

        sns.distplot(traffic_list[i])

        pyplot.show()

    # for i in range(len(traffic_list)):

        plot_acf(traffic_list[i], lags=len(traffic_list[i])//20+1)

        pyplot.show()

        

def test_stationarity(timeseries):

    # Perform Dickey-Fuller test:

    

    dftest = adfuller(timeseries, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():

        dfoutput['Critical Value (%s)' % key] = value



    print('Results of Dickey-Fuller Test:')

    print(dfoutput)

    return dftest[1]
def load_data():

#     file_name='hangzhou_base_station.csv'

    file_name='0506-0620.csv'

    traffic_df=pd.read_csv(DATA_DIR+file_name,encoding='utf-8')

    name_list,traffic=combine_data(traffic_df)

    return name_list,traffic
name_list,traffic=load_data()

traffic=clean_nan(traffic)

plot_waveform(name_list, traffic)