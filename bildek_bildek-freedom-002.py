



import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

from sklearn.neural_network import MLPRegressor

from sklearn.tree import DecisionTreeRegressor



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





data24 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/AUDJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data23 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/CADJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data22 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/GBPJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data21 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/CHFJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data20 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data19 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/USDJPY_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data18 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURAUD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data17 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURNOK_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data16 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/GBPAUD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data15 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/AUDNZD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data14 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/CADCHF_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data13 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/USDNOK_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data12 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/AUDCAD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data11 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/AUDCHF_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data10 = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURCHF_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data9  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/GBPCHF_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data8  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURCAD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data7  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/GBPCAD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data6  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/NZDUSD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data5  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/USDCAD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data4  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/USDCHF_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data3  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/AUDUSD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data2  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/EURUSD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')

data1  = pd.read_csv('/kaggle/input/majorscoursesmarketingdays/GBPUSD_Candlestick_1_D_BID_01.01.1995-15.08.2020.csv')







AUDJPY_O =  data24.iloc[:,1]

CADJPY_O =  data23.iloc[:,1]

GBPJPY_O =  data22.iloc[:,1]

CHFJPY_O =  data21.iloc[:,1]

EURJPY_O =  data20.iloc[:,1]

USDJPY_O =  data19.iloc[:,1]

EURAUD_O =  data18.iloc[:,1]

EURNOK_O =  data17.iloc[:,1]

GBPAUD_O =  data16.iloc[:,1]

AUDNZD_O =  data15.iloc[:,1]

CADCHF_O =  data14.iloc[:,1]

USDNOK_O =  data13.iloc[:,1]

AUDCAD_O =  data12.iloc[:,1]

AUDCHF_O =  data11.iloc[:,1]

EURCHF_O =  data10.iloc[:,1]

GBPCHF_O =  data9 .iloc[:,1]

EURCAD_O =  data8 .iloc[:,1]

GBPCAD_O =  data7 .iloc[:,1]

NZDUSD_O =  data6 .iloc[:,1]

USDCAD_O =  data5 .iloc[:,1]

USDCHF_O =  data4 .iloc[:,1]

AUDUSD_O =  data3 .iloc[:,1]

EURUSD_O =  data2 .iloc[:,1]

GBPUSD_O =  data1 .iloc[:,1]



AUDJPY_C = data24.iloc[:,4]

CADJPY_C = data23.iloc[:,4]

GBPJPY_C = data22.iloc[:,4]

CHFJPY_C = data21.iloc[:,4]

EURJPY_C = data20.iloc[:,4]

USDJPY_C = data19.iloc[:,4]

EURAUD_C = data18.iloc[:,4]

EURNOK_C = data17.iloc[:,4]

GBPAUD_C = data16.iloc[:,4]

AUDNZD_C = data15.iloc[:,4]

CADCHF_C = data14.iloc[:,4]

USDNOK_C = data13.iloc[:,4]

AUDCAD_C = data12.iloc[:,4]

AUDCHF_C = data11.iloc[:,4]

EURCHF_C = data10.iloc[:,4]

GBPCHF_C = data9 .iloc[:,4]

EURCAD_C = data8 .iloc[:,4]

GBPCAD_C = data7 .iloc[:,4]

NZDUSD_C = data6 .iloc[:,4]

USDCAD_C = data5 .iloc[:,4]

USDCHF_C = data4 .iloc[:,4]

AUDUSD_C = data3 .iloc[:,4]

EURUSD_C = data2 .iloc[:,4]

GBPUSD_C = data1 .iloc[:,4]


a24 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, AUDJPY_C]

a23 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, CADJPY_C]

a22 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, GBPJPY_C]

a21 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, CHFJPY_C]

a20 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURJPY_C]

a19 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, USDJPY_C]

a18 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURAUD_C]

a17 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURNOK_C]

a16 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, GBPAUD_C]

a15 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, AUDNZD_C]

a14 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, CADCHF_C]

a13 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, USDNOK_C]

a12 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, AUDCAD_C]

a11 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, AUDCHF_C]

a10 = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURCHF_C]

a9  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, GBPCHF_C]

a8  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURCAD_C]

a7  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, GBPCAD_C]

a6  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, NZDUSD_C]

a5  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, USDCAD_C]

a4  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, USDCHF_C]

a3  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, AUDUSD_C]

a2  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, EURUSD_C]

a1  = [AUDJPY_O,CADJPY_O,GBPJPY_O,CHFJPY_O,EURJPY_O,USDJPY_O,EURAUD_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,EURCHF_O,GBPCHF_O,EURCAD_O,GBPCAD_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O, GBPUSD_C]







T_AUDJPY = pd.concat(a24,axis = 1)

T_CADJPY = pd.concat(a23,axis = 1)

T_GBPJPY = pd.concat(a22,axis = 1)

T_CHFJPY = pd.concat(a21,axis = 1)

T_EURJPY = pd.concat(a20,axis = 1)

T_USDJPY = pd.concat(a19,axis = 1)

T_EURAUD = pd.concat(a18,axis = 1)

T_EURNOK = pd.concat(a17,axis = 1)

T_GBPAUD = pd.concat(a16,axis = 1)

T_AUDNZD = pd.concat(a15,axis = 1)

T_CADCHF = pd.concat(a14,axis = 1)

T_USDNOK = pd.concat(a13,axis = 1)

T_AUDCAD = pd.concat(a12,axis = 1)

T_AUDCHF = pd.concat(a11,axis = 1)

T_EURCHF = pd.concat(a10,axis = 1)

T_GBPCHF = pd.concat(a9 ,axis = 1)

T_EURCAD = pd.concat(a8 ,axis = 1)

T_GBPCAD = pd.concat(a7 ,axis = 1)

T_NZDUSD = pd.concat(a6 ,axis = 1)

T_USDCAD = pd.concat(a5 ,axis = 1)

T_USDCHF = pd.concat(a4 ,axis = 1)

T_AUDUSD = pd.concat(a3 ,axis = 1)

T_EURUSD = pd.concat(a2 ,axis = 1)

T_GBPUSD = pd.concat(a1 ,axis = 1)

corr = T_GBPUSD.corr()

fig, ax = plt.subplots(figsize=(20,20))

ax = sns.heatmap(

    corr, 

    vmin=-1,

    vmax=1,

    center=0,

    cmap=sns.diverging_palette(20, 220, n=99),

    square=True,

    annot=True,

    fmt = '.1'

    

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

)

plt.show()




T_AUDJPY_X = T_AUDJPY.iloc[:,:-1]

T_CADJPY_X = T_CADJPY.iloc[:,:-1]

T_GBPJPY_X = T_GBPJPY.iloc[:,:-1]

T_CHFJPY_X = T_CHFJPY.iloc[:,:-1]

T_EURJPY_X = T_EURJPY.iloc[:,:-1]

T_USDJPY_X = T_USDJPY.iloc[:,:-1]

T_EURAUD_X = T_EURAUD.iloc[:,:-1]

T_EURNOK_X = T_EURNOK.iloc[:,:-1]

T_GBPAUD_X = T_GBPAUD.iloc[:,:-1]

T_AUDNZD_X = T_AUDNZD.iloc[:,:-1]

T_CADCHF_X = T_CADCHF.iloc[:,:-1]

T_USDNOK_X = T_USDNOK.iloc[:,:-1]

T_AUDCAD_X = T_AUDCAD.iloc[:,:-1]

T_AUDCHF_X = T_AUDCHF.iloc[:,:-1]

T_EURCHF_X = T_EURCHF.iloc[:,:-1]

T_GBPCHF_X = T_GBPCHF.iloc[:,:-1]

T_EURCAD_X = T_EURCAD.iloc[:,:-1]

T_GBPCAD_X = T_GBPCAD.iloc[:,:-1]

T_NZDUSD_X = T_NZDUSD.iloc[:,:-1]

T_USDCAD_X = T_USDCAD.iloc[:,:-1]

T_USDCHF_X = T_USDCHF.iloc[:,:-1]

T_AUDUSD_X = T_AUDUSD.iloc[:,:-1]

T_EURUSD_X = T_EURUSD.iloc[:,:-1]

T_GBPUSD_X = T_GBPUSD.iloc[:,:-1]





T_AUDJPY_Y = T_AUDJPY.iloc[:,-1]

T_CADJPY_Y = T_CADJPY.iloc[:,-1]

T_GBPJPY_Y = T_GBPJPY.iloc[:,-1]

T_CHFJPY_Y = T_CHFJPY.iloc[:,-1]

T_EURJPY_Y = T_EURJPY.iloc[:,-1]

T_USDJPY_Y = T_USDJPY.iloc[:,-1]

T_EURAUD_Y = T_EURAUD.iloc[:,-1]

T_EURNOK_Y = T_EURNOK.iloc[:,-1]

T_GBPAUD_Y = T_GBPAUD.iloc[:,-1]

T_AUDNZD_Y = T_AUDNZD.iloc[:,-1]

T_CADCHF_Y = T_CADCHF.iloc[:,-1]

T_USDNOK_Y = T_USDNOK.iloc[:,-1]

T_AUDCAD_Y = T_AUDCAD.iloc[:,-1]

T_AUDCHF_Y = T_AUDCHF.iloc[:,-1]

T_EURCHF_Y = T_EURCHF.iloc[:,-1]

T_GBPCHF_Y = T_GBPCHF.iloc[:,-1]

T_EURCAD_Y = T_EURCAD.iloc[:,-1]

T_GBPCAD_Y = T_GBPCAD.iloc[:,-1]

T_NZDUSD_Y = T_NZDUSD.iloc[:,-1]

T_USDCAD_Y = T_USDCAD.iloc[:,-1]

T_USDCHF_Y = T_USDCHF.iloc[:,-1]

T_AUDUSD_Y = T_AUDUSD.iloc[:,-1]

T_EURUSD_Y = T_EURUSD.iloc[:,-1]

T_GBPUSD_Y = T_GBPUSD.iloc[:,-1]







X_train_AUDJPY, X_test_AUDJPY, Y_train_AUDJPY, Y_test_AUDJPY = train_test_split(T_AUDJPY_X, T_AUDJPY_Y,shuffle=True)

X_train_CADJPY, X_test_CADJPY, Y_train_CADJPY, Y_test_CADJPY = train_test_split(T_CADJPY_X, T_CADJPY_Y,shuffle=True)

X_train_GBPJPY, X_test_GBPJPY, Y_train_GBPJPY, Y_test_GBPJPY = train_test_split(T_GBPJPY_X, T_GBPJPY_Y,shuffle=True)

X_train_CHFJPY, X_test_CHFJPY, Y_train_CHFJPY, Y_test_CHFJPY = train_test_split(T_CHFJPY_X, T_CHFJPY_Y,shuffle=True)

X_train_EURJPY, X_test_EURJPY, Y_train_EURJPY, Y_test_EURJPY = train_test_split(T_EURJPY_X, T_EURJPY_Y,shuffle=True)

X_train_USDJPY, X_test_USDJPY, Y_train_USDJPY, Y_test_USDJPY = train_test_split(T_USDJPY_X, T_USDJPY_Y,shuffle=True)

X_train_EURAUD, X_test_EURAUD, Y_train_EURAUD, Y_test_EURAUD = train_test_split(T_EURAUD_X, T_EURAUD_Y,shuffle=True)

X_train_EURNOK, X_test_EURNOK, Y_train_EURNOK, Y_test_EURNOK = train_test_split(T_EURNOK_X, T_EURNOK_Y,shuffle=True)

X_train_GBPAUD, X_test_GBPAUD, Y_train_GBPAUD, Y_test_GBPAUD = train_test_split(T_GBPAUD_X, T_GBPAUD_Y,shuffle=True)

X_train_AUDNZD, X_test_AUDNZD, Y_train_AUDNZD, Y_test_AUDNZD = train_test_split(T_AUDNZD_X, T_AUDNZD_Y,shuffle=True)

X_train_CADCHF, X_test_CADCHF, Y_train_CADCHF, Y_test_CADCHF = train_test_split(T_CADCHF_X, T_CADCHF_Y,shuffle=True)

X_train_USDNOK, X_test_USDNOK, Y_train_USDNOK, Y_test_USDNOK = train_test_split(T_USDNOK_X, T_USDNOK_Y,shuffle=True)

X_train_AUDCAD, X_test_AUDCAD, Y_train_AUDCAD, Y_test_AUDCAD = train_test_split(T_AUDCAD_X, T_AUDCAD_Y,shuffle=True)

X_train_AUDCHF, X_test_AUDCHF, Y_train_AUDCHF, Y_test_AUDCHF = train_test_split(T_AUDCHF_X, T_AUDCHF_Y,shuffle=True)

X_train_EURCHF, X_test_EURCHF, Y_train_EURCHF, Y_test_EURCHF = train_test_split(T_EURCHF_X, T_EURCHF_Y,shuffle=True)

X_train_GBPCHF, X_test_GBPCHF, Y_train_GBPCHF, Y_test_GBPCHF = train_test_split(T_GBPCHF_X, T_GBPCHF_Y,shuffle=True)

X_train_EURCAD, X_test_EURCAD, Y_train_EURCAD, Y_test_EURCAD = train_test_split(T_EURCAD_X, T_EURCAD_Y,shuffle=True)

X_train_GBPCAD, X_test_GBPCAD, Y_train_GBPCAD, Y_test_GBPCAD = train_test_split(T_GBPCAD_X, T_GBPCAD_Y,shuffle=True)

X_train_NZDUSD, X_test_NZDUSD, Y_train_NZDUSD, Y_test_NZDUSD = train_test_split(T_NZDUSD_X, T_NZDUSD_Y,shuffle=True)

X_train_USDCAD, X_test_USDCAD, Y_train_USDCAD, Y_test_USDCAD = train_test_split(T_USDCAD_X, T_USDCAD_Y,shuffle=True)

X_train_USDCHF, X_test_USDCHF, Y_train_USDCHF, Y_test_USDCHF = train_test_split(T_USDCHF_X, T_USDCHF_Y,shuffle=True)

X_train_AUDUSD, X_test_AUDUSD, Y_train_AUDUSD, Y_test_AUDUSD = train_test_split(T_AUDUSD_X, T_AUDUSD_Y,shuffle=True)

X_train_EURUSD, X_test_EURUSD, Y_train_EURUSD, Y_test_EURUSD = train_test_split(T_EURUSD_X, T_EURUSD_Y,shuffle=True)

X_train_GBPUSD, X_test_GBPUSD, Y_train_GBPUSD, Y_test_GBPUSD = train_test_split(T_GBPUSD_X, T_GBPUSD_Y,shuffle=True)



GBPUSD_li = LinearRegression()

EURUSD_li = LinearRegression()

AUDUSD_li = LinearRegression()

NZDUSD_li = LinearRegression()

USDCAD_li = LinearRegression()

EURCAD_li = LinearRegression()

GBPCAD_li = LinearRegression()

EURAUD_li = LinearRegression()

EURNOK_li = LinearRegression()

GBPAUD_li = LinearRegression()

AUDNZD_li = LinearRegression()

CADCHF_li = LinearRegression()

USDNOK_li = LinearRegression()

AUDCAD_li = LinearRegression()

AUDCHF_li = LinearRegression()

EURCHF_li = LinearRegression()

GBPCHF_li = LinearRegression()

USDCHF_li = LinearRegression()

USDJPY_li = LinearRegression()

EURJPY_li = LinearRegression()

AUDJPY_li = LinearRegression()

CADJPY_li = LinearRegression()

GBPJPY_li = LinearRegression()

CHFJPY_li = LinearRegression()











GBPUSD_li.fit(X_train_GBPUSD, Y_train_GBPUSD)

EURUSD_li.fit(X_train_EURUSD, Y_train_EURUSD)

AUDUSD_li.fit(X_train_AUDUSD, Y_train_AUDUSD)

NZDUSD_li.fit(X_train_NZDUSD, Y_train_NZDUSD)

USDCAD_li.fit(X_train_USDCAD, Y_train_USDCAD)

EURCAD_li.fit(X_train_EURCAD, Y_train_EURCAD)

GBPCAD_li.fit(X_train_GBPCAD, Y_train_GBPCAD)

EURAUD_li.fit(X_train_EURAUD, Y_train_EURAUD)

EURNOK_li.fit(X_train_EURNOK, Y_train_EURNOK)

GBPAUD_li.fit(X_train_GBPAUD, Y_train_GBPAUD)

AUDNZD_li.fit(X_train_AUDNZD, Y_train_AUDNZD)

CADCHF_li.fit(X_train_CADCHF, Y_train_CADCHF)

USDNOK_li.fit(X_train_USDNOK, Y_train_USDNOK)

AUDCAD_li.fit(X_train_AUDCAD, Y_train_AUDCAD)

AUDCHF_li.fit(X_train_AUDCHF, Y_train_AUDCHF)

EURCHF_li.fit(X_train_EURCHF, Y_train_EURCHF)

GBPCHF_li.fit(X_train_GBPCHF, Y_train_GBPCHF)

USDCHF_li.fit(X_train_USDCHF, Y_train_USDCHF)

USDJPY_li.fit(X_train_USDJPY, Y_train_USDJPY)

EURJPY_li.fit(X_train_EURJPY, Y_train_EURJPY)

AUDJPY_li.fit(X_train_AUDJPY, Y_train_AUDJPY)

CADJPY_li.fit(X_train_CADJPY, Y_train_CADJPY)

GBPJPY_li.fit(X_train_GBPJPY, Y_train_GBPJPY)

CHFJPY_li.fit(X_train_CHFJPY, Y_train_CHFJPY)



GBPUSD_tahmin = GBPUSD_li.predict(X_test_GBPUSD)

EURUSD_tahmin = EURUSD_li.predict(X_test_EURUSD)

AUDUSD_tahmin = AUDUSD_li.predict(X_test_AUDUSD)

NZDUSD_tahmin = NZDUSD_li.predict(X_test_NZDUSD)

USDCAD_tahmin = USDCAD_li.predict(X_test_USDCAD)

EURCAD_tahmin = EURCAD_li.predict(X_test_EURCAD)

GBPCAD_tahmin = GBPCAD_li.predict(X_test_GBPCAD)

EURAUD_tahmin = EURAUD_li.predict(X_test_EURAUD)

EURNOK_tahmin = EURNOK_li.predict(X_test_EURNOK)

GBPAUD_tahmin = GBPAUD_li.predict(X_test_GBPAUD)

AUDNZD_tahmin = AUDNZD_li.predict(X_test_AUDNZD)

CADCHF_tahmin = CADCHF_li.predict(X_test_CADCHF)

USDNOK_tahmin = USDNOK_li.predict(X_test_USDNOK)

AUDCAD_tahmin = AUDCAD_li.predict(X_test_AUDCAD)

AUDCHF_tahmin = AUDCHF_li.predict(X_test_AUDCHF)

EURCHF_tahmin = EURCHF_li.predict(X_test_EURCHF)

GBPCHF_tahmin = GBPCHF_li.predict(X_test_GBPCHF)

USDCHF_tahmin = USDCHF_li.predict(X_test_USDCHF)

USDJPY_tahmin = USDJPY_li.predict(X_test_USDJPY)

EURJPY_tahmin = EURJPY_li.predict(X_test_EURJPY)

AUDJPY_tahmin = AUDJPY_li.predict(X_test_AUDJPY)

CADJPY_tahmin = CADJPY_li.predict(X_test_CADJPY)

GBPJPY_tahmin = GBPJPY_li.predict(X_test_GBPJPY)

CHFJPY_tahmin = CHFJPY_li.predict(X_test_CHFJPY)







print('GBPUSD_Tahmin : ' , np.round(r2_score(Y_test_GBPUSD, GBPUSD_tahmin),3))

print('EURUSD_Tahmin : ' , np.round(r2_score(Y_test_EURUSD, EURUSD_tahmin),3))

print('AUDUSD_Tahmin : ' , np.round(r2_score(Y_test_AUDUSD, AUDUSD_tahmin),3))

print('NZDUSD_Tahmin : ' , np.round(r2_score(Y_test_NZDUSD, NZDUSD_tahmin),3))

print('USDCAD_Tahmin : ' , np.round(r2_score(Y_test_USDCAD, USDCAD_tahmin),3))

print('EURCAD_Tahmin : ' , np.round(r2_score(Y_test_EURCAD, EURCAD_tahmin),3))

print('GBPCAD_Tahmin : ' , np.round(r2_score(Y_test_GBPCAD, GBPCAD_tahmin),3))

print('EURAUD_Tahmin : ' , np.round(r2_score(Y_test_EURAUD, EURAUD_tahmin),3))

print('EURNOK_Tahmin : ' , np.round(r2_score(Y_test_EURNOK, EURNOK_tahmin),3))

print('GBPAUD_Tahmin : ' , np.round(r2_score(Y_test_GBPAUD, GBPAUD_tahmin),3))

print('AUDNZD_Tahmin : ' , np.round(r2_score(Y_test_AUDNZD, AUDNZD_tahmin),3))

print('CADCHF_Tahmin : ' , np.round(r2_score(Y_test_CADCHF, CADCHF_tahmin),3))

print('USDNOK_Tahmin : ' , np.round(r2_score(Y_test_USDNOK, USDNOK_tahmin),3))

print('AUDCAD_Tahmin : ' , np.round(r2_score(Y_test_AUDCAD, AUDCAD_tahmin),3))

print('AUDCHF_Tahmin : ' , np.round(r2_score(Y_test_AUDCHF, AUDCHF_tahmin),3))

print('EURCHF_Tahmin : ' , np.round(r2_score(Y_test_EURCHF, EURCHF_tahmin),3))

print('GBPCHF_Tahmin : ' , np.round(r2_score(Y_test_GBPCHF, GBPCHF_tahmin),3))

print('USDCHF_Tahmin : ' , np.round(r2_score(Y_test_USDCHF, USDCHF_tahmin),3))

print('USDJPY_Tahmin : ' , np.round(r2_score(Y_test_USDJPY, USDJPY_tahmin),3))

print('EURJPY_Tahmin : ' , np.round(r2_score(Y_test_EURJPY, EURJPY_tahmin),3))

print('AUDJPY_Tahmin : ' , np.round(r2_score(Y_test_AUDJPY, AUDJPY_tahmin),3))

print('CADJPY_Tahmin : ' , np.round(r2_score(Y_test_CADJPY, CADJPY_tahmin),3))

print('GBPJPY_Tahmin : ' , np.round(r2_score(Y_test_GBPJPY, GBPJPY_tahmin),3))

print('CHFJPY_Tahmin : ' , np.round(r2_score(Y_test_CHFJPY, CHFJPY_tahmin),3))



           

def tahmin(AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O):

    print('GBPUSD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*GBPUSD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURUSD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURUSD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('AUDUSD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*AUDUSD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('NZDUSD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*NZDUSD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('USDCAD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*USDCAD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURCAD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURCAD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('GBPCAD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*GBPCAD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURAUD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURAUD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURNOK SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURNOK_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('GBPAUD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*GBPAUD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('AUDNZD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*AUDNZD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('CADCHF SAAT 00:00 DA FIYAT SU OLUR :',np.round(*CADCHF_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('USDNOK SAAT 00:00 DA FIYAT SU OLUR :',np.round(*USDNOK_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('AUDCAD SAAT 00:00 DA FIYAT SU OLUR :',np.round(*AUDCAD_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('AUDCHF SAAT 00:00 DA FIYAT SU OLUR :',np.round(*AUDCHF_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURCHF SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURCHF_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('GBPCHF SAAT 00:00 DA FIYAT SU OLUR :',np.round(*GBPCHF_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('USDCHF SAAT 00:00 DA FIYAT SU OLUR :',np.round(*USDCHF_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('GBPJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*GBPJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('CHFJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*CHFJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('CADJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*CADJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('EURJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*EURJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('AUDJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*AUDJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    print('USDJPY SAAT 00:00 DA FIYAT SU OLUR :',np.round(*USDJPY_li.predict([[AUDJPY_O,EURAUD_O,CADJPY_O,EURNOK_O,GBPAUD_O,AUDNZD_O,CADCHF_O,USDNOK_O,AUDCAD_O,AUDCHF_O,GBPJPY_O,CHFJPY_O,EURCHF_O,GBPCHF_O,EURCAD_O,EURJPY_O,GBPCAD_O,USDJPY_O,NZDUSD_O,USDCAD_O,USDCHF_O,AUDUSD_O,EURUSD_O,GBPUSD_O]]),5))

    




GBPUSD_T = np.array(GBPUSD_tahmin[-15:])

EURUSD_T = np.array(EURUSD_tahmin[-15:])

AUDUSD_T = np.array(AUDUSD_tahmin[-15:])

NZDUSD_T = np.array(NZDUSD_tahmin[-15:])

USDCAD_T = np.array(USDCAD_tahmin[-15:])

EURCAD_T = np.array(EURCAD_tahmin[-15:])

GBPCAD_T = np.array(GBPCAD_tahmin[-15:])

EURAUD_T = np.array(EURAUD_tahmin[-15:])

EURNOK_T = np.array(EURNOK_tahmin[-15:])

GBPAUD_T = np.array(GBPAUD_tahmin[-15:])

AUDNZD_T = np.array(AUDNZD_tahmin[-15:])

CADCHF_T = np.array(CADCHF_tahmin[-15:])

USDNOK_T = np.array(USDNOK_tahmin[-15:])

AUDCAD_T = np.array(AUDCAD_tahmin[-15:])

AUDCHF_T = np.array(AUDCHF_tahmin[-15:])

EURCHF_T = np.array(EURCHF_tahmin[-15:])

GBPCHF_T = np.array(GBPCHF_tahmin[-15:])

USDCHF_T = np.array(USDCHF_tahmin[-15:])

USDJPY_T = np.array(USDJPY_tahmin[-15:])

EURJPY_T = np.array(EURJPY_tahmin[-15:])

AUDJPY_T = np.array(AUDJPY_tahmin[-15:])

CADJPY_T = np.array(CADJPY_tahmin[-15:])

GBPJPY_T = np.array(GBPJPY_tahmin[-15:])

CHFJPY_T = np.array(CHFJPY_tahmin[-15:])



GBPUSD_G = np.array(Y_test_GBPUSD[-15:])

EURUSD_G = np.array(Y_test_EURUSD[-15:])

AUDUSD_G = np.array(Y_test_AUDUSD[-15:])

NZDUSD_G = np.array(Y_test_NZDUSD[-15:])

USDCAD_G = np.array(Y_test_USDCAD[-15:])

EURCAD_G = np.array(Y_test_EURCAD[-15:])

GBPCAD_G = np.array(Y_test_GBPCAD[-15:])

EURAUD_G = np.array(Y_test_EURAUD[-15:])

EURNOK_G = np.array(Y_test_EURNOK[-15:])

GBPAUD_G = np.array(Y_test_GBPAUD[-15:])

AUDNZD_G = np.array(Y_test_AUDNZD[-15:])

CADCHF_G = np.array(Y_test_CADCHF[-15:])

USDNOK_G = np.array(Y_test_USDNOK[-15:])

AUDCAD_G = np.array(Y_test_AUDCAD[-15:])

AUDCHF_G = np.array(Y_test_AUDCHF[-15:])

EURCHF_G = np.array(Y_test_EURCHF[-15:])

GBPCHF_G = np.array(Y_test_GBPCHF[-15:])

USDCHF_G = np.array(Y_test_USDCHF[-15:])

USDJPY_G = np.array(Y_test_USDJPY[-15:])

EURJPY_G = np.array(Y_test_EURJPY[-15:])

AUDJPY_G = np.array(Y_test_AUDJPY[-15:])

CADJPY_G = np.array(Y_test_CADJPY[-15:])

GBPJPY_G = np.array(Y_test_GBPJPY[-15:])

CHFJPY_G = np.array(Y_test_CHFJPY[-15:])





fig = plt.figure(figsize = (30,40))



a = 6

b = 4



plt.subplot(a, b, 1)

plt.plot(GBPUSD_T, label = 'tahmin')

plt.plot(GBPUSD_G, label = 'gerçek')

plt.title('GBPUSD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 2)

plt.plot(EURUSD_T, label = 'tahmin')

plt.plot(EURUSD_G, label = 'gerçek')

plt.title('EURUSD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 3)

plt.plot(AUDUSD_T, label = 'tahmin')

plt.plot(AUDUSD_G, label = 'gerçek')

plt.title('AUDUSD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 4)

plt.plot(NZDUSD_T, label = 'tahmin')

plt.plot(NZDUSD_G, label = 'gerçek')

plt.title('NZDUSD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 5)

plt.plot(USDCAD_T, label = 'tahmin')

plt.plot(USDCAD_G, label = 'gerçek')

plt.title('USDCAD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 6)

plt.plot(EURCAD_T, label = 'tahmin')

plt.plot(EURCAD_G, label = 'gerçek')

plt.title('EURCAD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 7)

plt.plot(GBPCAD_T, label = 'tahmin')

plt.plot(GBPCAD_G, label = 'gerçek')

plt.title('GBPCAD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 8)

plt.plot(EURAUD_T, label = 'tahmin')

plt.plot(EURAUD_G, label = 'gerçek')

plt.title('EURAUD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 9)

plt.plot(EURNOK_T, label = 'tahmin')

plt.plot(EURNOK_G, label = 'gerçek')

plt.title('EURNOK ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 10)

plt.plot(GBPAUD_T, label = 'tahmin')

plt.plot(GBPAUD_G, label = 'gerçek')

plt.title('GBPAUD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 11)

plt.plot(AUDNZD_T, label = 'tahmin')

plt.plot(AUDNZD_G, label = 'gerçek')

plt.title('AUDNZD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 12)

plt.plot(CADCHF_T, label = 'tahmin')

plt.plot(CADCHF_G, label = 'gerçek')

plt.title('CADCHF ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 13)

plt.plot(USDNOK_T, label = 'tahmin')

plt.plot(USDNOK_G, label = 'gerçek')

plt.title('USDNOK ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 14)

plt.plot(AUDCAD_T, label = 'tahmin')

plt.plot(AUDCAD_G, label = 'gerçek')

plt.title('AUDCAD ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 15)

plt.plot(AUDCHF_T, label = 'tahmin')

plt.plot(AUDCHF_G, label = 'gerçek')

plt.title('AUDCHF ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 16)

plt.plot(EURCHF_T, label = 'tahmin')

plt.plot(EURCHF_G, label = 'gerçek')

plt.title('EURCHF ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 17)

plt.plot(GBPCHF_T, label = 'tahmin')

plt.plot(GBPCHF_G, label = 'gerçek')

plt.title('GBPCHF ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 18)

plt.plot(USDCHF_T, label = 'tahmin')

plt.plot(USDCHF_G, label = 'gerçek')

plt.title('USDCHF ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 19)

plt.plot(USDJPY_T, label = 'tahmin')

plt.plot(USDJPY_G, label = 'gerçek')

plt.title('USDJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 20)

plt.plot(EURJPY_T, label = 'tahmin')

plt.plot(EURJPY_G, label = 'gerçek')

plt.title('EURJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 21)

plt.plot(AUDJPY_T, label = 'tahmin')

plt.plot(AUDJPY_G, label = 'gerçek')

plt.title('AUDJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 22)

plt.plot(CADJPY_T, label = 'tahmin')

plt.plot(CADJPY_G, label = 'gerçek')

plt.title('CADJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()





plt.subplot(a, b, 23)

plt.plot(GBPJPY_T, label = 'tahmin')

plt.plot(GBPJPY_G, label = 'gerçek')

plt.title('GBPJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.subplot(a, b, 24)

plt.plot(CHFJPY_T, label = 'tahmin')

plt.plot(CHFJPY_G, label = 'gerçek')

plt.title('CHFJPY ve makina isabet grafiyi')

plt.xlabel('fiyat')

plt.ylabel('tahmin sayisi')

plt.legend()



plt.show()


tahmin(

 75.772

,80.298

,138.502

,116.063

,124.820

,105.810

,1.64733

,10.63092

,1.82790

,1.09489

,0.69184

,9.01192

,0.94366

,0.65283

,1.07536

,1.19331

,1.55432

,1.72477

,0.65396

,1.31772

,0.91165

,0.71610

,1.17962

,1.30898)