# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd 

import datetime

import numpy as np 

import matplotlib.pyplot as plt 

import statsmodels.api as sm

from pylab import rcParams

import warnings

from pandas.core.nanops import nanmean as pd_nanmean

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt



from sklearn.metrics import mean_absolute_error

from datetime import datetime, timedelta

from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

%matplotlib inline
sputnik_data = pd.read_csv('/kaggle/input/sputnik/train.csv', sep =',')

sputnik_data.head(10)
import numpy as np



def smape(A, F):

    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))
x_best = []

y_best = []

z_best = []
err1 = 0

err2 = 0

err3 = 0

df_res = pd.DataFrame(columns=['id','error'])

for i in range(600):

#     print(i)

    df = sputnik_data[sputnik_data['sat_id']==i]

    k = len(df)-df['x'].isnull().sum()

    train = df[:int(k*2/3)]

    val = df[int(k*2/3):k]

    train['label'] = 'train'

    val['label'] = 'val'

    df1 = pd.concat((train,val), axis = 0)

    df1['target'] = np.where(df1.label == 'train', df1.x, np.nan)

    df2 = pd.concat((train,val), axis = 0)

    df2['target'] = np.where(df2.label == 'train', df2.y, np.nan)

    df3 = pd.concat((train,val), axis = 0)

    df3['target'] = np.where(df3.label == 'train', df3.z, np.nan)

    l = 24

    l1 = 21

    if i==265 or i==277:

        l=20

    if i==252 or 301:

        l=20

    if i==26 or i==199 or i==265 or i==277:

        l1=0

    if i==252 or 301:

        l1=0

#     if i in x_best:

    fit1 = ExponentialSmoothing(np.asarray(train['x']),seasonal_periods=24 , seasonal='add').fit()

    forecast1_1 = pd.Series(fit1.forecast(len(val)))

    forecast1_1.index = val.index

#     if i in y_best:

    fit2 = ExponentialSmoothing(np.asarray(train['y']),seasonal_periods=24 , seasonal='add').fit()

    forecast2_2 = pd.Series(fit1.forecast(len(val)))

    forecast2_2.index = val.index

#     if i in z_best:

    fit3 = ExponentialSmoothing(np.asarray(train['z']),seasonal_periods=24 , seasonal='add').fit()

    forecast3_3 = pd.Series(fit1.forecast(len(val)))

    forecast3_3.index = val.index

    lag_period = len(val)

    features1 = []

    for period_mult in range(1,2,1):

        df1["lag_period_{}".format(period_mult)] = df1.target.shift(period_mult*lag_period)

        features1.append("lag_period_{}".format(period_mult))

    lag_period1 = 24

    for period_mult in range(1,l,1):

        df1["lag_period_{}".format(period_mult+3)] = df1.target.shift(lag_period+period_mult)

        features1.append("lag_period_{}".format(period_mult+3))

    for period_mult in range(1,1,1):

        df1["lag_period_{}".format(period_mult+30)] = df1.target.shift(lag_period*2+period_mult+24)

        features1.append("lag_period_{}".format(period_mult+30))

    for period_mult in range(1,1,1):

        df1["lag_period_{}".format(period_mult+30)] = df1.target.shift(period_mult*lag_period+period_mult)

        features1.append("lag_period_{}".format(period_mult+30))

#     features1 = []



#     for period_mult in range(1,1,1):

#         df1["lag_period_{}".format(period_mult+3)] = 0

#         for i in train.index:

#     #         print(df1.index)

#             df1["lag_period_{}".format(period_mult+3)][i] = df1.target.shift(24*period_mult+i%24)[i]

#     #         print(df1.target.shift(24*period_mult)[i])

#         for i in val.index:

#             df1["lag_period_{}".format(period_mult+3)][i] = df1.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

# #             print(df1.target.shift(24*period_mult*((i-len(train)+24)//24))[i])

#     #     df1.target.shift(period_mult*lag_period)

#         features1.append("lag_period_{}".format(period_mult+3))

        

    features2 = []

    for period_mult in range(1,2,1):

        df2["lag_period_{}".format(period_mult)] = df2.target.shift(period_mult*lag_period)

        features2.append("lag_period_{}".format(period_mult))

    for period_mult in range(1,l,1):

        df2["lag_period_{}".format(period_mult+3)] = df2.target.shift(lag_period+period_mult)

        features2.append("lag_period_{}".format(period_mult+3))

    for period_mult in range(1,1,1):

        df2["lag_period_{}".format(period_mult+30)] = df2.target.shift(lag_period*2+period_mult+24)

        features2.append("lag_period_{}".format(period_mult+30))

    for period_mult in range(1,1,1):

        df2["lag_period_{}".format(period_mult+30)] = df2.target.shift(period_mult*lag_period+period_mult)

        features2.append("lag_period_{}".format(period_mult+30))

    lag_period1 = 24

#     features1 = []

    for period_mult in range(1,1,1):

        df2["lag_period_{}".format(period_mult+3)] = 0

        for i in train.index:

    #         print(df1.index)

            df2["lag_period_{}".format(period_mult+3)][i] = df2.target.shift(24*period_mult+i%24)[i]

    #         print(df1.target.shift(24*period_mult)[i])

        for i in val.index:

            df2["lag_period_{}".format(period_mult+3)][i] = df2.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

#             print(df2.target.shift(24*period_mult*((i-len(train)+24)//24))[i])

    #     df1.target.shift(period_mult*lag_period)

        features2.append("lag_period_{}".format(period_mult+3))

    

    

    features3 = []

    for period_mult in range(1,2,1):

        df3["lag_period_{}".format(period_mult)] = df3.target.shift(period_mult*lag_period)

        features3.append("lag_period_{}".format(period_mult))

    lag_period1 = 24

    for period_mult in range(1,l,1):

        df3["lag_period_{}".format(period_mult+3)] = df3.target.shift(lag_period+period_mult)

        features3.append("lag_period_{}".format(period_mult+3))

    for period_mult in range(1,1,1):

        df3["lag_period_{}".format(period_mult+30)] = df3.target.shift(period_mult*lag_period+period_mult)

        features3.append("lag_period_{}".format(period_mult+30))

    for period_mult in range(1,1,1):

        df3["lag_period_{}".format(period_mult+30)] = df3.target.shift(lag_period*2+period_mult+24)

        features3.append("lag_period_{}".format(period_mult+30))

    

    for period_mult in range(1,1,1):

        df3["lag_period_{}".format(period_mult+3)] = 0

        for i in train.index:

    #         print(df1.index)

            df3["lag_period_{}".format(period_mult+3)][i] = df3.target.shift(24*period_mult+i%24)[i]

    #         print(df1.target.shift(24*period_mult)[i])

        for i in val.index:

            df3["lag_period_{}".format(period_mult+3)][i] = df3.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

#             print(df3.target.shift(24*period_mult*((i-len(train)+24)//24))[i])

    #     df1.target.shift(period_mult*lag_period)

        features3.append("lag_period_{}".format(period_mult+3))

#         features1.append("lag_period_{}".format(period_mult+3))

#     for period_mult in range(3,40,1):

#         df3["lag_period_{}".format(period_mult)] = df3.target.shift(int(k*2/3)-period_mult)

#         features3.append("lag_period_{}".format(period_mult))

#     for period_mult in range(3,40,1):

#         df1["lag_period_{}".format(period_mult)] = df1.target.shift(int(k*2/3)-period_mult)

#         features1.append("lag_period_{}".format(period_mult))

#     for period_mult in range(3,40,1):

#         df2["lag_period_{}".format(period_mult)] = df2.target.shift(int(k*2/3)-period_mult)

#         features2.append("lag_period_{}".format(period_mult))

    # # лаговые статистики

    df1['lagf_mean'] = df1[features1[:1]].mean(axis = 1)

    df2['lagf_mean'] = df2[features2[:1]].mean(axis = 1)

    df3['lagf_mean'] = df3[features3[:1]].mean(axis = 1)

    df1['lagf_mean1'] = df1[features1[1:]].mean(axis = 1)

    df2['lagf_mean1'] = df2[features2[1:]].mean(axis = 1)

    df3['lagf_mean1'] = df3[features3[1:]].mean(axis = 1)



    features1.extend(['lagf_mean'])

    features1.extend(['lagf_mean1'])

    model1 = LinearRegression()

    train_df1 = df1[df1.label == 'train'][features1 + ['target']].dropna()

    test_df1 = df1[df1.label == 'val'][features1]



    features2.extend(['lagf_mean'])

    features2.extend(['lagf_mean1'])

    model2 = LinearRegression()

    train_df2 = df2[df2.label == 'train'][features2 + ['target']].dropna()

    test_df2 = df2[df2.label == 'val'][features2]



    features3.extend(['lagf_mean'])

    features3.extend(['lagf_mean1'])

    model3 = LinearRegression()

    train_df3 = df3[df3.label == 'train'][features3 + ['target']].dropna()

    test_df3 = df3[df3.label == 'val'][features3]

#     print(test_df3.head())

    model1.fit(train_df1.drop('target', axis = 1) ,train_df1['target'])

#     if i not in x_best:

    forecast1 = model1.predict(test_df1)

    test_df1['prediction'] = forecast1

    model2.fit(train_df2.drop('target', axis = 1) ,train_df2['target'])

#     if i not in y_best:

    forecast2 = model2.predict(test_df2)

    test_df2['prediction'] = forecast2

    model3.fit(train_df3.drop('target', axis = 1) ,train_df3['target'])

#     if i not in z_best:

    forecast3 = model3.predict(test_df3)

    test_df1['prediction'] = forecast1

    test_df2['prediction'] = forecast2

    test_df3['prediction'] = forecast3

    err11 = 0

    err22 = 0

    err33 = 0

    err11 = smape(test_df1.prediction,val['x'])

    err22 = smape(test_df2.prediction,val['y'])

    err33 = smape(test_df3.prediction,val['z'])

#     print(smape(forecast1_1,val['x']))

#     print(smape(forecast2_2,val['y']))

#     print(smape(forecast3_3,val['z']))

    if(err11>smape(forecast1_1,val['x'])):

        err11 = smape(forecast1_1,val['x'])

        x_best.append(i)

    if(err22>smape(forecast2_2,val['y'])):

        err22 = smape(forecast2_2,val['y'])

        y_best.append(i)

    if(err33>smape(forecast3_3,val['z'])):

        err33 = smape(forecast3_3,val['z'])

        z_best.append(i)

        

    

    err1 += err11

    err2 += err22

    err3 += err33

    print(i,err11,'err1',err22,'err2',err33,'err3')
err1/600
err2/600
err3/600
x_best
df_res = pd.DataFrame(columns=['id','error'])

for i in range(600):

#     if(i%10==0):

    print(i)

    df = sputnik_data[sputnik_data['sat_id']==i]

    k = len(df)-df['x'].isnull().sum()

    train = df[:k]

    val = df[k:]

    train['label'] = 'train'

    val['label'] = 'val'

    df1 = pd.concat((train,val), axis = 0)

    df1['target'] = np.where(df1.label == 'train', df1.x, np.nan)

    df2 = pd.concat((train,val), axis = 0)

    df2['target'] = np.where(df2.label == 'train', df2.y, np.nan)

    df3 = pd.concat((train,val), axis = 0)

    df3['target'] = np.where(df3.label == 'train', df3.z, np.nan)

    lag_period = len(val)

    features1 = []

    l = 24

    p = 5

    if i==26 or i==199 or i==265 or i==277:

        p=4

    if i==252 or 301:

        p=3

        

        

    if i in x_best:

        fit1 = ExponentialSmoothing(np.asarray(train['x']),seasonal_periods=24 , seasonal='add').fit()

        forecast1_1 = pd.Series(fit1.forecast(len(val)))

        forecast1_1.index = val.index

    if i in y_best:

        fit2 = ExponentialSmoothing(np.asarray(train['y']),seasonal_periods=24 , seasonal='add').fit()

        forecast2_2 = pd.Series(fit1.forecast(len(val)))

        forecast2_2.index = val.index

    if i in z_best:

        fit3 = ExponentialSmoothing(np.asarray(train['z']),seasonal_periods=24 , seasonal='add').fit()

        forecast3_3 = pd.Series(fit1.forecast(len(val)))

        forecast3_3.index = val.index    

        

    

    for period_mult in range(1,3,1):

        df1["lag_period_{}".format(period_mult)] = df1.target.shift(period_mult*lag_period)

        features1.append("lag_period_{}".format(period_mult))

    for period_mult in range(1,l,1):

        df1["lag_period_{}".format(period_mult+3)] = df1.target.shift(lag_period+period_mult)

        features1.append("lag_period_{}".format(period_mult+3))

#     for period_mult in range(1,3,1):

#         df1["lag_period_{}".format(period_mult+3)] = df1.target.shift(period_mult*lag_period+24)

#         features1.append("lag_period_{}".format(period_mult+3))

# #     for period_mult in range(1,p,1):

#         df1["lag_period_{}".format(period_mult+3)] = 0

#         for i in train.index:

#             df1["lag_period_{}".format(period_mult+3)][i] = df1.target.shift(24*period_mult+i%24)[i]

#         for i in val.index:

#             df1["lag_period_{}".format(period_mult+3)][i] = df1.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

#         features1.append("lag_period_{}".format(period_mult+3))

        

        

    features2 = []

    for period_mult in range(1,3,1):

        df2["lag_period_{}".format(period_mult)] = df2.target.shift(period_mult*lag_period)

        features2.append("lag_period_{}".format(period_mult))

    for period_mult in range(1,l,1):

        df2["lag_period_{}".format(period_mult+3)] = df2.target.shift(lag_period+period_mult)

        features2.append("lag_period_{}".format(period_mult+3))

#     for period_mult in range(1,3,1):

#         df2["lag_period_{}".format(period_mult+3)] = df2.target.shift(period_mult*lag_period+24)

#         features2.append("lag_period_{}".format(period_mult+3))    

#     for period_mult in range(1,p,1):

#         df2["lag_period_{}".format(period_mult+3)] = 0

#         for i in train.index:

#             df2["lag_period_{}".format(period_mult+3)][i] = df2.target.shift(24*period_mult+i%24)[i]

#         for i in val.index:

#             df2["lag_period_{}".format(period_mult+3)][i] = df2.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

#         features2.append("lag_period_{}".format(period_mult+3))  

    

    features3 = []

    for period_mult in range(1,3,1):

        df3["lag_period_{}".format(period_mult)] = df3.target.shift(period_mult*lag_period)

        features3.append("lag_period_{}".format(period_mult))

#     for period_mult in range(1,3,1):

#         df3["lag_period_{}".format(period_mult+3)] = df3.target.shift(period_mult*lag_period+24)

#         features3.append("lag_period_{}".format(period_mult+3))

    for period_mult in range(1,l,1):

        df3["lag_period_{}".format(period_mult+3)] = df3.target.shift(lag_period+period_mult)

        features3.append("lag_period_{}".format(period_mult+3))

#     for period_mult in range(1,p,1):

#         df3["lag_period_{}".format(period_mult+3)] = 0

#         for i in train.index:

#             df3["lag_period_{}".format(period_mult+3)][i] = df3.target.shift(24*period_mult+i%24)[i]

#         for i in val.index:

#             df3["lag_period_{}".format(period_mult+3)][i] = df3.target.shift(24*(period_mult+((i-train.index.max())//24))+(i%24))[i]

#         features3.append("lag_period_{}".format(period_mult+3))

    # # лаговые статистики

    

    

    df1['lagf_mean'] = df1[features1[:2]].mean(axis = 1)

    df2['lagf_mean'] = df2[features2[:2]].mean(axis = 1)

    df3['lagf_mean'] = df3[features3[:2]].mean(axis = 1)

    df1['lagf_mean1'] = df1[features1[2:]].mean(axis = 1)

    df2['lagf_mean1'] = df2[features2[2:]].mean(axis = 1)

    df3['lagf_mean1'] = df3[features3[2:]].mean(axis = 1)



    features1.extend(['lagf_mean'])

    features1.extend(['lagf_mean1'])

    model1 = LinearRegression()

    train_df1 = df1[df1.label == 'train'][features1 + ['target']].dropna()

    test_df1 = df1[df1.label == 'val'][features1]



    features2.extend(['lagf_mean'])

    features2.extend(['lagf_mean1'])

    model2 = LinearRegression()

    train_df2 = df2[df2.label == 'train'][features2 + ['target']].dropna()

    test_df2 = df2[df2.label == 'val'][features2]



    features3.extend(['lagf_mean'])

    features3.extend(['lagf_mean1'])

    model3 = LinearRegression()

    train_df3 = df3[df3.label == 'train'][features3 + ['target']].dropna()

    test_df3 = df3[df3.label == 'val'][features3]



    model1.fit(train_df1.drop('target', axis = 1) ,train_df1['target'])

    forecast1 = model1.predict(test_df1)

    if i in x_best:

        forecast1 = forecast1_1

    test_df1['prediction'] = forecast1

    model2.fit(train_df2.drop('target', axis = 1) ,train_df2['target'])

    forecast2 = model2.predict(test_df2)

    if i in y_best:

        forecast2 = forecast2_2

    test_df2['prediction'] = forecast2

    model3.fit(train_df3.drop('target', axis = 1) ,train_df3['target'])

    forecast3 = model3.predict(test_df3)

    if i in z_best:

        forecast3 = forecast3_3

    test_df3['prediction'] = forecast3

    a1 =  np.hstack((np.array(df1.x[:k]),np.array(test_df1.prediction.values)))

    a2 =  np.hstack((np.array(df2.y[:k]),np.array(test_df2.prediction.values)))

    a3 =  np.hstack((np.array(df3.z[:k]),np.array(test_df3.prediction.values)))

    a1 = pd.DataFrame(a1)

    a2= pd.DataFrame(a2)

    a3= pd.DataFrame(a3)

    a1.index = df1.index

    a2.index = df2.index

    a3.index = df3.index

    df['x'] = a1

    df['y'] = a2

    df['z'] = a3

    df['error']  = np.linalg.norm(df[['x', 'y', 'z']].values - df[['x_sim', 'y_sim', 'z_sim']].values, axis=1)

    df_cur = pd.DataFrame(df[k:])

    df_cur = df_cur[['id','error']]

    df_res = df_res.merge(df_cur, how='outer')
df_res.info()
df_res.tail()
# with open("res_123.csv", "w") as f:

#     f.write(df_res.to_csv(columns=("id", "error"), index=False))

df_res.to_csv("res_123.csv",index= False)