import numpy as np 

import pandas as pd 

from pandas import DataFrame 



import matplotlib.pyplot as plt 

import seaborn as sns

import itertools

%matplotlib inline



from datetime import datetime, timedelta 

from statsmodels.tsa.arima_model import ARIMA 



import os
# test = pd.read_csv("data/test.csv", sep = ",")

# train = pd.read_csv("data/train.csv", sep =",")

# submission = pd.read_csv("data/submission.csv")



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("/kaggle/input/traindata/train.csv", sep=',')

test = pd.read_csv("/kaggle/input/testdata/test.csv", sep=',')

changed_test = pd.read_csv("/kaggle/input/new-changedtest/test_DHWeight_onDacon.csv", sep=',')

changed_train = pd.read_csv("/kaggle/input/new-changedtrain/train_DHWeight_onDacon.csv", sep=',')
changed_train.head()
def dailyhourlyMean(df):

    df['Time'] = pd.to_datetime(df.Time) ; df = df.set_index('Time')

    df['hour'] = df.index.hour ;  df['day'] = df.index.dayofweek



    return df.columns[:-2], df.groupby(['day','hour']).mean()
def DaconDHmeanPrep(df): #이 때 df는 아직 time Index가 안 되어있는 raw

    ids, df_mean = dailyhourlyMean(df)

    df1 = df

    for k in range(1,len(df.columns)):

        counting = df.loc[ df.iloc[:,k].isnull()==False ][ df.columns[k] ].index

        _id = pd.DataFrame(list(zip(counting[:-1], #값이 존재하는 index(시간) 중 마지막 제외

                          counting[1:]-counting[:-1] - 1)), # index 중 첫번째 제외 - 마지막 제외 - 1 : 각 인덱스 사이 구간에 얼마의 차이가 있는가, 즉 Null값이 몇개나 존재했는가

                 columns=['index','count'] )

        p005 = np.percentile(df.iloc[:,k].dropna(), 0.5)

        med = df.iloc[:,k].median()



        na_ids = _id[(_id['count'] >0)].reset_index(drop=True)

        if k%50 ==0: 

            print(k)

        for i, j in zip(na_ids['index'], na_ids['count']):

            initial = df.iloc[i,k]

            if initial > med:

                timeid = pd.to_datetime(df.iloc[i,0])

                timerange = timeid + pd.to_timedelta (np.arange(j+1), 'h')

                dh = [(t.dayofweek, t.hour) for t in timerange]

                means = df_mean.loc[dh, ids[k-1]]

                mean_tot = means.sum()

                weights = means/mean_tot

                w_values = weights*df.iloc[i,k]

                wv = pd.DataFrame(list(map(lambda x:x if x>p005 else np.nan, 

                                      df.iloc[i,k]*weights.values)))

                if np.isnan(wv.iloc[0,0]):

                    wv.iloc[0,0] = initial

                df1.iloc[i:i+j+1, k] = wv.values   

                # print('column ', ids[k-1], '| row', i, '| na count ',j,'\ntimerange ',timerange,'\nweights',weights, wv)

    df1['Time'] = pd.to_datetime(df1.Time) ; df1 = df1.set_index('Time') 

    return df1      
train2 = train.dropna(how='all', axis=1).copy()

changed_train = DaconDHmeanPrep(train2)

changed_train.head()
test2 = test.dropna(how='all', axis=1).copy()

changed_test = DaconDHmeanPrep(test2)

changed_test.head()
changed_test.to_csv("test_DHWeight_onDacon.csv")

# from IPython.display import FileLink

# FileLink(r'test_DHWeight_onDacon.csv')
changed_train.to_csv("train_DHWeight_onDacon.csv")

# from IPython.display import FileLink

# FileLink(r'train_DHWeight_onDacon.csv')
_, ax = plt.subplots(2,2, figsize=(20,15)) 



a=sns.distplot(test.isnull().mean(axis=0), ax=ax[0,0]) #나열된 값을 distplot을 이용해 시각화 하고, 이를 첫 번째 그래프 창에 넣습니다.

a.set(ylim=(0, 30))

ax[0,0].set_title('Distribution of Missing Values Percentage in Test set')



b=sns.distplot(changed_test.isnull().mean(axis=0), ax=ax[0,1]) #test data에서의 결측치 비율을 시각화 하고, 이를 두 번째 그래프 창에 넣습니다.

b.set(ylim=(0, 30))

ax[0,1].set_title('Distribution of Missing Values Percentage in Test set after Missing value processing')



a=sns.distplot(train.isnull().mean(axis=0), ax=ax[1,0]) #나열된 값을 distplot을 이용해 시각화 하고, 이를 첫 번째 그래프 창에 넣습니다.

a.set(ylim=(0, 30))

ax[1,0].set_title('Distribution of Missing Values Percentage in TRAIN set')



b=sns.distplot(changed_train.isnull().mean(axis=0), ax=ax[1,1]) #test data에서의 결측치 비율을 시각화 하고, 이를 두 번째 그래프 창에 넣습니다.

b.set(ylim=(0, 30))

ax[1,1].set_title('Distribution of Missing Values Percentage in TRAIN set after Missing value processing')

plt.show()
#!pip install pmdarima
from statsmodels.tsa.arima_model import ARIMA # ARIMA 모델

import pmdarima as pm

from pmdarima.arima import auto_arima

# from pmdarima.datasets import load_wineind

import warnings

warnings.filterwarnings('ignore')
def dfPrep(df):

    time = []; place_id = []; target = []  

#     df['Time'] = pd.to_datetime(df['Time']) 

#     df = df.set_index('Time')

    for i in df.columns:

        for j in range(len(df)):

            place_id.append(i)

            time.append(df.index[j])

            target.append(df[i].iloc[j])

    df2 = pd.DataFrame({'place_id': place_id, 'time':time, 'target': target})

    df2 = df2.dropna().set_index('time')

    return df2
changed_train['Time'] = pd.to_datetime(changed_train['Time'])

changed_train=changed_train.set_index('Time')

df2 = dfPrep(changed_train)

df2.head()
def get_optimal_params(y):

    p = d = q = range(0, 2)

    pdq = list(itertools.product(p, d, q))

    param_dict = {}

    for param in pdq:

        try:

            model = ARIMA(y, order=param)

            results_ARIMA = model.fit(disp=-1)

            param_dict[results_ARIMA.aic] = param

        except:

            continue



    min_aic = min(param_dict.keys())

    optimal_params = param_dict[min_aic]

    return optimal_params
def runArima(df):

    df2 = dfPrep(df)

    agg={}

    n = 0

    hr_24 = [str(i+1) for i in range(24)] ; hr_7d = [str(i+1) for i in range(170)]

    hrs = hr_24 ;   hrs.extend(hr_7d)

    for key in df2['place_id'].unique(): # 미터ID리스트를 unique()함수로 추출, for loop

        n+=1   

        temp = df2.loc[df2['place_id']==key] # new_test2에서 key와 일치하는 place_id를 가지는 부분을 temp에 할당

#         temp_1h=temp.resample('1h').sum() # 1시간 단위 resampling(일종의 timeseries압축)



#         # 1 시간별 예측

#         model = ARIMA(temp_1h['target'],  # target 을 추측하고자 함

#                       order=get_optimal_params(temp_1h['target'])) # AIC를 최소화하는 최적의 파라미터 

        model = auto_arima(temp['target'].values, 

                  error_action='ignore',supress_warnings=True,

                  seasonal=True, m=12)

    

        print(n, key, 'predict begins')

        fcst1 = model.predict(24) # 24시간 예측

        fcst2 = model.predict(192) # 168(7days) + 24hr 예측

        a = pd.DataFrame() # a: 예측값을 담을 데이터프레임 생성



        for i in range(24):

            a[str(i+1)]=[fcst1[i]] # column명 지정 및 예측값 대입

        for i in range(192):

            a[str(i+1)]=[fcst2[i]]

            

        # a 에 meter_id를 현재 예측하고 있는 열의 id(key)로 대입

        a['meter_id'] = key 



        # agg{ 미터ID: 시간별(subimssion.columns는 예측시간칼럼들) a의 예측값}

        agg[key] = a[hrs]

    print('---- Modeling Done ----')

    return agg
train_after900 = changed_train.iloc[:,900:1292]
runArima(train_after900)
aggTrain = runArima(changed_train)
output1 = pd.concat(aggTrain, ignore_index=False)

output2 = output1.T

output2.columns = output2.columns.droplevel(level = 1)

hrsTrain_pred = output2
hrsTrain_pred.to_csv("TRAIN_DHWonDacon_24시간192시간예측.csv")

from IPython.display import FileLink

FileLink(r"TRAIN_DHWonDacon_24시간192시간예측.csv")
aggTest = runArima(changed_test)
output1 = pd.concat(aggTest, ignore_index=False)

output2 = output1.T

output2.columns = output2.columns.droplevel(level = 1)

hrsTest_pred = output2
hrsTest_pred.to_csv("TEST_DHWonDacon_24시간192시간예측.csv")

from IPython.display import FileLink

FileLink(r"TSET_DHWonDacon_24시간192시간예측.csv")