from IPython.core.display import display, HTML 

display(HTML("<style>.container { width:100% !important; font-size:14px;}</style>")) 



import time

import warnings; warnings.filterwarnings("ignore")



import pandas as pd

import numpy as np

from scipy import interpolate
train = pd.read_csv('../input/covid19-global-forecasting-week-5/train.csv')

print(train.shape)

train.head()
print('# of days:', train.Date.nunique())
print('# of countries:', train.Date.value_counts()[0] / 2)
print('start date:', train.iloc[0].Date)

print('end date:  ', train.iloc[-1].Date)
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

test['TargetValue'] = np.nan

print(test.shape)

test.head()
print('start date:', test.iloc[0].Date)

print('end date:  ', test.iloc[-1].Date)
# Public Leaderboard Period: 2020-04-27 - 2020-05-11

# Private Leaderboard Period: 2020-05-13 - 2020-06-10
sub = pd.read_csv('../input/covid19-global-forecasting-week-5/submission.csv')

print(sub.shape)

sub.head()
train = train[train.Date < '2020-05-11']

test = test[test.Date > '2020-05-10']



df = pd.concat([train, test])



df.Date = pd.to_datetime(df.Date)



df['geography'] = df.Country_Region.fillna('') + '_' + df.Province_State.fillna('') + '_' + df.County.fillna('')

df.geography.value_counts()
test_ids = []

pred = []



def predict(key):

    print(key)

    t1 = time.time()

    for i,(index,_df) in enumerate(df[df.Target == key].groupby('geography')):

        

        _train = _df[~_df.TargetValue.isnull()]

        _test = _df[_df.TargetValue.isnull()]



        print('\r[%ds] %d %s                              ' % (time.time()-t1, i, index), end='')

        span = 7



        span1 = _train.TargetValue.iloc[-span:].values

        span2 = _train.TargetValue.iloc[-span*2:-span].values



        mean1 = span1.mean()

        mean2 = span2.mean()

        ratio =  (mean1 / mean2)

        if np.isnan(ratio) or np.isinf(ratio):

            ratio = 1.0



        means = [mean2, mean1]

        x = [-10, -3]

        for i in range(7):

            if mean1 == 0 or mean2 == 0:

                means.append(0)

            else:

                means.append(means[-1]*ratio)

            x.append(x[-1]+7)

            ratio = ratio + (1-ratio)/2

            ratio = max(min(ratio, 1.5), 0.5)

    

        f = interpolate.interp1d(x, means, 'linear')

        for i,id in enumerate(_test.ForecastId):

            test_ids.append(int(id))

            pred.append(max(f(i), 0))



    print()



            

predict('ConfirmedCases')

predict('Fatalities')
test = pd.read_csv('../input/covid19-global-forecasting-week-5/test.csv')

rest = np.array(test[~test.ForecastId.isin(test_ids)].ForecastId)



ids_all = np.concatenate([test_ids, rest])

pred_all = np.concatenate([pred, np.zeros(len(rest))])



ids_all,pred_all = map(np.array, zip(*sorted(zip(ids_all, pred_all), key=lambda x: int(x[0]))))

ids_all = np.array(ids_all).astype('str').astype(object)



df_pred_q05 = pd.DataFrame({"ForecastId_Quantile": ids_all + "_0.05", "TargetValue": 0.7 * pred_all})

df_pred_q50 = pd.DataFrame({"ForecastId_Quantile": ids_all + "_0.5", "TargetValue": pred_all})

df_pred_q95 = pd.DataFrame({"ForecastId_Quantile": ids_all + "_0.95", "TargetValue": 1.3 * pred_all})



df_submit = pd.concat([df_pred_q05, df_pred_q50, df_pred_q95])

df_submit.to_csv('submission.csv', index=False)

df_submit
italy = df[(df.Country_Region == 'Italy') & (~df.ForecastId.isnull())].ForecastId.values.astype(int)

pred_all[np.where(np.isin(ids_all.astype(int), italy))][::2]
germany = df[(df.Country_Region == 'Germany') & (~df.ForecastId.isnull())].ForecastId.values.astype(int)

pred_all[np.where(np.isin(ids_all.astype(int), germany))][::2]
np.array(pred_all).max(), np.array(pred_all).min()
df[df.ForecastId == int(ids_all[np.argmax(pred_all)])]