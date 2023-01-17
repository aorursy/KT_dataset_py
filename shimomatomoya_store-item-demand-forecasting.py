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
sub_df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/sample_submission.csv')

train_df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/train.csv')

test_df = pd.read_csv('/kaggle/input/demand-forecasting-kernels-only/test.csv')
#10店舗xアイテム50個

print(len(test_df.drop_duplicates(['store','item'])))



#90日

print(len(test_df.drop_duplicates('date')))
train_df.isnull().sum()
#2014-01-15 store 6 item 4 がsales0以外は売上が立っている

train_df.sort_values('sales').head(10)
#10店舗50アイテムで欠損日はなし。

print(len(train_df.drop_duplicates('date')) * len(train_df.drop_duplicates(['store','item'])))



print(len(train_df))
train_check = train_df[train_df['item'] == 1][train_df[train_df['item'] == 1]['store'] == 1]

train_check
import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('seaborn-darkgrid')

plt.rcParams['font.family'] = 'Yu Gothic'
plt.figure(figsize=(300,15))

sns.lineplot(data=train_check, x='date', y='sales')

plt.xticks(rotation=270)

plt.show()
#週の周期性を確認

import statsmodels.api as sm

import matplotlib as mpl

res_week = sm.tsa.seasonal_decompose(train_check['sales'].values, period=7, model='multipilicative', two_sided=False)

with mpl.rc_context():

    mpl.rc('figure', figsize=(16,8))

    res_week.plot()
#月の周期性を確認

res_month = sm.tsa.seasonal_decompose(res_week.trend[~np.isnan(res_week.trend)], period=30, model='multiplicative', two_sided=False)

with mpl.rc_context():

    mpl.rc('figure', figsize=(16, 8))

    res_month.plot()
#年の周期性を確認

#他にも隠れた周期がありそう

res_year = sm.tsa.seasonal_decompose(res_month.trend[~np.isnan(res_month.trend)], period=365, model='multiplicative', two_sided=False)

with mpl.rc_context():

    mpl.rc('figure', figsize=(16, 8))

    res_year.plot()
#年の周期性を確認

#他にも隠れた周期がありそう

res_q_year = sm.tsa.seasonal_decompose(res_month.trend[~np.isnan(res_month.trend)], period=round(365 / 4), model='multiplicative', two_sided=False)

with mpl.rc_context():

    mpl.rc('figure', figsize=(16, 8))

    res_q_year.plot()
res_year_1 = sm.tsa.seasonal_decompose(res_q_year.trend[~np.isnan(res_q_year.trend)], period=365, model='multiplicative', two_sided=False)

with mpl.rc_context():

    mpl.rc('figure', figsize=(16, 8))

    res_year_1.plot()
fig,ax = plt.subplots(2,1, figsize=(16,4))

ax[0].plot(res_year_1.resid) #365 / 4の周期性も除いた誤差

ax[1].plot(res_year.resid) # 除いていない誤差

plt.show()
#対数変換

train_check['sales_log'] = train_check['sales'].apply(lambda x: np.log(x + 0.5))

fig, ax = plt.subplots(1, 2, figsize=(16,8))

sns.distplot(np.array(train_check['sales']), ax=ax[0] )

ax[0].set_title('sales')



sns.distplot(np.array(train_check['sales_log']), ax=ax[1])

ax[1].set_title('sales_log')



plt.show()
from scipy.stats import shapiro

print(shapiro(np.array(train_check['sales'])), 'そのまま')

print(shapiro(np.array(train_check['sales_log'])), '対数変換')
import datetime

#乗法回帰したいので、0をなくすために対数変換

train_df['sales_log'] = train_df['sales'].apply(lambda x: np.log(x + 0.5))

train_df['date_1'] = train_df['date'].apply(lambda x: datetime.datetime.strptime(x,'%Y-%m-%d'))



#ストアx商品のDFを作る

train_list_df = train_df.drop_duplicates(['store','item'])[['store','item']].reset_index(drop=True)
print(train_list_df)

print(train_df)
from fbprophet import Prophet

import tqdm

import math
train_df_slice = []

for i in range(len(train_list_df)):

    train_df_slice.append(train_df[train_df['store'] == train_list_df.at[i, 'store']][train_df[train_df['store'] == train_list_df.at[i, 'store']]['item'] == train_list_df.at[i, 'item']].reset_index(drop=True))

    

#train_df_sliceには500個の各日DFが入る

forecast_slice = []

for i in tqdm.tqdm(range(len(train_df_slice))):

    df = train_df_slice[i][['date_1','sales_log']].rename(columns={'date_1':'ds','sales_log':'y'})

    

    model = Prophet(seasonality_mode='multiplicative', weekly_seasonality=True, yearly_seasonality=True,

                   daily_seasonality=False)

    model.add_seasonality(name='monthly', period=30, fourier_order=5, mode='multiplicative')

    model.add_seasonality(name='quarterly', period = round(365 / 4),  fourier_order=7, mode='multiplicative')

    model.fit(df)

    

    future = model.make_future_dataframe(periods = len(test_df.drop_duplicates('date')) , freq = 'D') #90

    forecast = model.predict(future)

    forecast_slice.append(forecast[['ds','yhat']]) 
#train_list_dfの順番に予測結果が入っているので合わせる

for i in tqdm.tqdm(range(len(forecast_slice))):

    forecast_slice[i]['store'] = train_list_df.at[i,'store'] #カラムをつける

    forecast_slice[i]['item'] = train_list_df.at[i, 'item']

    



forecast_total_df = pd.concat([ forecast_slice[d] for d in range(len(forecast_slice))], axis=0)

#total_であとでプロットしてみる
forecast_total = forecast_total_df[forecast_total_df['ds'] >= datetime.datetime(2018,1,1)]

#カラムを元に戻す



forecast_total['sales'] = forecast_total['yhat'].apply(lambda x : math.exp(x) - 0.5)

forecast_total['date'] = forecast_total['ds'].apply(lambda x:datetime.datetime.strftime(x, '%Y-%m-%d'))

forecast_total
from decimal import Decimal, ROUND_HALF_UP

submit_df = pd.merge(test_df, forecast_total, how='inner', left_on=['date','store','item'], right_on=['date','store','item'])

submit_df['sales'] = submit_df['sales'].apply(lambda x: int(Decimal(x).quantize(Decimal('1'), rounding=ROUND_HALF_UP )))
#ちょっと順番違うけど比較

yhat_check = forecast_total_df[forecast_total_df['store'] == 1][forecast_total_df[forecast_total_df['store'] == 1]['item'] == 1]

yhat_check['sales'] = yhat_check['yhat'].apply(lambda x : math.exp(x) - 0.5)

yhat_check['sales'] = yhat_check['sales'].apply(lambda x : int(Decimal(x).quantize(Decimal('1'), rounding=ROUND_HALF_UP )))
plt.figure(figsize=(300, 15))

sns.lineplot(data=pd.melt(pd.merge(train_check, yhat_check, how='inner', left_index=True, right_index=True), id_vars='date', value_vars=['sales_x','sales_y']),

             x='date', y='value', hue='variable')

plt.xticks(rotation=270)

plt.show()
submit_df
submit_df[['id','sales']].to_csv('submission.csv', index=False)