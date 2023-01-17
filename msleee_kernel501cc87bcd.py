# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime as dt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# 데이터 분석 라이브러리
import numpy as np
import pandas as pd 
import datetime as dt

# 시각화 라이브러리
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns
%matplotlib inline

# 시계열 분석 라이브러리
from fbprophet import Prophet

# 파이프라인 라이브러리
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
# 데이터 불러오기
DATA_PATH = "../input/avocado-prices/avocado.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

avocado = load_data()
avocado.head()
# 불필요한 열 삭제
avocado = avocado.drop(['Unnamed: 0', 'Total Bags', 'Small Bags', 'Large Bags', 'XLarge Bags'], 1)
# 열 이름 재설정
names = ["date", "average_price", "total_volume", "small", "large", "xlarge", "type", "year", "region"]
avocado = avocado.rename(columns=dict(zip(avocado.columns, names)))

avocado.head()
# 데이터 형식 살펴보기
avocado.info()
# float 타입 데이터 살펴보기 
avocado.describe()
# object 타입 데이터 살펴보기
avocado["type"].value_counts()
avocado["region"].value_counts()
# date 열의 데이터가 datatime 타입이 아니기 때문에, 날짜를 년, 월, 일로 나누기
dates = [dt.datetime.strptime(ts, "%Y-%m-%d") for ts in avocado['date']]
dates.sort()
sorted_dates = [dt.datetime.strftime(ts, "%Y-%m-%d") for ts in dates]
avocado['date'] = pd.DataFrame({'date':sorted_dates})
avocado['year'], avocado['month'], avocado['day'] = avocado['date'].str.split('-').str
avocado.head()
# 모든 float 타입 데이터 시각화하여 살펴보기
avocado.hist(bins=50, figsize=(10, 10))
plt.show()
# 평균 가격 데이터만 시각화하여 살펴보기
plt.figure(figsize=(15,5))
ax = sns.distplot(avocado["average_price"])
# 특성 사이의 상관관계 확인
from pandas.plotting import scatter_matrix
attributes = ["average_price", "total_volume", "small", "large", "xlarge"]
scatter_matrix(avocado[attributes], figsize=(12, 8))
# 날짜에 따른 가격 분포 살펴보기
price_date=avocado.groupby('date').mean()
plt.figure(figsize=(15,5))
price_date['average_price'].plot(x=avocado.date)
plt.title('Average Price')
# 년도에 따른 가격 분포 살펴보기
price_year=avocado.groupby('year').mean()
fig, ax = plt.subplots(figsize=(15,5))
price_year['average_price'].plot(x=avocado. year)
plt.title('Average Price by Year')
# 월별 가격 분포 살펴보기
price_month=avocado.groupby('month').mean()
fig, ax = plt.subplots(figsize=(15,5))
price_month['average_price'].plot(x=avocado.month)
plt.title('Average Price by Month')
# 일별 가격 분포 살펴보기
price_day=avocado.groupby('day').mean()
fig, ax = plt.subplots(figsize=(15,5))
price_day['average_price'].plot(x=avocado.day)
plt.title('Average Price by Day')
# 지방에 따른 연도별 아보카도 평균 가격 살펴보기
plt.figure(figsize=(30,15))
sns.pointplot(x='average_price', y='region', data=avocado, hue='year',join=False)
plt.xlabel('Region')
plt.ylabel('Average Price')
plt.title("Average Price by Year in Each Region")
# 시계열 분석을 위한 데이터 프레임 만들기
df_avocado = avocado.loc[:, ["date","average_price"]]
df_avocado['date'] = pd.DatetimeIndex(df_avocado['date'])

# 시계열 분석을 위한 열 이름 바꾸기
df_avocado = df_avocado.rename(columns={'date': 'ds', 'average_price': 'y'})
df_avocado.head()
df_avocado.dtypes
# 훈련 데이터와 테스트 데이터 나누기
n_data = 1296
train_avocado = df_avocado[:-n_data]
test_avocado = df_avocado[-n_data:]
print(len(train_avocado), "train +", len(test_avocado), "test")
test_avocado.head()
# 계절성 고려
train_avocado['cap'] = train_avocado.y.max()
train_avocado['floor'] = train_avocado.y.min()

time_model = Prophet(growth='logistic', interval_width=0.95)
time_model.add_seasonality(name='monthly', period=30.5, fourier_order=1)
time_model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)
time_model.add_seasonality(name='yearly', period=365.25, fourier_order=10)

time_model.fit(train_avocado)
# 미래 가격 예측
future_dates = time_model.make_future_dataframe(periods=12, freq='w')
future_dates['cap'] = train_avocado.y.max()
future_dates['floor'] = train_avocado.y.min()

forecast = time_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# ds: the datestamp of the forecasted value
# yhat: the forecasted value of metric
# yhat_lower: the lower bound of forecasts
# yhat_upper: the upper bound of forecasts
fig = time_model.plot(forecast)
fig = time_model.plot_components(forecast)
# 예측된 그래프 살펴보기
forecast_copy = forecast['ds']
forecast_copy2 = forecast['yhat']
forecast_copy = pd.concat([forecast_copy,forecast_copy2], axis=1)

mask = (forecast_copy['ds'] > "2018-01-07") & (forecast_copy['ds'] <= "2018-03-25")
forecasted_values = forecast_copy.loc[mask]
mask = (forecast_copy['ds'] > "2015-01-04") & (forecast_copy['ds'] <= "2018-01-07")
forecast_copy = forecast_copy.loc[mask]

fig, ax1 = plt.subplots(figsize=(16, 8))
ax1.plot(forecast_copy.set_index('ds'), color='b')
ax1.plot(forecasted_values.set_index('ds'), color='r')
ax1.set_ylabel('Average Prices')
ax1.set_xlabel('Date')
print("Red = Predicted Values, Blue = Base Values")
# 평가를 위해 테스트 테이터 가져오기
test_avocado = pd.concat([test_avocado.set_index('ds'),forecast.set_index('ds')], axis=1, join='inner')

columns = ['y', 'yhat', 'yhat_lower', 'yhat_upper']
test_avocado = test_avocado[columns]

test_avocado.head()
# 정확성 측정
test_avocado['e'] = test_avocado.y - test_avocado.yhat
rmse = np.sqrt(np.mean(test_avocado.e**2)).round(2)
mape = np.round(np.mean(np.abs(100*test_avocado.e/test_avocado.y)), 0)
print('RMSE = $', rmse)
print('MAPE =', mape, '%')