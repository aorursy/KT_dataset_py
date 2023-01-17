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
import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from matplotlib import font_manager, rc

import datetime
pollution = pd.read_csv('/kaggle/input/seoulairreport/SeoulHourlyAvgAirPollution.csv')

pollution.columns = ['date','state','co2','ozone','co','so2','pm10','pm2.5']

pollution.head(3)
pollution.shape
pollution.info()
pollution=pollution.fillna(method='ffill')
sns.pairplot(pollution)
pm10_score=pollution.groupby(pollution.state)['pm10'].mean()

pm10_score=pm10_score.sort_values(ascending=False)

pm2_5_score = pollution.groupby(pollution.state)['pm2.5'].mean()

pm2_5_score = pm2_5_score.sort_values(ascending=False)
rc('font',family='AppleGothic')

plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(10,6))

plt.subplot(2,1,1) # 2행 1열로 나타낼거고 지금은 그중에 1째 그래프를 그리겠다.

plt.title('미세먼지')

plt.xticks(rotation=45)

plt.bar(pm10_score.index,pm10_score.values,color='gold')

plt.subplot(2,1,2)

plt.title('초미세먼지')

plt.xticks(rotation=45)

plt.bar(pm2_5_score.index,pm2_5_score.values,color='brown')

plt.tight_layout()
pm10_state=pollution.groupby(pollution.state)['pm10'].mean()

pm10_state[pm10_state.index=='양천구']
pm10_t = pollution.pm10

t_alpha, p_value= stats.ttest_1samp(pollution.pm10,35)

print('t statistic : %.3f \np-value : %.10f' % (t_alpha,p_value))
pollution.date=pollution.date.astype('str')

pollution.date=pollution.date.str[:-4]

date_obj=pd.to_datetime(pollution.date,yearfirst=True)

#pollution.date.strptime(date_string,'%Y-%m-%d')

pollution['date_obj'] = date_obj

print(f'최대기간:{date_obj.max()},최소기간:{date_obj.min()}')

print(f'관측기간:{date_obj.max()-date_obj.min()}')
plt.figure(figsize=(9,7))

sns.lineplot(pollution.date_obj,pollution.pm10,label='미세먼지')

plt.xticks(rotation=45)

sns.lineplot(pollution.date_obj,pollution['pm2.5'],label='초미세먼지')
corr = pollution.drop(['date','date_obj'],axis=1).corr()

corr