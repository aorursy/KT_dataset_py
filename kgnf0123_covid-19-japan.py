# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import statsmodels.api as sm
import requests
import io
from matplotlib import pylab as plt
%matplotlib inline
# グラフを横長にする
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 6
covid_jpn_total = pd.read_csv('../input/covid19-dataset-in-japan/covid_jpn_total.csv')
covid_jpn_metadata = pd.read_csv('../input/covid19-dataset-in-japan/covid_jpn_metadata.csv')
covid_jpn_prefecture = pd.read_csv('../input/covid19-dataset-in-japan/covid_jpn_prefecture.csv')
#appleのmobility情報
applemobility = pd.read_csv('../input/applemobility/applemobilitytrends-2020-04-17.csv')
print(covid_jpn_total.shape)
covid_jpn_total.head(10)
print(covid_jpn_metadata.shape)
covid_jpn_metadata.head(10)
print(covid_jpn_prefecture.shape)
covid_jpn_prefecture.head(10)
print(applemobility[applemobility['region'] == 'Japan'].shape)
applemobility_japan = applemobility[applemobility['region'] == 'Japan']
applemobility_japan = applemobility_japan.set_index('transportation_type')
#ちゃんと見てないけど1/13を100とした時の移動量を表している
applemobility_japan
# 感染者累計・全国
date_positive_all = covid_jpn_prefecture.groupby('Date').sum()
date_positive_all
# 感染者・日別増加人数・全国
positive_days = date_positive_all['Positive'] - date_positive_all.shift(1)['Positive']
positive_days
# 感染者・日別増加人数・県ごと
date_positive_prefecture = covid_jpn_prefecture['Positive'] - covid_jpn_prefecture.shift(47)['Positive']
covid_jpn_prefecture['Positive_days'] = date_positive_prefecture
covid_jpn_prefecture.plot(x='Date', y='Positive_days')
# 行列変換（行を日毎に）
applemobility_japan_stack = applemobility_japan.T
applemobility_japan_stack['Date'] = applemobility_japan_stack.index
print(applemobility_japan_stack.shape)
applemobility_japan_stack
date_positive_all['Positive_days'] = positive_days
date_positive_and_mobility = date_positive_all.merge(applemobility_japan_stack, on='Date', how='left')
date_positive_and_mobility
fig, ax = plt.subplots()
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
ax.grid()
x = date_positive_and_mobility['Date']
add_plot = lambda attr: ax.plot(x, date_positive_and_mobility[attr])
add_plot('Positive_days')
add_plot('transit')
add_plot('walking')
from fbprophet import Prophet
date_positive_and_mobility_fbp = date_positive_and_mobility.copy()
date_positive_and_mobility_fbp.drop({'Tested', 'Discharged', 'Fatal'}, axis=1, inplace=True)
date_positive_and_mobility_fbp = date_positive_and_mobility_fbp.rename(columns={'Date': 'ds', 'Positive_days': 'y'})
#date_positive_and_mobility_fbp['cap'] = 3000000
date_positive_and_mobility_fbp.fillna(0, inplace=True)
date_positive_and_mobility_fbp
m = Prophet()
m.fit(date_positive_and_mobility_fbp)
future = m.make_future_dataframe(periods=12,freq='M')
#future['cap'] = 3000000
forecast = m.predict(future)
m.plot(forecast)

