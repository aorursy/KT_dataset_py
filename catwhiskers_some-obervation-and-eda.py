# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-1/train.csv')

train_df.head()
grouped_df = train_df.groupby(['Country/Region', 'Date']).sum().reset_index()

grouped_df.head()

last_date = train_df.Date.max()

latest_grouped = grouped_df[grouped_df['Date']== last_date]

latest_grouped.head()
import plotly.express as px



fig = px.choropleth(latest_grouped, locations="Country/Region", 

                    locationmode='country names', color="ConfirmedCases", 

                    hover_name="Country/Region", range_color=[1,5000], 

                    color_continuous_scale="portland", 

                    title='Countries with Confirmed Cases')

fig.show()
import matplotlib.pyplot as plt

import numpy as np

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import cm

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from collections import OrderedDict

def get_color_map(count): 

    viridis = cm.get_cmap('coolwarm', count)

    return viridis(np.linspace(0, 1, count))



def plot_by_country_and_date(grouped_df, region='Country/Region', target='ConfirmedCases'):

    cdate = grouped_df['Date'].unique()

    countries = grouped_df[region].unique()



    colors = get_color_map(len(countries)) 

    plt.figure(figsize=(15,10),dpi=100,linewidth = 2)

    grouped_df = grouped_df.sort_values(by=['Date'])

    

    for i, c in enumerate(countries): 

        cur_Y = grouped_df[grouped_df[region] == c][[target]]

        plt.plot(cdate,cur_Y,'o-',color = colors[i], label=c)

    date_tick = [] 

    for i, d in enumerate(cdate):

        if i % 7 == 0:

            date_tick.append(d)

        

    plt.xticks(date_tick,fontsize=10)

    plt.yticks(fontsize=20)

    plt.xlabel("date", fontsize=20)

    plt.ylabel(target, fontsize=20)

    plt.yscale('log')

    plt.legend(loc = "best", fontsize=10)

    plt.show() 

    

    

latest_grouped= latest_grouped.sort_values(by='ConfirmedCases') 

top_10 = latest_grouped.iloc[-10:,:]['Country/Region'].unique()

botton_10 = latest_grouped.iloc[:10,:]['Country/Region'].unique()

grouped_df['top10'] = grouped_df['Country/Region'].apply(lambda x: x in top_10)

grouped_df['botton10'] = grouped_df['Country/Region'].apply(lambda x: x in botton_10)



data_top_10 = grouped_df[grouped_df['top10']]

plot_by_country_and_date(data_top_10, target='ConfirmedCases')

plot_by_country_and_date(data_top_10, target='Fatalities')
import geopandas as gpd

import math

province_shp = gpd.read_file('/kaggle/input/chinaprovince/gadm36_CHN_1.shp')

china_df = train_df[train_df['Country/Region']=='China']

g_p_china_df = china_df[china_df['Date']==last_date]

g_p_china_df  = china_df.groupby(['Province/State']).sum().reset_index()

g_p_china_df['ConfirmedCasesLog'] = g_p_china_df.ConfirmedCases.apply(lambda x: math.log(x+1))

all_df = province_shp.merge(g_p_china_df, left_on=('NAME_1'), right_on=('Province/State'))

all_df.plot(column='ConfirmedCasesLog', cmap='OrRd')





g_china_df = china_df.groupby(['Province/State','Date']).sum().reset_index()



plot_by_country_and_date(g_china_df, region='Province/State', target='ConfirmedCases')

plot_by_country_and_date(g_china_df, region='Province/State', target='Fatalities')

train_time_feature_config = {

    "ConfirmedCases": 21,

    "Fatalities": 21,

}

from datetime import datetime

date_format = "%Y-%m-%d"

min_date = datetime.strptime(train_df['Date'].min(), date_format)

print(min_date) 



m_train_df = train_df[['Date', 'Province/State', 'Country/Region']]

m_train_df = m_train_df.fillna('')

m_train_df['Loc']=m_train_df['Province/State']+m_train_df['Country/Region']

m_train_df = pd.get_dummies(m_train_df, columns=['Loc'], prefix = ['loc'])

m_train_df['days'] = m_train_df.Date.apply(lambda x: (datetime.strptime(x, date_format) - min_date).days)

m_train_df = pd.concat([m_train_df, train_df[['Lat','Long']]], axis=1)







m_train_df = m_train_df.drop([ 'Province/State', 'Country/Region'], axis=1)



columns = list(m_train_df.columns)

for k, v in train_time_feature_config.items():

    print('key:'+k)

    lag = v

    target = train_df[[k]]

    np_target = target.to_numpy()

    columns.append(k)

    for j in range(1, v):

        columns.append(k+":"+str(j))

        columns.append(k+"diff:"+str(j))

        next_target = target.shift(j)

        next_target = next_target.to_numpy()

        diff_target = target - next_target

        np_target = np.concatenate((np_target, next_target, diff_target), axis=1)

    m_train_df = np.concatenate((m_train_df, np_target), axis=1)



target = train_df['ConfirmedCases'].shift(-1).to_numpy()

m_train_df = np.concatenate((m_train_df, target.reshape(target.shape[0],1)), axis=1)

columns.append('target1d')

m_train_df = pd.DataFrame(data=m_train_df, columns=columns)    

m_train_df = m_train_df.dropna()

m_train_df.head()
from sklearn.model_selection import train_test_split

import xgboost



xgb = xgboost.XGBRegressor(max_depth=7,eval_metric='rmse', reg_lambda=1, num_round=1000)



features = m_train_df.iloc[:,2:-1]

data = np.array(features.iloc[0,:].values, copy=False, dtype=np.float32)

 

X_train, X_test, y_train, y_test = train_test_split(features.values, m_train_df.target1d.values,

                                                    train_size=0.75, test_size=0.25, random_state=42)

xgb.fit(X_train,y_train)

rpredictions = xgb.predict(X_test)

predictions = [] 

for p in rpredictions: 

    if p<0: 

        p=0

    predictions.append(p)
import math 

import matplotlib.pyplot as plt

import numpy as np 



predictions = np.array(predictions)



log_test = np.array(np.log((y_test+1).tolist()))

log_predictions = np.array(np.log((predictions+1).tolist()))







print( 'rmsle')

print( math.sqrt(((log_test - log_predictions)**2).mean()))

importances = [] 

for i, im in enumerate(xgb.feature_importances_):

    importances.append((columns[i],im))

    

importances = list(reversed(sorted(importances, key=lambda x: x[1]))) 

importances = importances[:10]

weight = []

name = []

for i, im in enumerate(importances): 

    weight.append(im[1])

    name.append(im[0])



y_pos = np.arange(len(weight))

plt.figure(figsize=(30,10))

plt.bar(y_pos, weight, align='center', alpha=0.5)

plt.xticks(y_pos, name)

plt.ylabel('feature importance')
