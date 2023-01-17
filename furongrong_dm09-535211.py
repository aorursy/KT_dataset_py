import os



raw_path = "./dataset"

if not os.path.exists(raw_path):

    os.mkdir(raw_path)
import requests

import pandas as pd

import json



# 用于爬取的不同指标的代号

indicators = {'ST.INT.RCPT.XP.ZS':'国际旅游收入（占总出口的百分比）',

              'ST.INT.TVLR.CD':'国际旅游收入（现价美元）', 

              'ST.INT.DPRT':'国际旅游离境人数',

              'ST.INT.ARVL':'国际旅游入境人数',

              'ST.INT.XPND.CD':'国际旅游支出（现价美元）',

              'ST.INT.XPND.MP.ZS':'国际旅游支出（占总进口的百分比）'}



for i in indicators.keys():

  # max_page初始设置成3，爬取到页面后更新max_page

  page, max_page = 1, 3

  # 为每个指标建立一个新的dataframe

  df_country = pd.DataFrame()



  while page <= max_page:

    try:

      url = f'https://api.worldbank.org/v2/country/indicator/{i}?format=json&page={page}'

      s = requests.Session()

      r = s.get(url)

      

      if r.status_code==200:

        # 从xml页面上的参数得到更新后的max_page和数据

        max_page = r.json()[0]['pages']

        data = r.json()[1]

#         print(f'Fetching {indicators[i]} page {page}....')

        

        # 把json的层级化结构打平

        df = pd.io.json.json_normalize(data, max_level=1)

        df_country = pd.concat([df_country, df])

        page += 1

    except requests.exceptions.Timeout:

      print('Time out: ', url)

      break



  filename = f'./dataset/{indicators[i]}.csv'

  df_country.to_csv(filename, index=None, encoding='utf-8')
import pandas as pd

import os



path = "./clean_dateset"

if not os.path.exists(path):

    os.mkdir(path)
import numpy as np



raw_path = "./dataset"

df_income_group = pd.read_csv('../input/dateset/country_en.csv', index_col='Country Name')

# 创建国家代码dataframe列表

country_code = df_income_group[['Country Code']]

# 创建非国家组织的dataframe列表

orga = df_income_group.loc[df_income_group.IncomeGroup.isnull()]

orga.index
# 原始csv数据展示

df_country.head()
import pandas as pd

df_in_m = pd.read_csv(f'{raw_path}/国际旅游收入（现价美元）.csv', index_col='country.value')

df_out_m = pd.read_csv(f'{raw_path}/国际旅游支出（现价美元）.csv', index_col='country.value')



df_in_ppl = pd.read_csv(f'{raw_path}/国际旅游入境人数.csv', index_col='country.value')

df_out_ppl = pd.read_csv(f'{raw_path}/国际旅游离境人数.csv', index_col='country.value')



df_in_percent = pd.read_csv(f'{raw_path}/国际旅游收入（占总出口的百分比）.csv', index_col='country.value')

df_out_percent = pd.read_csv(f'{raw_path}/国际旅游支出（占总进口的百分比）.csv', index_col='country.value')



df_dict = {'国际旅游收入': df_in_m, '国际旅游支出': df_out_m, 

           '国际旅游入境人数':df_in_ppl, '国际旅游离境人数':df_out_ppl,

           '国际旅游收入(百分比)': df_in_percent, '国际旅游支出(百分比)': df_out_percent}



for key in list(df_dict.keys()):

  df = pd.DataFrame(df_dict[key][['date', 'value']])

  df_final = pd.DataFrame()

  # 每一组是一个国家，k是国家名，v为数据

  for k, v in df.groupby('country.value', as_index=False, sort=False):

    v.set_index('date', inplace=True)

    v.columns = [k]

    df_final = pd.concat([df_final, v], axis=1)

  #转置使原本为列的时间轴变为行

  df_dict[key] = df_final.transpose()

  # 移除1960到1994年和2018到2019年的空数据

  df_dict[key] = df_dict[key].drop(columns=[int(col) for col in range(1960, 1995)])

  df_dict[key] = df_dict[key].drop(columns=[int(col) for col in range(2018, 2020)])



  #清洗后的数据存入文件夹clean_dataset备用

  filename = f'{path}/{key}.csv'

  df_dict[key].to_csv(filename, encoding='utf-8')
# 处理后的dataframe

df_dict[key].head()
year = 2017

# 计算净收入

df_net = df_dict['国际旅游收入'][year] - df_dict['国际旅游支出'][year]

# 排除其中的非国家

df_net = df_net.loc[[i for i in df_net.index if i not in orga.index]]

df_net = df_net.sort_values(ascending=False)



# 加入国家或地区代码方便画图

df_net = pd.DataFrame(df_net).join(country_code, how='left')

df_net.head()
import plotly.graph_objects as go

import pandas as pd



df = df_net



# 以下为2017年各国的国际旅游业的净收入可视化对比图

fig = go.Figure(data=go.Choropleth(

    locations = df['Country Code'],

    z = df[year],

    text = df.index,

    colorscale = 'mint',

    autocolorscale=False,

    reversescale=True,

    marker_line_color='darkgray',

    marker_line_width=0.1,

    colorbar_tickprefix = '$',

    colorbar_title = 'Tourism Net Value<br>Billion US$',

    colorbar_x = -0.05

))



fig.update_layout(

    title_text='2017 International tourism net receipts',

    geo=dict(

        showframe=False,

        showcoastlines=False,

        projection_type='equirectangular'

    ),

    annotations = [dict(

        x=0.8,

        y=0.1,

        text='Source: World Bank',

        showarrow = False

    )]

)



fig.show()
import matplotlib.pyplot as plt



# 计算该国平均每位入境游客的消费（美元）

df_income_ppl = df_dict['国际旅游收入'] / df_dict['国际旅游入境人数']

# 计算该国平均每位离境游客的消费（美元）

df_xpen_ppl = df_dict['国际旅游支出'] / df_dict['国际旅游离境人数']



fig = plt.figure(figsize = (8,12))

plt.subplot(2,1,1)

df_income_ppl_top10 = df_income_ppl[2017].sort_values(ascending=False).iloc[:10]

plt.barh(df_income_ppl_top10.index, df_income_ppl_top10)

plt.xlim(0, 6000)

plt.title('Average Tourism Income from per arrival')

plt.xlabel('US$')

for i, v in enumerate(df_income_ppl_top10):

    plt.text(v + 3, i + 0.0, str(round(v,2)), color='black')



plt.subplot(2,1,2)

df_xpen_ppl_top10 = df_xpen_ppl[2017].sort_values(ascending=False).iloc[:10]

plt.barh(df_xpen_ppl_top10.index, df_xpen_ppl_top10, color='#CBA6C3')

plt.xlim(0, 5000)

plt.title('Average Tourism Expenditure from per departure')

plt.xlabel('US$')

for i, v in enumerate(df_xpen_ppl_top10):

    plt.text(v + 3, i + 0.0, str(round(v,2)), color='black')

df_xpen_ppl_top10.head()
# 排除非国家 & 降序排序

df_income_percent = df_dict['国际旅游收入(百分比)'][2017].loc[[i for i in df_net.index if i not in orga.index]].sort_values(ascending=False)[:10]

df_expend_percent = df_dict['国际旅游支出(百分比)'][2017].loc[[i for i in df_net.index if i not in orga.index]].sort_values(ascending=False)[:10]
import matplotlib.pyplot as plt

from matplotlib import cm

from math import log10



to_draw = [df_income_percent, df_expend_percent]



fig,ax = plt.subplots(1,2,figsize = (18,4))

for a, df in enumerate(to_draw):



  labels = to_draw[a].index

  data = to_draw[a]

  

  # 环数为top国家数

  n = len(data)

  # 完整一圈的值为100%

  m = 100

  # 最大半径为2

  r = 2

  # 每一环的宽度为半径/环数

  w = r / n 

  #设置每一环的不同颜色

  colors = [cm.coolwarm(i / n) for i in range(n)]

  

  for i in range(n):

      # 每一环的左半圈设置label为空

      innerring, _ = ax[a].pie([m - data[i], data[i]], radius = r - i * w, 

                             startangle = 90, labels = ["", labels[i]], 

                             labeldistance = 1 - 1 / (1.5 * (n - i)),

                            # 每一环的左半圈设置为透明

                             textprops = {"alpha": 0}, colors = ["None", colors[i]])

      plt.setp(innerring, width = w, edgecolor = "white")



ax[0].text(-1.9,2.2,'International tourism expenditures (% of total imports)',fontsize=12) 

ax[1].text(-1.8,2.2,'International tourism receipts (% of total exports)', fontsize=12) 

ax[0].legend( loc='upper right', bbox_to_anchor=(2.21, 1.4))

ax[1].legend( loc='upper right', bbox_to_anchor=(1.75, 1.4))
import matplotlib.pyplot as plt

from matplotlib import cm

%matplotlib inline



fig = plt.figure(figsize = (16,15))

plt.subplot(3,2,1)

plt.fill_between(df_dict['国际旅游入境人数'].columns, df_dict['国际旅游入境人数'].loc['China'], color='#85C1E9',

                 alpha=0.5, label='Number of arrivals')

plt.fill_between(df_dict['国际旅游离境人数'].columns, df_dict['国际旅游离境人数'].loc['China'], color='#F7DC6F',

                 alpha=0.5, label='Number of departures')

plt.xlim(1995,2017)

plt.title('International tourism \nnumber of arrivals and departures (China)')

plt.ylabel('Billion')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')



plt.subplot(3,2,3)

plt.fill_between(df_dict['国际旅游入境人数'].columns, df_dict['国际旅游入境人数'].loc['United States'], color='#85C1E9',

                 alpha=0.5, label='Number of arrivals')

plt.fill_between(df_dict['国际旅游离境人数'].columns, df_dict['国际旅游离境人数'].loc['United States'], color='#F7DC6F',

                 alpha=0.5, label='Number of departures')

plt.xlim(1995,2017)

plt.title('International tourism \nnumber of arrivals and departures (North America)')

plt.ylabel('Billion')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')



plt.subplot(3,2,2)

plt.fill_between(df_dict['国际旅游收入'].columns, df_dict['国际旅游收入'].loc['China'], color='#fdc4b6',

                 alpha=0.5, label='Income')

plt.fill_between(df_dict['国际旅游支出'].columns, df_dict['国际旅游支出'].loc['China'], color='#C5C6B6',

                 alpha=0.5, label='Expenditure')

plt.xlim(1995,2017)

plt.title('International tourism income and receipts (China)')

plt.ylabel('Billion US$')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')



plt.subplot(3,2,4)

plt.fill_between(df_dict['国际旅游收入'].columns, df_dict['国际旅游收入'].loc['United States'], color='#fdc4b6',

                 alpha=0.5, label='Income')

plt.fill_between(df_dict['国际旅游支出'].columns, df_dict['国际旅游支出'].loc['United States'], color='#C5C6B6',

                 alpha=0.5, label='Expenditure')

plt.xlim(1995,2017)

plt.title('International tourism income and receipts (North America)')

plt.ylabel('Billion US$')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')



plt.subplot(3,2,5)

plt.fill_between(df_dict['国际旅游入境人数'].columns, df_dict['国际旅游入境人数'].loc['Spain'], color='#85C1E9',

                 alpha=0.5, label='Number of arrivals')

plt.fill_between(df_dict['国际旅游离境人数'].columns, df_dict['国际旅游离境人数'].loc['Spain'], color='#F7DC6F',

                 alpha=0.5, label='Number of departures')

plt.xlim(1995,2017)

plt.title('International tourism \nnumber of arrivals and departures (Spain)')

plt.ylabel('Billion')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')



plt.subplot(3,2,6)

plt.fill_between(df_dict['国际旅游收入'].columns, df_dict['国际旅游收入'].loc['Spain'], color='#fdc4b6',

                 alpha=0.5, label='Income')

plt.fill_between(df_dict['国际旅游支出'].columns, df_dict['国际旅游支出'].loc['Spain'], color='#C5C6B6',

                 alpha=0.5, label='Expenditure')

plt.xlim(1995,2017)

plt.title('International tourism income and receipts (Spain)')

plt.ylabel('Billion US$')

plt.xticks(range(2017,1994,-2))

plt.legend(loc='upper left')
# 计算国际旅游收入的历年增长率

df_dollar_rate_in = (df_dict['国际旅游收入'].shift(periods=1, axis='columns') - df_dict['国际旅游收入']) / df_dict['国际旅游收入'] * 100

df_dollar_rate_in.columns = range(2018,1995,-1)

# 计算国际旅游支出的历年增长率

df_dollar_rate_out = (df_dict['国际旅游支出'].shift(periods=1, axis='columns') - df_dict['国际旅游支出']) / df_dict['国际旅游支出'] * 100

df_dollar_rate_out.columns = range(2018,1995,-1)

# 计算国际旅游入境人数的历年增长率

df_ppl_rate_in = (df_dict['国际旅游入境人数'].shift(periods=1, axis='columns') - df_dict['国际旅游入境人数']) / df_dict['国际旅游入境人数']* 100

df_ppl_rate_in.columns = range(2018,1995,-1)

# 计算国际旅游离境人数的历年增长率

df_ppl_rate_out = (df_dict['国际旅游离境人数'].shift(periods=1, axis='columns') - df_dict['国际旅游离境人数']) / df_dict['国际旅游离境人数'] * 100

df_ppl_rate_out.columns = range(2018,1995,-1)

# 计算平均每位入境游客消费的历年增长率

df_income_ppl_rate =  (df_income_ppl.shift(periods=1, axis='columns') - df_income_ppl) / df_income_ppl * 100

df_income_ppl_rate.columns = range(2018,1995,-1)

# 计算平均每位离境游客消费的历年增长率

df_xpen_ppl_rate =  (df_xpen_ppl.shift(periods=1, axis='columns') - df_xpen_ppl) / df_xpen_ppl * 100

df_xpen_ppl_rate.columns = range(2018,1995,-1)



#画出折线图

fig = plt.figure(figsize=(27,5))

plt.style.use('seaborn-whitegrid')

plt.subplot(1,3,1)

df_dollar_rate_in.loc['China'].plot(label='Growth rate of receipts (China)')

df_dollar_rate_out.loc['China'].plot(label='Growth rate of expenditures (China)')

plt.ylabel('Growth rate (%)')

plt.ylim(top=100, bottom=-30)

plt.xticks(range(2018,1995,-2))

plt.legend()



plt.subplot(1,3,2)

df_ppl_rate_in.loc['China'].plot(colormap='Spectral',label='Growth rate of arrivals (China)')

df_ppl_rate_out.loc['China'].plot(colormap='seismic',label='Growth rate of departures (China)')

plt.legend()

plt.xticks(range(2018,1995,-2))

plt.ylabel('Growth rate (%)')

plt.ylim(top=100, bottom=-30)



plt.subplot(1,3,3)

df_income_ppl_rate.loc['China'].plot(colormap='gray', label='Growth rate of Tourism Income from per billion arrivals (China)')

df_xpen_ppl_rate.loc['China'].plot(color='#4f953b', label='Growth rate of Tourism Expenditure from per billion departures (China)')

plt.ylim(top=100, bottom=-30)

plt.ylabel('Growth rate (%)')

plt.legend()

plt.xticks(range(2018,1995,-2))

from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.sandbox.stats.diagnostic import acorr_ljungbox

from pandas.plotting import autocorrelation_plot

fig, axes = plt.subplots(ncols=3, nrows=1, figsize=(15, 3))

df = df_dict['国际旅游入境人数']



# 将dataframe的index顺序调转为从1995年排到2017年

df = df.loc['China'].sort_index()



#绘制acf自相关图

plot_acf(df, ax=axes[0])

autocorrelation_plot(df, ax=axes[1])

LB2, P2 = acorr_ljungbox(df)

plt.plot(P2)
from statsmodels.tsa.stattools import arma_order_select_ic

import warnings





warnings.filterwarnings('ignore')



# 由于数据量较少只取出3个进行预测

train_data = df[:-3]  

test_data = df[-3:]

# 训练数据

arma_order_select_ic(train_data.diff().dropna(), ic=['aic', 'bic'], trend='nc')['aic_min_order']  #AIC
arma_order_select_ic(train_data, ic='bic')['bic_min_order']  # BIC
arma_order_select_ic(train_data, ic='hqic')['hqic_min_order']  # HQIC
from statsmodels.tsa.arima_model import ARIMA



# 训练模型

arima = ARIMA(train_data, order=(2, 1, 1)).fit() 
import pandas as pd



# 输出后续3个预测结果

plt.plot(arima.forecast(steps=3)[0], '.-', label="predict")  

plt.plot(test_data.values, '.-', label="real")

plt.legend()