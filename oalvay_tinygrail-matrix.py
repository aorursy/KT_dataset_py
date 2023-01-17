import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import json

import requests

from requests.exceptions import Timeout



from multiprocessing import Pool



import datetime

from datetime import timedelta

import time

import os

import gc



import seaborn as sns

import matplotlib.pyplot as plt

import plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot, init_notebook_mode



df = pd.read_csv('/kaggle/input/tinygrail-reload/tinygrail_reload.csv')



ss = []

count = 0

for i in df.deals:

    temp = eval(i)

    temp2 = df.id[count]

    for j in temp:

        j.append(temp2)

        ss.append(j)

    count+=1

    

df2 = pd.DataFrame(ss, columns = ['Amount','Price','DateTime', 'types', 'chara_id'])

def change_time(i):

    return i[:19].replace('T',' ')

df2['DateTime'] = df2.DateTime.map(change_time).astype("datetime64")#+ timedelta(hours=8)

df2['Total'] = df2['Amount'] * df2['Price']

df2['Date'] = df2.DateTime.dt.date

df2['Time'] = df2.DateTime.dt.time

df2.loc[(df2.types == 'ico')&(df2.Amount<9000), 'types'] = 'normal'

df2.loc[(df2.DateTime.dt.weekday == 6)&(df2.Time == datetime.time(0, 0, 0))&(df2.Date >= datetime.date(2019, 10, 3)),

        'types'] = 'auction'

def est_price(chara_Id, est_amt,df = df2):

    df_to_use = df.loc[(df['chara_id'] == chara_Id)&(df['types'] != 'auction')].sort_values('DateTime', ascending = False)

    range_to_use = range(len(df_to_use))

    last_notexceed_yet = True

    used_amt = 0

    total_value = 0

    for i in range_to_use:

        this_amount = df_to_use.iloc[i]['Amount']

        used_amt += this_amount

        if used_amt < est_amt:

            total_value += this_amount * df_to_use.iloc[i]['Price']

        else:

            total_value += (est_amt - (used_amt - this_amount)) * df_to_use.iloc[i]['Price']

            break

    return round(total_value/est_amt,2)

dfstk = pd.read_csv('/kaggle/input/tinygrail-reload/tinygrail_stk.csv')
def intial_stock(deals):

    """ for first deal in history"""

    return deals[0][0] if (deals[0][0] >=9500) & (deals[0][1] >= 10) else 0

df['icolevel'] = df.deals.map(eval).map(intial_stock)



def intial_stock2(info):

    """ flowing stock"""

    return info['Total']

df['icolevel2'] = df['info'].map(eval).map(intial_stock2)



def intial_price(deals):

    """ for first deal in history"""

    return deals[0][1] if (deals[0][0] >=9500) & (deals[0][1] >= 10) else 0

df['icoprice'] = df.deals.map(eval).map(intial_price)



def intial_date(deals):

    """ for first deal in history"""

    return deals[0][2][:19].replace('T',' ')

df['icodate'] = df.deals.map(eval).map(intial_date).astype("datetime64")





df = pd.merge(dfstk.drop('uid', axis = 1), df, how='outer', on=['id'])

df['icolevel22'] = df.apply(lambda x: x['icolevel2']+x['flows'], axis = 1) ### addup flowing stock and bidding pool
def ico_level(i):

    if (i['icolevel'] <= 10000) & (i['icolevel'] >= 9900):

    ### definitely lv1

        return i['icolevel']

    elif (abs(i['icolevel'] - i['icolevel22']) <= 1000 ):

    ### small difference is fine

        return i['icolevel']

    elif (abs(i['icolevel'] - i['icolevel22']) <= 2500 ) & (i['icolevel'] >= 17000):

    ### a bit larger difference is fine if 1st deal is high

        return i['icolevel']

    elif (i['icolevel'] == 0):

    ### in no recorded case I believe I can trust icolevel22

        return i['icolevel22']

    elif (i['icolevel22'] <= 10000) & (i['icolevel22'] >= 9900):

    ### similar to 1st case

        return i['icolevel22']

    elif (i['icoprice'] >= 20) & (i['icolevel'] >= 17000):

    ### large price large amount is okay

        return i['icolevel']

    else:

    ### the size of remaining case should be small, which therefore is tolerable

        return 10000

df['ico_lvl'] = df.apply(ico_level, axis = 1)

df = df.drop(['icolevel22','icolevel2','icolevel'], axis = 1)

del ico_level

gc.collect()
def classify_lvl(i):

    if i <= 13000:

        return 'level 1'

    elif i <= 20000:

        return 'level 2'

    elif i <= 28000:

        return 'level 3'

    elif i <= 35000:

        return 'level 4'

    else:

        return 'level 5 or above'

df['ico_level'] = df['ico_lvl'].map(classify_lvl)
df['ico_level'].value_counts()
#df.loc[df['ico_level'] == 'level 4']
fig = px.scatter(df, x='ico_lvl', y='icoprice', hover_name = 'id')

# fig.update_layout(

#     title="茶话会年代记",

#     xaxis_title="日期",

#     yaxis_title="楼层数")

fig.show()
# ids = 13314

# prices = 11

# y_ = [[i*10, est_price(ids, i*10)] for i in range(300) if (est_price(ids, i*10)<=prices+0.5)&(est_price(ids, i*10)>=prices-0.5)]
est_price(29282, 2000, df2.loc[(df2.Date <= datetime.date(2019,12,5))])
# df_temp = df2.loc[(df2.DateTime <= datetime.datetime(2019,12,6))]

# notsomeday=([est_price(13314, i*20, df_temp) for i in range(125)])

# x_ = [20*i for i in range(125)]

# alltime = [est_price(13314, i*20) for i in range(125)]
# # x_ = [10*i for i in range(300)]

# # y_ = [est_price(29282, i*10) for i in range(300)]

# true_sac = 10.8

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x_, y=alltime,mode='lines+markers',

#                     opacity = 0.8, name= 'alltime'))

# fig.add_trace(go.Scatter(x=x_, y=notsomeday,mode='lines+markers',

#                         opacity = 0.8, name= 'xxx'))

# fig.add_trace(go.Scatter(x=x_, y=[true_sac for i in range(125)],mode='lines+markers',

#                     opacity = 0.8, name= 'true_sac'))
df3 = df2[['Date', 'Total','Amount']].copy().groupby('Date').sum().reset_index()

df4 = df2[['Date', 'Total','Amount', 'types']].copy().groupby(['Date','types']).sum().reset_index()
color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 

            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 

            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']

fig = go.Figure()

fig.add_trace(go.Scatter(x=df3['Date'], y=df3.Amount*15,mode='lines+markers',

                    opacity = 0.8, line = dict(color = '#E61C5D'), name= '交易总量（放大15倍）'))

fig.add_trace(go.Scatter(x=df3['Date'], y=df3.Total,mode='lines+markers',

                    opacity = 0.8, line = dict(color = '#3A0088'), name= '交易总额'))

fig.add_trace(go.Scatter(x=df3['Date'], y=df3.Amount,mode='lines',

                    opacity = 0.8, line = dict(color = '#C9D6DF'), name= '交易总量'))

fig.update_layout(

    title="小圣杯日交易总额总量",

    xaxis_title="日期",

)

fig.update_layout(

    #showlegend=False,

    annotations=[

        go.layout.Annotation(x=datetime.date(2019,8,15),y=2566000,

            xref="x",yref="y",text="推荐码上线",arrowhead=6,ax=0,ay=-180),

        go.layout.Annotation(x=datetime.date(2019,9,13),y=45296600,

            xref="x",yref="y",text="中秋福利发放&印花税上线",arrowhead=6,ax=0,ay=-15),

        go.layout.Annotation(x=datetime.date(2019,9,30),y=1231780,

            xref="x",yref="y",text="英灵殿实装",arrowhead=6,ax=-5,ay=-180),

        go.layout.Annotation(x=datetime.date(2019,10,24),y=1150000,

            xref="x",yref="y",text="刮刮乐上线",arrowhead=6,ax=0,ay=-300),

        go.layout.Annotation(x=datetime.date(2019,12,1),y=39468300,

            xref="x",yref="y",text="刮刮乐升级",arrowhead=6,ax=0,ay=-30),

    ]

)



iplot(fig)
plotly.offline.plot(fig, filename='index_overall.html')
# fig = px.line(df4, x='Date', y='Total', color='types')

# fig.show()
fig = go.Figure()

count=0

color_op = [['#3FC1C9', '#364F6B', '#FC5185'],['#A8D8EA','#AA96DA','#FCBAD3', '#FFFFD2']]

trace_name = ['ICO', '普通交易','竞拍']

for i in ['ico','normal','auction']:

    fig.add_trace(go.Scatter(x=df4.loc[df4.types == i, 'Date'], y=df4.loc[df4.types == i, 'Total'],

                    mode='lines+markers',line = dict(color = color_op[0][count]),

                    name=trace_name[count]))

    count+=1

fig.update_layout(

    title="小圣杯日交易量细分",

    xaxis_title="日期",

    yaxis_title="金额",

)

fig.show()
plotly.offline.plot(fig, filename='index_detailed.html')