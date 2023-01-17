import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

from datetime import datetime



init_notebook_mode()

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data= pd.read_csv("../input/covid19-global-forecasting-week-3/train.csv")

data
data.drop("Id", axis=1, inplace=True)

data.info()
data.describe()
duplicate_rows=data.duplicated(['Province_State','Country_Region','Date'])

data[duplicate_rows]
country_list=list(data['Country_Region'].unique())

print(country_list)

print(len(country_list))
data.loc[data['Country_Region']=='Mainland China','Country_Region']='China'
print(list(data['Date'].unique()))

print(len(list(data['Date'].unique())))
data['Date'] = pd.to_datetime(data['Date'])

data['Date_date']=data['Date'].apply(lambda x:x.date())
df_by_date=data.groupby(['Date_date']).sum().reset_index(drop=None)



df_by_date['daily_ConfirmedCases']=df_by_date.ConfirmedCases.diff()

df_by_date['daily_Fatalities']=df_by_date.Fatalities.diff()



print(df_by_date)
sns.axes_style("whitegrid")



sns.barplot(

x="Date_date",

y="ConfirmedCases", data=data.groupby(['Date_date']).sum().reset_index(drop=None)

)



plt.xticks(rotation=60)

plt.ylabel('Number of confirmed cases',fontsize=15)

plt.xlabel('Dates',fontsize=4)
data1= pd.read_excel("../input/week3/2020.xlsx")

data1
fig_oversea = px.line(data1, x='日期', y='数量',

                      line_group='输入地',

                      color='输入地',

                      color_discrete_sequence=px.colors.qualitative.D3,

                      hover_name='输入地',

)



fig_oversea.show()
fig_oversea = px.line(data1, x='日期', y='数量',

                      line_group='来源地',

                      color='来源地',

                      color_discrete_sequence=px.colors.qualitative.D3,

                      hover_name='来源地',

)



fig_oversea.show()
!pip install pyecharts
from pyecharts.charts import Sankey

from pyecharts import options as opts
nodes = []

for i in set(pd.concat([data1.来源地, data1.输入地])):

    d1 = {}

    d1['name'] = i

    nodes.append(d1)
links = []

for x, y, z in zip(data1.来源地, data1.输入地, data1.数量):

    d2 = {}

    d2['source'] = x

    d2['target'] = y

    d2['value'] = z

    links.append(d2)
pic = (

    Sankey(init_opts=opts.InitOpts(width="1600px", height="800px"))

    .add('确诊病例', 

         nodes,    

         links,   

         linestyle_opt=opts.LineStyleOpts(opacity = 0.3, curve = 0.5, color = "source"),

         label_opts=opts.LabelOpts(position="right"),         

         node_gap = 10 

    )

    .set_global_opts(title_opts=opts.TitleOpts(title = 'TOP10境外输入统计'))

)



pic.render('TOP10境外输入统计.html')