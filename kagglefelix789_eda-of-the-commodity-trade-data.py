# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Import the libraries
from matplotlib import pyplot as plt
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Load the dataset
df = pd.read_csv('../input/commodity_trade_statistics_data.csv',)
# make a copy of the original data
df_clean = df.copy()
df.sample()
# checking missing data in stack data 
total = df_clean.isnull().sum().sort_values(ascending = False)
percent = (df_clean.isnull().sum()/df_clean.isnull().count()*100).sort_values(ascending = False)
missing_df_clean = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_df_clean
top15_trade_count=list(df_clean.country_or_area.value_counts().head(15).index)
df_top15_trade_count = df_clean[df_clean['country_or_area'].isin(top15_trade_count)]
# This is the trade amount of each country
fig,ax = plt.subplots(figsize=(10,8))
sns.countplot(y='country_or_area',
           data=df_top15_trade_count[df_top15_trade_count['flow']=='Import'],
             color='#FFDEAD',
             order = df_clean.country_or_area.value_counts().head(15).index,
             label="Import",)
sns.countplot(y='country_or_area',
           data=df_top15_trade_count[df_top15_trade_count['flow']=='Export'],
             color='#CD853F',
             order = df_clean.country_or_area.value_counts().head(15).index,
             label="Export")
sns.countplot(y='country_or_area',
           data=df_top15_trade_count[df_top15_trade_count['flow']=='Re-Import'],
             color='#FF6347',
             order = df_clean.country_or_area.value_counts().head(15).index,
             label="Re-Import")
sns.countplot(y='country_or_area',
           data=df_top15_trade_count[df_top15_trade_count['flow']=='Re-Export'],
             color='#FF8C00',
             order = df_clean.country_or_area.value_counts().head(15).index,
             label="Re-Export")
ax.legend(loc="lower right", frameon=True)
ax.set(title ="Top 10 Trade Most Countries",
    ylabel="",
       xlabel="Countries trade amount")
ax.grid(which='minor',alpha=0.5)
plt.show()
df_countries_trade = df_clean.groupby(['country_or_area','flow'],as_index=False)['trade_usd'].agg('sum')
top10_countries = df_clean.groupby(['country_or_area'],as_index=False)['trade_usd'].agg('sum')
top10_countries = top10_countries.sort_values(['trade_usd'],ascending=False)[10:0:-1]
df_countries_trade = df_countries_trade[df_countries_trade['country_or_area'].isin(top10_countries['country_or_area'])]
df_countries_trade=df_countries_trade.pivot('country_or_area','flow','trade_usd')
df_countries_trade=df_countries_trade.reindex(top10_countries['country_or_area'])
# Draw the stacked bar plot of the most 10 traded countries

traces=[]
colors=['rgba(50, 171, 96, 1.0)']
for flow in list(df_countries_trade):
    traces.append(go.Bar(
        y=df_countries_trade.index,
        x=df_countries_trade[flow],
        name=flow,
        orientation='h',
        opacity=0.8,
    ))

data = traces
layout = go.Layout(
    title='Top 10 Trade Volumn Countries',
    margin=go.Margin(
        l=100,
        r=50,
        b=100,
        t=100,
        pad=1
        ),
    barmode='stack',
    
)

fig = go.Figure(data=data, layout=layout)
iplot(fig,filename='stack_barplot')
# Group the df by country and year
df_trade = df_clean.groupby(['country_or_area','year'],as_index=False)['trade_usd'].agg('sum')
country_trade_list = df_clean.groupby(['country_or_area'],as_index=False)['trade_usd'].agg('sum')
country_trade_list = country_trade_list.sort_values(['trade_usd'],ascending=False).head(11)
# Draw the line of serveral countries
# Create a trace, which is a Data object

traces=[]
for i in country_trade_list['country_or_area']:
    traces.append(go.Scatter(
        x=df_trade[df_trade['country_or_area'] == i].year,
        y=df_trade[df_trade['country_or_area'] == i].trade_usd,
        mode='lines+markers',
        name=i,
        line=dict(
            shape='spline'
        )
    )
    )
    
# Create a Layout ohject
layout = go.Layout(
    title='Trade Volume of Top 10 Countries(area)',
    xaxis=dict(
        title="Year",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        #tickangle=270,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(120, 120, 120)',
        ),
    ),
    yaxis=dict(
        
        title="Trade Volume",
        showgrid=True,
        zeroline=False,
        showline=False,
        showticklabels=True,
       
        
    ),
    #autosize=True,
    
    showlegend=True,
)


fig = go.Figure(data=traces, layout=layout)

py.offline.iplot(fig, filename='basic-line')
quantity_name = df_clean['quantity_name'].unique()
category_name =[]
category_number = []
for i in df_clean['quantity_name'].unique():
    category_name.append(i)
    category_number.append(len(df_clean[df_clean['quantity_name']== i]['category'].unique()))
# Create a Bar plot of category numbers belongs to different quantity name


Data = [go.Bar(
            x=category_name,
            y=category_number,
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6
        )]


Layout = go.Layout(
    title='Category Numbers of Each Quantity Name'
    )

fig = go.Figure(data=Data, layout=Layout)
py.offline.iplot(fig, filename='category numbers')

df_category = df_clean.groupby(['category'],as_index=False)['trade_usd'].agg('sum').sort_values(['trade_usd'],ascending=False)
df_category = df_category[10:0:-1]
df_category['category'] = df_category['category'].str.replace('_',' ').str.extract(r'\d{1,2}\s?(.*)',expand=True)
# This is the trade volume of each category
Data = [go.Bar(
            x=df_category['trade_usd'],
            y=df_category['category'].str.capitalize(),
            marker=dict(
                color='rgb(158,202,225)',
                line=dict(
                    color='rgb(8,48,107)',
                    width=1.5),
            ),
            opacity=0.6,
            orientation='h',
        )]


Layout = go.Layout(
    title='Trade Amount of Top 10 Category',
    margin=go.Margin(
        l=300,
        r=50,
        b=100,
        t=100,
        pad=1
        ),
    xaxis=dict(
        title='Trade Amount',
    )
    
    )

fig = go.Figure(data=Data, layout=Layout)
py.offline.iplot(fig)

df_energy = df_clean[df_clean['category'] == '27_mineral_fuels_oils_distillation_products_etc']
df_energy_pie = df_energy['commodity'].value_counts()[:15]
fig = {
    'data': [
        {
            
            'labels': [x.split(',')[0] for x in df_energy_pie.index],
            'values': df_energy_pie.values,
            'type': 'pie',
            'name': 'Fuel',
            'domain': {'x': [0, .8],
                       'y': [0, 1]},
            'hoverinfo':'label+percent',
            'textinfo':'percent',
            'textfont':dict(size=15),
        },
        
       
    ],
    'layout': {'title': 'Composition of Vary Commodity Belongs to Mineral Fuels',
               'showlegend': True,
              }
}

py.offline.iplot(fig,)
def check_outlier(arr):
    return arr.quantile(0.99)
df_petroleum = df_clean[(df_clean['commodity'].str.contains('petroleum',case=False)) & (df_clean['flow']=='Export')][['country_or_area','year','quantity']]
petro_groupby = df_petroleum.groupby(['country_or_area','year'])
petroleum_merge = pd.merge(df_petroleum, petro_groupby.transform(check_outlier), left_index=True,right_index = True,suffixes=['_value','_quantile'])
df_petroleum = petroleum_merge[petroleum_merge['quantity_value'] < petroleum_merge['quantity_quantile']]
df_petroleum = df_petroleum.drop(labels='quantity_quantile',axis=1).rename(columns={"quantity_value":"quantity"})
df_petroleum = df_petroleum.groupby(['country_or_area','year'],as_index=False)['quantity'].agg('sum').fillna(0)
df_petroleum_sum = df_petroleum.groupby(['country_or_area'],as_index=False)['quantity'].agg('sum')
df_petroleum_sum = df_petroleum_sum.sort_values(['quantity'],ascending=False)
df_petroleum_sum = df_petroleum_sum[1:11]
# Draw the line of serveral countries
# Create a trace, which is a Data object

traces=[]
for i in df_petroleum_sum['country_or_area']:
    traces.append(go.Scatter(
        x=df_petroleum[df_petroleum['country_or_area'] == i].year,
        y=df_petroleum[df_petroleum['country_or_area'] == i].quantity,
        mode='lines+markers',
        name=i,
    )
    )
    
# Create a Layout ohject
layout = go.Layout(
    title= 'Petroleum Export of Top 10 Countries(area)',
    xaxis=dict(
        title="Year",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        #tickangle=270,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(120, 120, 120)',
        ),
    ),
    yaxis=dict(
        
        title="Export Petroleum Quantity",
        showgrid=True,
        zeroline=False,
        showline=False,
        showticklabels=True,
       
        
    ),
    #autosize=True,
    
    showlegend=True,
)


fig = go.Figure(data=traces, layout=layout)

py.offline.iplot(fig,)
df_petroleum = df_clean[(df_clean['commodity'].str.contains('petroleum',case=False)) & (df_clean['flow']=='Import')][['country_or_area','year','quantity']]
petro_groupby = df_petroleum.groupby(['country_or_area','year'])
petroleum_merge = pd.merge(df_petroleum, petro_groupby.transform(check_outlier), left_index=True,right_index = True,suffixes=['_value','_quantile'])
df_petroleum = petroleum_merge[petroleum_merge['quantity_value'] < petroleum_merge['quantity_quantile']]

df_petroleum = df_petroleum.groupby(['country_or_area','year'],as_index=False)['quantity_value'].agg('sum').fillna(0)
df_petroleum_sum = df_petroleum.groupby(['country_or_area'],as_index=False)['quantity_value'].agg('sum')
df_petroleum_sum = df_petroleum_sum.sort_values(['quantity_value'],ascending=False)
df_petroleum_sum = df_petroleum_sum[1:11]
# Draw the line of serveral countries
# Create a trace, which is a Data object

traces=[]
for i in df_petroleum_sum['country_or_area']:
    traces.append(go.Scatter(
        x=df_petroleum[df_petroleum['country_or_area'] == i].year,
        y=df_petroleum[df_petroleum['country_or_area'] == i].quantity_value,
        mode='lines+markers',
        name=i,
    )
    )
    
# Create a Layout ohject
layout = go.Layout(
    title= 'Petroleum Import of Top 10 Countries(area)',
    xaxis=dict(
        title="Year",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        #tickangle=270,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(120, 120, 120)',
        ),
    ),
    yaxis=dict(
        
        title="Export Petroleum Quantity",
        showgrid=True,
        zeroline=False,
        showline=False,
        showticklabels=True,
       
        
    ),
    #autosize=True,
    
    showlegend=True,
)


fig = go.Figure(data=traces, layout=layout)

py.offline.iplot(fig,)
df_electric = df_clean[(df_clean['commodity']=='Electrical energy') & (df_clean['flow']=='Import')]
df_electric = df_electric.groupby(['country_or_area','year'],as_index=False)['quantity'].agg('sum')
df_electric_sum = df_electric.groupby(['country_or_area'],as_index=False)['quantity'].agg('sum')
df_electric_sum = df_electric_sum.sort_values(['quantity'])[-10:]
# Draw the line of serveral countries
# Create a trace, which is a Data object

traces=[]
for i in df_electric_sum['country_or_area']:
    traces.append(go.Scatter(
        x=df_electric[df_electric['country_or_area'] == i].year,
        y=df_electric[df_electric['country_or_area'] == i].quantity,
        mode='lines+markers',
        name=i,
        
    )
    )
    
# Create a Layout ohject
layout = go.Layout(
    title= 'Import Electric of Top 10 Countries(area)',
    xaxis=dict(
        title="Year",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        #tickangle=270,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(120, 120, 120)',
        ),
    ),
    yaxis=dict(
        
        title="Import Electric Quantity",
        showgrid=True,
        zeroline=False,
        showline=False,
        showticklabels=True,
       
        
    ),
    #autosize=True,
    
    showlegend=True,
)


fig = go.Figure(data=traces, layout=layout)

py.offline.iplot(fig, filename='Import Electric')
# Extract the China trade
df_clean_China = df_clean[df_clean['country_or_area'] == 'China']
df_clean_China =df_clean_China.groupby(['year','flow'],as_index=False)[['trade_usd']].sum()
# Create a flow category list
flow_cate=[]
for i in df_clean_China.flow.unique():
    flow_cate.append(i)
# Create a trace
traces=[]
for i in flow_cate:
    traces.append(go.Scatter(
        x=df_clean_China[df_clean_China['flow'] == i].year,
        y=df_clean_China[df_clean_China['flow'] == i].trade_usd,
        mode='lines+markers',
        name=i
    )
    )

layout = go.Layout(
    title='Trade Volume of China',
    xaxis=dict(
        title="Year",
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        autotick=False,
        ticks='outside',
        tickcolor='rgb(204, 204, 204)',
        tickwidth=2,
        ticklen=5,
        #tickangle=270,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(120, 120, 120)',
        ),
    ),
    yaxis=dict(
        title="Trade Volume",
        showgrid=True,
        zeroline=False,
        showline=False,
        showticklabels=True,
        
    ),
    autosize=True,
    
    showlegend=True,
)


fig = go.Figure(data=traces, layout=layout)

py.offline.iplot(fig, filename='basic-line')
df_china = df_clean[(df_clean['country_or_area'] == "China") & (df_clean['category'] != 'all_commodities') ][['year','commodity','flow','trade_usd','category']]
china_group = df_china.groupby(['year','category'],as_index=False)
def top(df, n=1, column='trade_usd'):
    return df.sort_values(column)[-n:]
df_commodity = df_china.groupby(['year','flow']).apply(top)
pie_list = df_commodity.commodity.str.split(',',expand=True)[0].value_counts()
# Create a Pie chart for the most common commodity traded
labels = pie_list.index
values = pie_list.values

trace = go.Pie(labels=labels, values=values,
               hoverinfo='label+percent', 
               #textinfo='label+percent',
               textfont=dict(size=15),
               marker=dict(colors=colors,
                           line=dict(color='#000000', width=1)))

py.offline.iplot([trace], filename='styled_pie_chart')