# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

import plotly.graph_objects as go

from datetime import datetime

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
## Read and clean data

df_province = pd.read_csv('../input/covid19-in-italy/covid19_italy_province.csv')

df_province = df_province[df_province['ProvinceName'] != 'In fase di definizione/aggiornamento']

df_province['Date'] = df_province['Date'].apply(pd.Timestamp)

#pd.set_option('display.max_rows', df_province.shape[0]+1)

#df_province.head()

## Sort values by region and province, and add 'NewCases' column

#df_province.loc[df_province['RegionName'] == 'Lombardia'].sort_values(['ProvinceName', 'Date']).groupby('ProvinceName')['TotalPositiveCases'].diff().reset_index()

df_province = df_province.sort_values(['RegionName', 'ProvinceName', 'Date'])

df_province['NewPositiveCases'] = df_province.sort_values(['ProvinceName', 'Date']).groupby('ProvinceName')['TotalPositiveCases'].diff()



early_lockdown = ['Alessandria', 'Asti', 'Bergamo', 'Brescia', 'Como', 'Cremona', 'Lecco', 'Mantova', 'Milano', 'Modena', 'Monza e della Brianza', 'Novara', 'Padova', 'Parma', 'Pavia', 'Pesaro e Urbino', 'Piacenza', "Reggio nell'Emilia", 'Rimini', 'Sondrio', 'Treviso', 'Varese', 'Venezia', 'Verbano-Cusio-Ossola', 'Vercelli']

start_date = pd.Timestamp("2020-02-24 18:00:00")

df_early = df_province.loc[(df_province['ProvinceName'].isin(early_lockdown)) & (df_province['Date'] >= start_date)].groupby('Date').sum().reset_index()

df_late = df_province.loc[(~df_province['ProvinceName'].isin(early_lockdown)) & (df_province['Date'] >= start_date)].groupby('Date').sum().reset_index()

df_sum = df_province.loc[df_province['Date'] >= start_date].groupby('Date').sum().reset_index()



df_early['NewCasesRate'] = df_early['NewPositiveCases'].div((df_early['NewPositiveCases'].shift(1) + df_early['NewPositiveCases'].shift(2) + df_early['NewPositiveCases'].shift(3) + df_early['NewPositiveCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

df_late['NewCasesRate'] = df_late['NewPositiveCases'].div((df_late['NewPositiveCases'].shift(1) + df_late['NewPositiveCases'].shift(2) + df_late['NewPositiveCases'].shift(3) + df_late['NewPositiveCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)

df_sum['NewCasesRate'] = df_sum['NewPositiveCases'].div((df_sum['NewPositiveCases'].shift(1) + df_sum['NewPositiveCases'].shift(2) + df_sum['NewPositiveCases'].shift(3) + df_sum['NewPositiveCases'].shift(4)) / 4).replace([np.inf, -np.inf], np.nan)



df_early['TotalCasesRate'] = df_early['TotalPositiveCases'].div(df_early['TotalPositiveCases'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_late['TotalCasesRate'] = df_late['TotalPositiveCases'].div(df_late['TotalPositiveCases'].shift(1)).replace([np.inf, -np.inf], np.nan)

df_sum['TotalCasesRate'] = df_sum['TotalPositiveCases'].div(df_sum['TotalPositiveCases'].shift(1)).replace([np.inf, -np.inf], np.nan)



df_early['Isolation'] = '8/3 lockdown'

df_late['Isolation'] = '10/3 lockdown'

df_sum['Isolation'] = 'All Italy'



df_all = pd.concat([df_sum, df_early, df_late])

df_all.columns
## Print all provinces sorted by region

# for region in df_province['RegionName'].unique():

#     print(region)

#     for province in df_province.loc[df_province['RegionName'] == region]['ProvinceName'].unique():

#         print('   ', province)

#df_province['ProvinceName'].sort_values().unique()
# fig = plt.Figure(figsize=(50,20))

# plt.plot(df_north['Date'][2:], df_north['TotalCasesRate'][2:])

# plt.plot(df_south['Date'][2:], df_south['TotalCasesRate'][2:])

# # naming the x axis 

# plt.xlabel('') 

# # naming the y axis 

# plt.ylabel('y - axis') 

# # giving a title to my graph 

# plt.title('Two lines on same graph!') 

  

# # plt.gca().set_xlim(left=0, right=25)

# # plt.gca().set_ylim(bottom=0, top=6000)

# plt.gcf().autofmt_xdate()

# plt.legend(['Lockdown at 8/3', 'Lockdown at 10/3'])

# plt.grid(True)

# plt.title("Total cases increase rate")

# plt.ylabel("Ratio")

# plt.xlabel("Date")

# #plt.yscale(value='log')

# plt.show()
fig = px.line(df_all, x="Date", y="TotalCasesRate", color='Isolation', labels={'x': 'Date', 'y':'Ratio'}, title='Italy: The rate of *total* confirmed cases in the regions with different isolation measurements')

fig.show()

fig = px.line(df_all, x="Date", y="NewCasesRate", color='Isolation', labels={'x': 'Date', 'y':'Ratio'}, title='Italy: The rate of *new* confirmed cases in the regions with different isolation measurements')

fig.show()
y = df_sum['TotalPositiveCases'][1:]

x = np.arange(len(y))

coeff1 = np.polyfit(x[:14], np.log(y[:14]), 1)

coeff2 = np.polyfit(x[10:], np.log(y[10:]), 1)

z1 = np.poly1d(coeff1)

z2 = np.poly1d(coeff2)



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Total positive cases'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z1(x)), name='Prediction1'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z2(x)), name='Prediction2'))

# fig.add_trace(go.Scatter(x=[14], y=[y[14]], marker=dict(size=20), name='time of lockdown'))

fig.update_layout(

    title="ITALY: Total confirmed cases vs. days",

    xaxis_title="Days since 25/2/2020",

    yaxis_title="y Axis Title",

)

fig.show()
y = df_early['TotalPositiveCases'][1:]

x = np.arange(len(y))

coeff = np.polyfit(x, np.log(y), 1)

z = np.poly1d(coeff)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Total positive cases in early lockdown provinces'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z(x)), name='Prediction'))

fig.add_trace(go.Scatter(x=[12], y=[y[12]], marker=dict(size=20), name='time of lockdown'))

fig.update_layout(

    title="ITALY: Total confirmed cases vs. days",

    xaxis_title="Days since 25/2/2020",

    yaxis_title="y Axis Title",

)

fig.show()
y_late = df_late['TotalPositiveCases'][1:]

x = np.arange(len(y_late))

coeff1_late = np.polyfit(x[:20], np.log(y_late[:20]), 1)

z1_late = np.poly1d(coeff1_late)

coeff2_late = np.polyfit(x[15:], np.log(y_late[15:]), 1)

z2_late = np.poly1d(coeff1_late)



y_ear = df_early['TotalPositiveCases'][1:]

coeff1_ear = np.polyfit(x[:20], np.log(y_ear[:20]), 1)

z1_ear = np.poly1d(coeff1_ear)

coeff2_ear = np.polyfit(x[15:], np.log(y_ear[15:]), 1)

z2_ear = np.poly1d(coeff2_ear)



fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y_late, mode='markers', name='Total positive cases in late lockdown provinces'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z1_late(x)), name='Prediction1 - late lockdown'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z2_late(x)), name='Prediction2 - late lockdown'))

fig.add_trace(go.Scatter(x=x, y=y_ear, mode='markers', name='Total positive cases in early lockdown provinces'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z1_ear(x)), name='Prediction1 - early lockdown'))

fig.add_trace(go.Scatter(x=x, y=np.exp(z2_ear(x)), name='Prediction2 - early lockdown'))

# fig.add_trace(go.Scatter(x=[14], y=[y[14]], marker=dict(size=20), name='time of lockdown'))

fig.update_layout(

    title="ITALY: Total confirmed cases vs. days",

    xaxis_title="Days since 25/2/2020",

    yaxis_title="Total confirmed cases",

)

fig.show()
## Plot NewCases graph for every region

# for region in df_province['RegionName'].unique():

#     fig = plt.figure(figsize=(25,10))

#     provinces_in_region = df_province.loc[df_province['RegionName'] == region]['ProvinceName'].unique()

#     for province in provinces_in_region:

#         plt.plot(df_province.loc[df_province['ProvinceName'] == province]['Date'], df_province.loc[df_province['ProvinceName'] == province]['NewCases'], label=province)

#     ylim = df_province.loc[df_province['ProvinceName'].isin(provinces_in_region)]['NewCases'].max()

#     plt.plot(['2020-03-08 18:00:00', '2020-03-08 18:00:00'], [0, ylim], 'r--', label='date of lock-down')

#     plt.gcf().autofmt_xdate()

#     #plt.gca().set_ylim(bottom=10)

#     plt.legend()

#     plt.grid(True)

#     plt.xlabel("Date")

#     plt.title("New cases per day in %s region"%region)

#     #plt.yscale(value='log')

#     plt.show()
# fig = go.Figure()

# fig.add_trace(go.Scatter(x=df_early['Date'], y=df_early['NewCasesRate'], mode='lines', name='8/3 lockdown provinces', line=dict(color='rgb(67,67,67)', width=2)))

# fig.add_trace(go.Scatter(x=df_early['Date'], y=df_early['NewCasesRate'], mode='lines', name='8/3 lockdown provinces', line=dict(color='rgb(67,67,67)', width=2)))

# fig.add_trace(go.Scatter(x=df_late['Date'], y=df_late['NewCasesRate'], mode='lines', name='10/3 lockdown provinces', line=dict(color='rgb(115,115,115)', width=2)))

# fig.add_trace(go.Scatter(x=df_late['Date'], y=df_late['NewCasesRate'], mode='lines', name='10/3 lockdown provinces', line=dict(color='rgb(115,115,115)', width=2)))

# fig.add_trace(go.Scatter(x=[df_early['Date'].iloc[4], df_early['Date'].iloc[-1]], y=[df_early['NewCasesRate'].iloc[4], df_early['NewCasesRate'].iloc[-1]], mode='markers', marker=dict(color='rgb(67,67,67)', size=8)))

# fig.update_layout(xaxis=dict(showline=True, showgrid=True, showticklabels=True, linecolor='rgb(204,204,204)', linewidth=2, ticks='outside', tickfont=dict(family='Arial', size=12, color='rgb(82,82,82)')), yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False), autosize=False, margin=dict(autoexpand=True, l=100, r=20, t=110), showlegend=False, plot_bgcolor='white')