import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import plotly_express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import folium
from folium import plugins
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode()

xls = pd.ExcelFile("/kaggle/input/civilian-complaints-against-nyc-police-officers/CCRB Data Layout Table.xlsx")
df_command = pd.read_excel(xls, 'Command Abbrevs')
df_disposition = pd.read_excel(xls, 'Dispositions')
df_fado = pd.read_excel(xls, 'FADO')
df_layout = pd.read_excel(xls, 'Layout')
df_rank = pd.read_excel(xls, 'Rank Abbrevs')

df = pd.read_csv("/kaggle/input/civilian-complaints-against-nyc-police-officers/allegations_202007271729.csv")


df
df_layout
df.info()
df_disposition
df.board_disposition.unique()
df_fado.sample(n=20)
df_command
df_rank
df_year = pd.DataFrame(df.year_received.value_counts()).reset_index()
df_year.rename(columns = {'index':'year', 'year_received':'number of civilian complaints'}, inplace=True)
df_year.sort_values(by = ['year'], inplace=True)

fig = px.scatter(df_year, x='year', y='number of civilian complaints')
fig.show()

df['day_received'] = 15
df['date_received'] = pd.to_datetime((df.year_received*10000+df.month_received*100+df.day_received).apply(str),format='%Y%m%d')
df_month = pd.DataFrame(df.date_received.value_counts()).reset_index()
df_month.rename(columns = {'index':'month', 'date_received':'number of civilian complaints'}, inplace=True)
df_month.sort_values(by = ['month'], inplace=True)

fig = px.scatter(df_month, x='month', y='number of civilian complaints')
fig.show()

df_recent = df[(df.year_received >= 2008) & (df.year_received < 2020)]
df_year = pd.DataFrame(df_recent.year_received.value_counts()).reset_index()
df_year.rename(columns = {'index':'year', 'year_received':'number of civilian complaints'}, inplace=True)
df_year.sort_values(by = ['year'], inplace=True)

fig = px.scatter(df_year, x='year', y='number of civilian complaints')
fig.show()

df_month = pd.DataFrame(df_recent.date_received.value_counts()).reset_index()
df_month.rename(columns = {'index':'month', 'date_received':'number of civilian complaints'}, inplace=True)
df_month.sort_values(by = ['month'], inplace=True)

fig = px.scatter(df_month, x='month', y='number of civilian complaints')
fig.show()
import calendar

df_month = pd.DataFrame(df_recent.month_received.value_counts()).reset_index().rename(columns={'index':'month', 'month_received':'number of civilian complaints'})
df_month.sort_values(by=['month'], inplace=True)
df_month['month'] = df_month['month'].apply(lambda x: calendar.month_abbr[x])
fig = px.bar(df_month, x = 'month', y = 'number of civilian complaints', color='number of civilian complaints', hover_name = 'month')
fig.update_layout(xaxis_title='')
fig.update_layout(coloraxis_colorbar=dict(
    title=""
))
fig.show()
def strip_specific(text):
    if '(' in text:
        text, _ = text.split('(')
        text = text.strip()
    return text

df_recent['board_disposition_general'] = df_recent['board_disposition'].apply(lambda x: strip_specific(x))

df_board_result = pd.DataFrame(df_recent.groupby(['year_received', 'board_disposition_general']).size()).reset_index()
df_board_result.rename(columns={'year_received':'year', 'board_disposition_general':'board_disposition', 0:'number of civilian complaints'}, inplace=True)

fig = px.bar(df_board_result, x='year', y='number of civilian complaints', color='board_disposition')
fig.show()

for i in range(df_board_result.shape[0]):
    year = df_board_result.loc[i, 'year']
    num = df_board_result.loc[i, 'number of civilian complaints']
    tot = df_board_result[df_board_result['year'] == year]['number of civilian complaints'].sum()
    df_board_result.loc[i, 'percentage of civilian complaints'] = num/tot

fig = px.line(df_board_result, x='year', y ='percentage of civilian complaints', color='board_disposition')

fig.add_trace(px.scatter(df_board_result, x ='year', y='percentage of civilian complaints', color='board_disposition').data[0])
fig.add_trace(px.scatter(df_board_result, x ='year', y='percentage of civilian complaints', color='board_disposition').data[1])
fig.add_trace(px.scatter(df_board_result, x ='year', y='percentage of civilian complaints', color='board_disposition').data[2])
fig['data'][0]['showlegend']=False
fig['data'][1]['showlegend']=False
fig['data'][2]['showlegend']=False
fig.layout.yaxis.tickformat = ',.0%' 
fig.show()


