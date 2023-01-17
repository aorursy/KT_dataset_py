import pandas as pd
import datetime
from datetime import datetime as dt
from datetime import timedelta
import numpy as np
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
import plotly.offline as py
import plotly
dt_inicial = '2020-01-22'
dt_final = (dt.today()+timedelta(days=-2)).strftime('%Y-%m-%d')

datas = pd.DataFrame(pd.date_range(dt_inicial, dt_final).tolist(), columns =['Data'])
diretorio = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/'
datas['Arquivo'] = diretorio + pd.to_datetime(datas['Data']).dt.strftime('%m-%d-%Y').astype(str) + '.csv' 

df_cov = pd.DataFrame()

for f in datas.index:
    df = pd.read_csv(datas.Arquivo[f])
    df.columns = df.columns.str.replace("/", "_")
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("Long_", "Longitude")
    df.columns = df.columns.str.replace("Latitude", "Lat")
    df['Date'] = datas.Data[f]
    df_cov =  pd.concat([df_cov, df], ignore_index = True)

#df_cov2 = pd.concat([pd.read_csv(f) for f in datas.Arquivo], ignore_index = True)


df_cov['Province_State'] = df_cov['Province_State'].fillna('Others')
df_cov['Country_Region'] = df_cov.Country_Region.str.replace('UK', 'United Kingdom').str.replace('Mainland China', 'China')

#df_cov2['Province_State'] = df_cov2['Province_State'].fillna('Others')

df_cov['Cases'] = df_cov['Confirmed'].fillna(0) - df_cov['Deaths'].fillna(0) -  df_cov['Recovered'].fillna(0)
# Import world population data
df_pop = df_notes = pd.read_csv('https://raw.githubusercontent.com/andre-trierveiler/COV-19_Daily_Following/master/data/world_population_2018.csv', sep = ';')
df_pop['Country Name'] = df_pop['Country Name'].str.replace('UK', 'United Kingdom').str.replace('Mainland China', 'China').str.replace('United States', 'US')
def country_grid(country):
    country = 'Brazil'
    df = df_cov[df_cov['Country_Region'] == country]
    df = df.groupby(['Country_Region','Date']).sum().reset_index()
            
    df['Recover Rate'] = df['Recovered'].fillna(0)/df['Confirmed']
    df['Death Rate'] = df['Deaths'].fillna(0)/df['Confirmed']
    df['New Cases'] = df.Confirmed.rolling(2).sum()
    df['New Cases'] = 2*df.Confirmed - df['New Cases']
            
    df['New Deaths'] = df.Deaths.rolling(2).sum()
    df['New Deaths'] = 2*df.Deaths - df['New Deaths']
            
    df.set_index('Date', inplace = True)

    sns.set(style="darkgrid", palette="Set2")
    fig = plt.figure(figsize = (20,10), tight_layout =False, frameon = True)
    fig.autofmt_xdate()

    a1 = plt.subplot2grid((3,3),(0,0),rowspan = 2, sharex = None)
    a2 = plt.subplot2grid((3,3),(0,2), rowspan = 3, sharex = None)
    a3 = plt.subplot2grid((3,3),(0,1),rowspan = 2, sharex = None)
    a4 = plt.subplot2grid((3,3),(2,0),colspan = 2, sharex = None)

    def thousands(x, pos):
        'The two args are the value and tick position'
        if x >= 1000:    
          return '%1.fK' % (x*1e-3)
        else:
          return '%1.f' % (x)
    

    a2.stackplot(df.index,[df['Recovered'], df['Deaths'], df['Cases']],colors=['Mediumseagreen','tomato','dodgerblue'], labels=['Recovered','Death','Diagnosed'], alpha = 0.4)
    a2.set_xlim([datetime.date(2020, 1, 22),pd.to_datetime(dt_final, format='%Y-%m-%d')]) 
    a2.legend(loc='upper left')
    a2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    a2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    a2.yaxis.set_major_formatter(mtick.FuncFormatter(thousands))
    a2.set_title('Total Cases', loc = 'left', horizontalalignment='left')

    a1.bar(df.index,df['New Cases'], alpha=0.8, color='dodgerblue')
    a1.set_xlim([datetime.date(2020, 1, 22),pd.to_datetime(dt_final, format='%Y-%m-%d')]) 
    a1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    a1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    a1.yaxis.set_major_formatter(mtick.FuncFormatter(thousands))
    a1.set_title('Daily New Cases', loc = 'left', horizontalalignment='left')


    a3.bar(df.index,df['New Deaths'], alpha=0.8, color = 'tomato')
    #ax3.legend(loc='upper left')
    a3.set_xlim([datetime.date(2020, 1, 22),pd.to_datetime(dt_final, format='%Y-%m-%d')]) 
    a3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    a3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    a3.yaxis.set_major_formatter(mtick.FuncFormatter(thousands))
    a3.set_title('Daily New Deaths', loc = 'left', horizontalalignment='left')

    #a4.set_ylabel('Death Rate')
    a4.set_title('Death Rate', loc = 'left', horizontalalignment='left')
    a4.plot(df.index, df['Death Rate']*100, linestyle='-', color = 'tomato', alpha = 0.8)
    a4.set_xlim([datetime.date(2020, 1, 22), pd.to_datetime(dt_final, format='%Y-%m-%d')]) 
    a4.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    a4.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    a4.xaxis.set_major_locator(mdates.MonthLocator(interval=1))


    plt.subplots_adjust(left=0.05, right=0.87, bottom = 0.20, top = 0.83, wspace=0.15, hspace=0.5)
    fig.text(x=0.87, y=0.125, s='Last Updated '+df_cov.Date.max().strftime('%B, %d'), fontfamily = 'serif', horizontalalignment='right',color='#524939', fontname = 'DejaVu Sans')
    fig.text(x=0.872, y=0.1, s='by André Trierveiler ', fontfamily = 'serif', horizontalalignment='right',color='#524939', fontname = 'DejaVu Sans')
    fig.text(x=0.755, y=0.93, s='Sources: John Hopkins University', fontfamily = 'serif', horizontalalignment='left',color='#524939', fontname = 'DejaVu Sans')
    fig.patch.set_facecolor('whitesmoke')
    fig.suptitle('Novel Coronavirus in ' + country, fontsize = 24,x=0.05, y=0.93, horizontalalignment='left',color='#524939', fontname='Liberation Serif')
    #fig.title('COVID')
    #a1.set_facecolor('gainsboro')
    #plt.margins(x=1, y=1, tight=True)
    #plt.tight_layout()
    plt.show()

country_grid('Brazil')