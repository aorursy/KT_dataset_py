import matplotlib.pyplot as plt
import mplleaflet

plt.figure(figsize=(6, 6))

# ANA's station
plt.plot(-44.120881, -20.197752, 'bo', ms=10)   # Melo Franco (02044008) [8,81 km (5.47 mi)]

# CEMADEN's stations
# plt.plot(-44.2, -20.143, 'go', ms=10)     # Centro (310900601A) [8,98 km (5.58 mi)]
# plt.plot(-44.216, -20.146, 'go', ms=10)   # Progresso (310900602A) [10,67 km (6.63 mi)]
plt.plot(-44.047, -20.094, 'go', ms=10)   # Casa Branca district (310900603A) [7,91 km (4.92 mi)]
# plt.plot(-44.023, -20.165, 'go', ms=10)   # Prefeito Maciel street (310900604A) [11,18 km (6.95 mi)]
plt.plot(-44.107, -20.135, 'go', ms=10)   # Córrego do Feijão (310900605A) [2,16 km (1.34 mi)]
# plt.plot(-44.2009, -20.1409, 'go', ms=10) # Rio Paraopeba (310900605H) [9,00 km (5.59 mi)]
plt.plot(-44.147, -20.156, 'go', ms=10)   # Alberto Flores (310900606A) [5,14 km (3.19 mi)]
plt.plot(-44.198, -20.142, 'go', ms=10)   # Centro (310900607A) [8,74 km (5.43 mi)]
plt.plot(-44.105, -20.196, 'go', ms=10)   # Aranha (310900608A) [8,72 km (5.42 mi)]
# plt.plot(-44.227, -20.12, 'go', ms=10)    # Inhotim (310900609A) [11,38 km (7.07 mi)]

# Dam I
plt.plot(-44.118047, -20.118579, 'rX', ms=10)
    
mplleaflet.display()
from IPython.display import display, Pretty

with open('../input/ana-melo-franco-weather-station/chuvas_C_02044008sample.csv', 'r', encoding='latin-1') as input_file:
    # ANA's Melo Franco station data (first 50 file lines)
    display(Pretty(data=input_file.read()))
# CEMADEN's stations data for January 2014 in Brumadinho (first 50 file lines)
display(Pretty(filename='../input/cemaden-weather-stations/3277_MG_2014_1sample.csv'))
import pandas as pd

# For the sake of performance, Pandas by default truncates the total rows and columns to be displayed.
# The following lines modify this setting and instruct the library to display more or less data.
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', None)

"""
Here we use _ as the name of our DataFrame being processed. Although _ is a valid Python variable name,
it is generally not good practice in software engineering to name a variable with such a generic name. However,
as we will see later, many operations require frequent repetition of the DataFrame name, which can make the lines
very long, so we’ll use this trick to keep the lines of code as short as possible until we’ve finished cleaning
and processing our data.
"""

_ = pd.read_csv('../input/ana-melo-franco-weather-station/chuvas_C_02044008.csv', decimal=',', index_col=False, sep=';', skiprows=12)
_
_['Data'] = pd.to_datetime(_['Data'], dayfirst=True)
_[['Data','Total']]
mask1 = _.groupby(['Data'])['NivelConsistencia'].transform(max) == _['NivelConsistencia']
mask2 = (_['Data'].dt.year < 2019) | ((_['Data'].dt.year == 2019) & (_['Data'].dt.month == 1))

_ = _[mask1 & mask2].sort_values('Data')
_.set_index('Data', inplace=True)
_
_['TotalStatus'].value_counts()
_[_['TotalStatus'] == 0]
# Updating 'Total' e 'TotalStatus'
# Note that if any of the days had a status other than 1, our 'TotalStatus' would also be different
_.at['2011-10-01', 'Total'] = _.loc['2011-10-01'].filter(regex=("Chuva[0-9]{2}$")).sum().round(1)
_.at['2011-10-01', 'TotalStatus'] = _.loc['2011-10-01'].filter(regex=("Chuva[0-9]{2}Status$")).max()
_.at['2011-12-01', 'Total'] = _.loc['2011-12-01'].filter(regex=("Chuva[0-9]{2}$")).sum().round(1)
_.at['2011-12-01', 'TotalStatus'] = _.loc['2011-12-01'].filter(regex=("Chuva[0-9]{2}Status$")).max()
_.at['2014-10-01', 'Total'] = _.loc['2014-10-01'].filter(regex=("Chuva[0-9]{2}$")).sum().round(1)
_.at['2014-10-01', 'TotalStatus'] = _.loc['2014-10-01'].filter(regex=("Chuva[0-9]{2}Status$")).max()

_.loc[[pd.to_datetime('2011-10-01'), pd.to_datetime('2011-12-01'), pd.to_datetime('2014-10-01')]]
df_ana = _[_['TotalStatus'] == 1]['Total'].rename('ANA')
df_ana.index.name = None
df_ana
from glob import glob

pd.set_option('display.max_rows', 70)

df1 = pd.concat([pd.read_csv(f,
                             sep=';',
                             index_col=False,
                             decimal=',',
                             usecols=['codEstacao','datahora','valorMedida']) for f in glob('../input/cemaden-weather-stations/3277_MG_*.csv')])

df1.loc[0]
df_br_date = pd.concat([pd.read_csv(f,
                                    sep=';',
                                    index_col=False,
                                    decimal=',',
                                    usecols=['codEstacao','datahora','valorMedida']) for f in glob('../input/cemaden-weather-stations-renamed/_3277_MG_*.csv')],
                       ignore_index=True)

df_dot_decimal = pd.concat([pd.read_csv(f,
                                        sep=';',
                                        index_col=False,
                                        usecols=['codEstacao','datahora','valorMedida']) for f in glob('../input/cemaden-weather-stations-renamed/dot_3277_MG_*.csv')],
                           ignore_index=True)

df_us_date = pd.concat([pd.read_csv(f,
                                    sep=';',
                                    index_col=False,
                                    decimal=',',
                                    usecols=['codEstacao','datahora','valorMedida']) for f in glob('../input/cemaden-weather-stations-renamed/3277_MG_*.csv')],
                       ignore_index=True)

df_br_date['datahora'] = pd.to_datetime(df_br_date['datahora'], dayfirst=True)
df_dot_decimal['datahora'] = pd.to_datetime(df_dot_decimal['datahora'])
df_us_date['datahora'] = pd.to_datetime(df_us_date['datahora'])

print(df_br_date.head())
print(df_dot_decimal.head())
print(df_us_date.head())
df2 = pd.concat([df_br_date, df_dot_decimal, df_us_date], ignore_index=True)
df2.shape
df3 = df2[df2['codEstacao'] == '310900605A'].copy()
df3.shape
def hours_in_month(month, year):
    if month in (1,3,5,7,8,10,12):
        return 744
    elif month in (4,6,9,11):
        return 720
    elif year == 2016:
        return 696
    else:
        return 672
_ = df3[['valorMedida']].groupby(df3['datahora'].dt.floor('H')).sum().groupby(pd.Grouper(freq='MS')).count()
_['%'] = _.index.map(lambda dt: (_.loc[dt]['valorMedida']/hours_in_month(dt.month, dt.year)*100).round(2))
_ = _.rename(columns={'valorMedida': 'Hours with measurements'})
_
df4 = df2[df2['codEstacao'].isin(['310900603A', '310900605A', '310900606A', '310900607A', '310900608A'])].copy()
df4['datahora'] = df4['datahora'].dt.floor('H')

df_corrego_feijao = df4[df4['codEstacao'] == '310900605A'][['datahora','valorMedida']]
df_alberto_flores = df4[df4['codEstacao'] == '310900606A'][['datahora','valorMedida']]
df_casa_branca = df4[df4['codEstacao'] == '310900603A'][['datahora','valorMedida']]
df_aranha = df4[df4['codEstacao'] == '310900608A'][['datahora','valorMedida']]
df_centro = df4[df4['codEstacao'] == '310900607A'][['datahora','valorMedida']]

df_corrego_feijao = df_corrego_feijao.groupby('datahora').sum()
df_alberto_flores = df_alberto_flores.groupby('datahora').sum()
df_casa_branca = df_casa_branca.groupby('datahora').sum()
df_aranha = df_aranha.groupby('datahora').sum()
df_centro = df_centro.groupby('datahora').sum()

df_cemaden = (df_corrego_feijao.combine_first(df_alberto_flores)
                               .combine_first(df_casa_branca)
                               .combine_first(df_aranha)
                               .combine_first(df_centro))
_ = pd.DataFrame(df_cemaden.groupby(pd.Grouper(freq='MS')).count()).rename(columns={'valorMedida': 'Hours with measurements'})
_['%'] = _.index.map(lambda dt: (_.loc[dt]['Hours with measurements']/hours_in_month(dt.month, dt.year)*100).round(2))
_
df_cemaden = df_cemaden.groupby(pd.Grouper(freq='MS')).sum().round(1)['valorMedida'].rename('CEMADEN')
df_cemaden.index.name = None
df_cemaden
import calendar
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import numpy as np
def precipitation_fourth_quarter(ana, cemaden):
    _ = ana[:-1].groupby(pd.Grouper(freq='QS')).sum()
    __ = cemaden[:-1].groupby(pd.Grouper(freq='QS')).sum()
    
    _ = _.groupby(_.index.month)
    __ = __.groupby(__.index.month)

    maxs_ana = _.max()
    maxs_cemaden = __.max()
    maxs_ana = maxs_ana.loc[10].sum()
    maxs_cemaden = maxs_cemaden.loc[10].sum()
    
    means_ana = _.mean()
    means_cemaden = __.mean()
    means_ana = means_ana.loc[10].sum()
    means_cemaden = means_cemaden.loc[10].sum()

    g_labels = ['ANA (1941 to 2018)', 'CEMADEN (2014 to 2018)']
    g_maxs = [maxs_ana, maxs_cemaden]
    g_means = [means_ana, means_cemaden]
    g_72p = [means_ana*1.72, means_cemaden*1.72]
    g_2018 = [ana.iloc[-4:-1].sum(), cemaden.iloc[-4:-1].sum()]
    
    x = np.arange(2)
    width = 0.20

    fig, ax = plt.subplots()
    fig.set_size_inches(6.4, 4.8)
    
    rects1 = ax.bar(x - width, g_maxs, width, label='All-time high', color='tab:blue')
    rects2 = ax.bar(x, g_means, width, label='All-time average', color='tab:cyan')
    rects3 = ax.bar(x + width, g_2018, width, label='2018 Q4', color='tab:orange')
    
    for i, rects in enumerate([rects1, rects2, rects3]):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., height + 15,
                    '%d mm' % int(height), ha='center', va='bottom')

    ax.set_title('Accumulated precipitation for the 4th quarter in Brumadinho', pad=17)
    ax.set_xticks(x)
    ax.set_xticklabels(g_labels)
    ax.set_yticklabels([])
    plt.tick_params(axis='both', length=0)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=3)

    fig.tight_layout()
    
    plt.ylim(0, 1300)
    
    for s in ax.spines:
        ax.spines[s].set_visible(False)

    plt.show()
def precipitation_by_year(df):
    plt.figure(figsize=(6.4, 4.8))

    df_by_year = df.groupby(pd.Grouper(freq='Y')).sum()[1:-1]
    
    precip_plot = df_by_year.plot()
    blue_line = mlines.Line2D([], [], color='blue')
    
    means_plot, = plt.plot([-29, 48], [df_by_year.mean(), df_by_year.mean()], color='tab:orange')

    plt.tick_params(axis='x', length=0)

    plt.title('Annual precipitation in Brumadinho [{}]'.format(df.name), pad=17)

    plt.ylim(0, 2000)
    
    ax = plt.gca()
    span_area = ax.axvspan(46, 48, alpha=0.3, color='red')
    
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d mm'))
    
    plt.legend(handles=[blue_line, means_plot, span_area],
           labels=['Annual Precipitation', 'Average', '2016 to 2018 span'],
           loc='upper center',
           fontsize='small',
           ncol=3,
           frameon=False,
           bbox_to_anchor=(0.5, -0.07))
    
    for s in ['bottom','right','top']:
        ax.spines[s].set_visible(False)

    plt.show()
def precipitation_profile(df):
    plt.figure(figsize=(6.4, 4.8))

    _ = df.iloc[:-7].groupby(df.iloc[:-7].index.month)

    maxs = _.max().rename(index=lambda x: calendar.month_abbr[x])
    means = _.mean().rename(index=lambda x: calendar.month_abbr[x])
    std_errors = (_.std()/np.sqrt(_.count())).rename(index=lambda x: calendar.month_abbr[x])
    
    shift_mask = calendar.month_abbr[7:14] + calendar.month_abbr[1:7]
    maxs = maxs.loc[shift_mask]
    means = means.loc[shift_mask]
    std_errors = std_errors.loc[shift_mask]

    x_axis = np.arange(12)
    means_plot = plt.bar(x_axis, means, yerr=std_errors*2, ecolor='tab:gray', color='tab:cyan')
    maxs_plot, = plt.plot(x_axis, maxs, '.:', color='tab:blue')
    measured_plot, = plt.plot(np.arange(7), df.iloc[-7:], 'o-',  color='tab:orange')
    
    plt.xticks(x_axis, shift_mask)
    plt.tick_params(axis='x', length=0)

    plt.title('Brumadinho\'s precipitation profile [{}]'.format(df.name), pad=17)

    plt.ylim(0, 700)
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d mm'))

    plt.legend(handles=[maxs_plot, means_plot, measured_plot],
               labels=['All-time highs', 'All-time averages', 'Jul/18 to Jan/19'],
               loc='upper center',
               fontsize='small',
               ncol=3,
               frameon=False,
               bbox_to_anchor=(0.5, -0.07))
    
    for s in ['bottom','right','top']:
        ax.spines[s].set_visible(False)

    plt.show()
precipitation_fourth_quarter(df_ana, df_cemaden)
precipitation_by_year(df_ana)
precipitation_profile(df_ana)
precipitation_profile(df_cemaden)
