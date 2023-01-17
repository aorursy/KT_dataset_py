import pandas as pd

import numpy as np



dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y %H:%M')

df = pd.read_csv('../input/VNS_PU1.csv', sep=';', decimal=',', parse_dates=['date'], date_parser=dateparse, index_col=['date'])

#df.info()

df.head()
df.describe()
df.drop(['press_in2'], axis=1, inplace=True)
df = df.asfreq(pd.infer_freq(df.index))

#df = df.set_index('date').resample('D').mean().fillna(0)

df.index
df.info()
_df_tmp = df[['volume']].copy()



# helper для поиска пропущенных данных

_df_tmp['_v1'] = -_df_tmp['volume'].diff(periods=-1)

_df_tmp['_v1'].fillna(0, inplace=True)

# Для всех отрицательных суточных значений (на границе годов) и нулей (данные не снимались) исходные данные помечаем недействительными

_df_tmp['_is_daily_missing'] = _df_tmp['_v1'] <= 0

# Устранение выбросов на границе годов (для графика)

_df_tmp.loc[_df_tmp['_v1'] < 0, '_v1'] = 0

#df.loc[df['vol_tmp'].isna()] #.reset_index().groupby('vol_tmp')['date'].apply(np.array)
missing_intervals = pd.DatetimeIndex([])

missing_indices = []

for d in _df_tmp[_df_tmp['_is_daily_missing']].index:

    d_start = d - pd.offsets.Day()

    if d_start not in _df_tmp.index:

        d_start = d

    d_end = d + pd.offsets.Day()

    if d_end not in _df_tmp.index:

        d_end = d

    missing_intervals = missing_intervals.union(pd.date_range(start=d_start, end=d_end))

    missing_indices.append(_df_tmp.index.get_loc(d))



#_df_tmp.loc[missing_intervals]
%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg' 

#дефолтный размер графиков

from pylab import rcParams

rcParams['figure.figsize'] = 8, 5



fig, ax = plt.subplots()

ax.set_title('Пропуски в данных по объему')



#def highlight_datetimes(indices, ax):

ax.plot(_df_tmp['_v1'], 'g-')

for i in missing_indices:

    i2 = i + 1

    if i2 >= len(_df_tmp.index):

        i2 = i

    ax.axvspan(_df_tmp.index[i], _df_tmp.index[i2], facecolor='red', edgecolor='none', alpha=.2)

#plt.savefig('ts_anomaly.png')
_df_s = _df_tmp[['volume', '_v1', '_is_daily_missing']].copy()



# Недостоверные данные накопительного итога на день опережают недостоверные суточные

_is_cum_missing = _df_tmp['_is_daily_missing'].shift(periods=-1).ffill().to_frame('_is_cum_missing')

_v2 = _df_tmp['_v1'].shift(periods=-1).ffill().to_frame('_v2')

_df_s = pd.concat([_df_s, pd.DataFrame(_is_cum_missing), pd.DataFrame(_v2)], axis=1, join_axes=[_df_s.index])

#_df_s.iloc[missing_indices]
grp = _df_s.iloc[missing_indices].groupby('volume') # Надо добавить в группировку год



mis_df = pd.DataFrame(grp['_v2'].count().to_frame('count'))

mis_df = mis_df.rename(columns={'_v2': 'count'})

mis_df['mean'] = grp['_v2'].mean()

mis_df['sum'] = grp['_v2'].sum()



mis_df
counter_err = mis_df[(mis_df['count']>1) & (mis_df['mean']>(_df_tmp['_v1'].mean()/2))]

counter_err.index.values
_df_tmp['_vol_tmp'] = _df_tmp['volume']

# Зачистка повторяющихся значений для последующей интерполяции

for i in counter_err.index.values:

    _df_tmp.loc[_df_tmp['_vol_tmp'] == i, '_vol_tmp'] = np.NaN
_df_tmp['_vol_daily'] = -_df_tmp['_vol_tmp'].diff(periods=-1)

_df_tmp['_vol_daily'].fillna(0, inplace=True)

# Устранение выбросов на границе годов (для графика)

_df_tmp.loc[_df_tmp['_vol_daily'] < 0, '_vol_daily'] = 0

#_df_tmp.iloc[missing_indices]
df['_volume_fixed'] = _df_tmp['_vol_tmp']



df['_volume_fixed'].interpolate(inplace=True) # method='time'

df['press_in1'].interpolate(inplace=True)

df['press_out'].interpolate(inplace=True)

df['flow'].interpolate(inplace=True)

#df.loc[missing_inerval]
df['volume_d_'] = -df['_volume_fixed'].diff(periods=-1)

# Для последнего дня таблицы расчитать суточный объем невозможно - поэтому обнуляем

df['volume_d_'].fillna(0, inplace=True)

#df.drop(['cum_vol'], axis=1, inplace=True)

# Замена отрицательных значений на границе года нулями

df.loc[df['volume_d_'] < 0, 'volume_d_'] = 0



data_cols = ['volume_d_', 'flow', 'press_in1', 'press_out']
df.describe()
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected = True)
# matplotlib.style.available

# matplotlib.style.use('classic')

#a = df[data_cols].plot(subplots=True, figsize=(10, 10), grid=True, x_compat=True)



# Вариант с plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.plotly as py

from plotly import graph_objs as go

init_notebook_mode(connected = True)



def plotly_df(df, title = ''):

    data = []



    for column in df.columns:

        trace = go.Scatter(

            x = df.index,

            y = df[column],

            mode = 'lines',

            name = column

        )

        data.append(trace)



    layout = dict(title = title)

    fig = dict(data = data, layout = layout)

    iplot(fig, show_link=False)

#%run ./plotly_df.py

plotly_df(df[['volume_d_']], title = 'Суточный отпуск')

plotly_df(df[['flow', 'press_in1', 'press_out']], title = 'Давления и расход')
from pandas.plotting import scatter_matrix

a = scatter_matrix(df[data_cols], alpha=0.2, figsize=(8, 8), hist_kwds={'color':['burlywood'], 'bins':20})#, diagonal='kde')



# Variant 2

#%run ./scatter_matrix_lowess.py

#fig = scatter_matrix_lowess1(df[data_cols], alpha=0.4, figsize=(10,10), hist_kwds={'bins':20});

#fig.suptitle('Scatterplot matrix with lowess smoother', fontsize=16);
bp_color = {'boxes': 'DarkGreen', 'whiskers': 'DarkOrange', 'medians': 'DarkBlue', 'caps': 'Gray'}

df[['flow', 'press_in1', 'press_out']].plot.box(color=bp_color, vert=False, sym='r+')
df[['volume_d_']].plot.box(color=bp_color, vert=False, sym='r+')
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df[data_cols], model='additive') # model='multiplicative')



def decomposition_data(decomposition, df, field):

    data = [

        go.Scatter(

            x=df.index, y=decomposition.trend[field],

            name='Trend', mode='lines'

        ),

        go.Scatter(

            x=df.index, y=decomposition.seasonal[field],

            name='Seasonal', mode='lines'

        ),

        go.Scatter(

            x=df.index, y=decomposition.resid[field],

            name='Residual', mode='lines'

        ),

        go.Scatter(

            x=df.index, y=df[field],

            name='Observed', mode='lines'

        )

    ]

    return data



#%run ./plotly_df.py

layout = dict(title = 'Расход (м3/ч)') #yaxis = dict(zeroline = False), xaxis = dict(zeroline = False)

fig = dict(data=decomposition_data(decomposition, df, 'flow'), layout=layout)

iplot(fig) #, filename='styled-scatter')
layout = dict(title = 'Давление выход (м. вод.ст.)') 

fig = dict(data=decomposition_data(decomposition, df, 'press_out'), layout=layout)

iplot(fig)
df['month_'] = df.index.month

df['day_'] = df.index.day

df['dayofweek_'] = df.index.dayofweek

df['dayofyear_'] = df.index.dayofyear



# Праздники и переносы рабочих дней

holidays_2018 = ['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05', 

                 '2018-01-08', '2018-02-23', '2018-03-08', '2018-03-09', '2018-04-30', 

                 '2018-05-01', '2018-05-02', '2018-05-09', '2018-06-11', '2018-06-12', 

                 '2018-11-05', '2018-12-31']

moved_wd_2018 = ['2018-04-28', '2018-06-09','2018-12-29']

holidays_2019 = ['2019-01-01', '2019-01-02', '2019-01-03', '2019-01-04', '2019-01-07',

                 '2019-01-08']

df['is_workday_'] = np.logical_not(df['dayofweek_'].isin([5,6]) | df.index.isin(holidays_2018) & ~df.index.isin(moved_wd_2018)) 



#df['dayofweek_name'] = df.dayofweek.map(pd.Series('Mon Tue Wed Thu Fri Sat Sun'.split()))

#df.index.month_name()



date_cols = ['month_', 'day_', 'dayofweek_', 'dayofyear_', 'is_workday_'] #'dayofweek_name'

#date_cols = [col for col in df.columns if 'day' in col]
corr_matrix = df[data_cols + ['month_', 'dayofweek_']].corr()

sns.heatmap(corr_matrix, center=0, cmap="seismic");

corr_matrix
df['d_cat'] = pd.cut(df['is_workday_'], bins=2)

ax = sns.pairplot(df[data_cols + ['d_cat']],

                 hue="d_cat",

                 diag_kind="kde",

                 hue_order=df['d_cat'].cat.categories,

                 markers=["o", "s"],

#                 height=1.5,

                 palette='husl') #"YlGnBu"

#ax.set(xlabel='common xlabel', ylabel='common ylabel')                

plt.show()

#g.savefig("pairplot.png")
wd_vol = df[df['is_workday_']].groupby('month_')['volume_d_'].mean()

we_vol = df[~df['is_workday_']].groupby('month_')['volume_d_'].mean()



months_names = ['Янв', 'Фев', 'Мар', 'Апр', 'Май', 'Июн', 'Июл', 'Авг', 'Сен', 'Окт', 'Ноя', 'Дек']

ind = np.arange(len(months_names))

width = 0.35



fig, ax = plt.subplots()

rects1 = ax.bar(ind - width/2, wd_vol, width, color='SkyBlue', label='Будни')

rects2 = ax.bar(ind + width/2, we_vol, width, color='IndianRed', label='Выходные')

ax.set_ylabel('Отпуск, (м3/сутки)')

ax.set_title('Среднемесячный отпуск')

ax.set_xticks(ind)

ax.set_xticklabels(months_names)

ax.legend()



plt.show()
week_vol = df.groupby('dayofweek_')['volume_d_'].mean()



dow_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']

ind = np.arange(len(dow_names))

width = 0.35



fig, ax = plt.subplots()

ax.bar(ind, week_vol, align='center', color='green', ecolor='black')

ax.set_ylabel('Отпуск, (м3/сутки)')

ax.set_title('Отпуск по дням недели')

ax.set_xticks(ind)

ax.set_xticklabels(dow_names)

plt.show()
# plotly

data = []

days = df['dayofweek_'].unique()

days[::-1].sort()

for dow in days:

    data.append(go.Box(x=df[df['dayofweek_']==dow]['volume_d_'], name=dow_names[dow]))   #.astype(str)

layout = go.Layout(

    title='Суточный отпуск по дням недели',

    yaxis=dict(title='random distribution'),

    xaxis=dict(title='linspace')

)

iplot(data, show_link = False)



# sns

#sns.boxplot(y="dayofweek_", x="volume_d_", data=df, orient="h")
dow_volumes = df.pivot_table(

                        index='dayofweek_', 

                        columns='month_', 

                        values='volume_d_', 

                        aggfunc=np.mean).fillna(0).applymap(float)

#dow_volumes.info()

#sns.heatmap(dow_volumes, annot=True, fmt=".1f", linewidths=.5) # fmt=".1f", 

hm = [go.Heatmap(z=dow_volumes.values.tolist(),

                 x=months_names,

                 y=dow_names,

#                 title = 'Отпуск в зависимости от дня недели и месяца года',

                 colorscale='Viridis')] #.index) #, y=dow_volumes['dayofweek_'])

iplot(hm, filename='pandas-heatmap')
from IPython.display import HTML



HTML('''<script>

code_show=true; 

function code_toggle() {

 if (code_show){

 $('div.input').hide();

 } else {

 $('div.input').show();

 }

 code_show = !code_show

} 

$( document ).ready(code_toggle);

</script>

<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')