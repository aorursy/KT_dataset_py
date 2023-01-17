# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/forest-fires-in-brazil/amazon.csv', encoding = "ISO-8859-1")
df.head()
list_ = range(1, 13)
m = dict(zip(df['month'].unique(), list_))
df['month'] = df['month'].map(m)
def quarter(month):
    if month == 1 or month == 2 or month == 3:
        return '1_Quarter'
    elif month == 4 or month == 5 or month == 6:
        return '2_Quarter'
    elif month == 7 or month == 8 or month == 9:
        return '3_Quarter'
    elif month == 10 or month == 11 or month == 12:
        return '4_Quarter'
    
    
df['quarter'] = df['month'].apply(lambda x : quarter(x))
aux = df.groupby(['year', 'quarter'])['number'].agg('sum')
indexs = []
for year,quarter in list(aux.index):
    indexs.append(str(year)+'_'+str(quarter))
import plotly.express as px
import plotly.graph_objects as go
fig = px.line(x=indexs, y=list(aux[:]), title='Observed')
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

adf_test(list(aux[:]))
import statsmodels.api as sm
import plotly.graph_objects as go

res = sm.tsa.seasonal_decompose(list(aux[:]), period=3, model="additive")
fig = go.Figure()

fig.add_trace(go.Scatter(x=indexs, y=res.observed, mode='lines',name='Obeserved'))
fig.add_trace(go.Scatter(x=indexs, y=res.trend, mode='lines',name='Trend'))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
fig = px.line(x=indexs, y=res.resid, title='Residual component')
fig.show()
def get_correlation_between_states(df):
    dict_ = {}
    for state in df['state'].unique():
        select = df[df['state']==state]
        aux2 = select.groupby(['year', 'quarter'])['number'].agg('sum')
        aux = df.groupby(['year', 'quarter'])['number'].agg('sum')
        #indexs = []
        #for year,quarter in list(aux.index):
        #    indexs.append(str(year)+'_'+str(quarter))
        dict_[state] =  np.corrcoef(aux2[:], aux[:])[0][1]
    return dict_

dict_ =  get_correlation_between_states(df)
import plotly.express as px

sorted_dict = sorted(dict_.items(), key=lambda kv: kv[1])
sorted_dict = dict(sorted_dict)
fig = px.bar(x=list(sorted_dict.keys()), y=list(sorted_dict.values()))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
def get_state(df, state):
    aux = df[df['state']==state]
    aux2 = aux.groupby(['year', 'quarter'])['number'].agg('sum')
    indexs = []
    for year, quarter in list(aux2.index):
        indexs.append(str(year)+'_'+str(quarter))
    return indexs, list(aux2[:])
import plotly.graph_objects as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(indexs), y=list(aux[:]), mode='lines',name='Observed'))
x, y = get_state(df, 'Sergipe')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Sergipe'))
x, y = get_state(df, 'Roraima')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Roraima'))
x, y = get_state(df, 'Maranhao')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Maranhao'))
x, y = get_state(df, 'Rio')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Rio'))
x, y = get_state(df, 'Acre')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Acre'))
x, y = get_state(df, 'Paraiba')
fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Paraiba'))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff

def get_correlation_between_states_detrend(df):
    dict_ = {}
    for state in df['state'].unique():
        select = df[df['state']==state]
        aux2 = select.groupby(['year'])['number'].agg('sum')
        aux = df.groupby(['year'])['number'].agg('sum')
        detrend2 = difference(list(aux2[:]))
        detrend = difference(list(aux[:]))
        dict_[state] =  np.corrcoef(detrend2, detrend)[0][1]
    return dict_
dict_ = get_correlation_between_states_detrend(df)
sorted_dict = sorted(dict_.items(), key=lambda kv: kv[1])
sorted_dict = dict(sorted_dict)
fig = px.bar(x=list(sorted_dict.keys()), y=list(sorted_dict.values()))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
fig = go.Figure()
y = difference(list(aux[:]))

fig.add_trace(go.Scatter(x=indexs, y=y, mode='lines',name='Observed'))

x, y = get_state(df, 'Distrito Federal')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Distrito Federal'))
x, y = get_state(df, 'Rondonia')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Rondonia'))

x, y = get_state(df, 'Amazonas')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Amazonas'))

x, y = get_state(df, 'Pernambuco')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Pernambuco'))
x, y = get_state(df, 'Bahia')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Bahia'))
x, y = get_state(df, 'Mato Grosso')
y = difference(y)

fig.add_trace(go.Scatter(x=x, y=y, mode='lines',name='Mato Grosso'))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
def fires_count(df):
    dict_ ={}
    for state in df['state'].unique():
        aux = df[df['state']==state]
        tmp = aux.groupby(['state'])['number'].agg('sum')
        dict_[state] = sum(tmp[:])
    return dict_


dict_ = fires_count(df)
sorted_dict = sorted(dict_.items(), key=lambda kv: kv[1])
sorted_dict = dict(sorted_dict)
fig = px.bar(x=list(sorted_dict.keys()), y=list(sorted_dict.values()))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
fig = px.line(x=indexs, y=res.seasonal, title='Residual component')
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
def fires_count_deseasonality(df):    
    for state in df['state'].unique():
        aux = df[df['state']==state]
        tmp = aux.groupby(['year', 'quarter'])['number'].agg('sum')
        y = difference(list(tmp[:]), 3)
        dict_[state] = sum(y)
    return dict_

dict_ = fires_count_deseasonality(df)
sorted_dict = sorted(dict_.items(), key=lambda kv: kv[1])
sorted_dict = dict(sorted_dict)
fig = px.bar(x=list(sorted_dict.keys()), y=list(sorted_dict.values()))
fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
})
fig.show()
