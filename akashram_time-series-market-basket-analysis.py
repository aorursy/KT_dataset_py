import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go
data = pd.read_csv("../input/BreadBasket_DMS.csv")

data.shape
data.head(10)
data['Date'] = pd.to_datetime(data['Date'],format='%Y-%m-%d')
data['Time'] = pd.to_datetime(data['Time'])

data['Times'] = data['Time'].dt.time
#tot_tr = data.groupby('Date', as_index=True)['Transaction'].sum().reset_index()

tot_tr1 = data.groupby(['Date', 'Transaction']).size().reset_index()

tot_tr1.columns = ['Date', 'Transaction', 'count']
tot_tr = tot_tr1.groupby('Date', as_index=True)['count'].sum().reset_index()

tot_tr.columns = ['Date', 'Transaction']
tot1 = tot_tr.iloc[:56]

tot2 = tot_tr.iloc[56:]
b = pd.to_datetime('2017-01-02 00:00:00',format='%Y-%m-%d')

c = pd.to_datetime('2016-12-25 00:00:00',format='%Y-%m-%d')

d = pd.to_datetime('2016-12-26 00:00:00',format='%Y-%m-%d')
tot2.loc[-2] = [c, 0]

tot2.loc[-1] = [d, 0] 

tot2.index = tot2.index + 2

tot2 = tot2.sort_index()

tot2 = tot2.rename(index={0: 56, 1:57})
#tot1.append(tot2)

tot_tr2 = tot1.append(tot2)
tot1 = tot_tr2.iloc[:64]

tot2 = tot_tr2.iloc[64:]
tot2.loc[-1] = [b, 0]

tot2.index = tot2.index + 1

tot2 = tot2.sort_index()

tot2 = tot2.rename(index={0:64})
#tot1.append(tot2)

tot_tr3 = tot1.append(tot2)
tot_tr3 = tot_tr3.replace({'Transaction': {0: 1}})
import matplotlib.pyplot as plt

import seaborn as sns

color = sns.color_palette()



%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



xtr = tot_tr.loc[tot_tr['Transaction'].idxmax()][0]

ytr = tot_tr['Transaction'].max()



data1 = [go.Scatter(

          x=tot_tr.Date,

          y=tot_tr.Transaction)]



layout = go.Layout(

   showlegend=False,

    annotations=[

        dict(

            x=xtr,

            y=ytr,

            xref='x',

            yref='y',

            text='Highest',

            showarrow=True,

            arrowhead=7,

            ax=0,

            ay=-40

        )

    ]

)



fig = go.Figure(data=data1, layout=layout)

py.iplot(fig, filename='multiple-annotation')
data['hour'] = data['Time'].dt.hour

#data.head(10)
s = data['hour'].value_counts().reset_index()
plt.rcParams['figure.figsize']=(20,20)

g = sns.jointplot(x=s['index'], y=s['hour'], data=s, kind="kde", color = "m", size=12, aspect=3);

g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")

g.ax_joint.collections[0].set_alpha(0)

g.set_axis_labels("Hour", "Count of transactions");
item_cnt = data['Item'].value_counts().reset_index()

item_cnt.columns = ['Item', 'Count']

item_cnt = item_cnt[item_cnt.Item != 'NONE']

item_cnt = item_cnt.head(15)

#item_cnt
import plotly.graph_objs as go



labels = item_cnt['Item'].values.tolist()

values = item_cnt['Count'].values.tolist()



trace = go.Pie(labels=labels, values=values)



py.iplot([trace], filename='item_chart')
item_cnt = data.groupby(['Item', 'hour']).size().reset_index()

item_cnt.columns = ['Item', 'Hour', 'Count']

item_cnt = item_cnt[item_cnt.Item != 'NONE']

item_cnt = item_cnt.sort_values(by='Count', ascending=False)

#item_cnt#.head(100)
item_cnt_cf = data.groupby(['Item', 'hour']).size().reset_index()

item_cnt_cf.columns = ['Item', 'Hour', 'Count']

item_cnt_cf = item_cnt[item_cnt['Item'].isin(['Coffee', 'Bread', 'Tea'])]

#item_cnt_cf
g = sns.PairGrid(item_cnt_cf, hue="Item", height=10, aspect=1)

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
item_cnt_cake = data.groupby(['Item', 'hour']).size().reset_index()

item_cnt_cake.columns = ['Item', 'Hour', 'Count']

item_cnt_cake = item_cnt[item_cnt['Item'].isin(['Cake', 'Pastry', 'Sandwich', 'Medialuna'])]



g = sns.PairGrid(item_cnt_cake, hue="Item", height=10, aspect=1)

g.map_diag(plt.hist)

g.map_offdiag(plt.scatter)

g.add_legend();
train= tot_tr3[0:100] 

test= tot_tr3[100:]
train_series = pd.Series(train.Transaction.values, index=pd.date_range(train.Date.min(),train.Date.max(),freq='D'))

test_series = pd.Series(test.Transaction.values, index=pd.date_range(test.Date.min(),test.Date.max(),freq='D'))
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 18, 12

plt.plot(train_series, label='train')

plt.plot(test_series, label='test')

plt.title('Train and test Graph')

plt.legend()

plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose

ts_trs = pd.Series(tot_tr3.Transaction.values, index=pd.date_range(tot_tr3.Date.min(),tot_tr3.Date.max(),freq='D'))

deompose = seasonal_decompose(ts_trs, freq=24)

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 15, 10

deompose.plot()
tot_tr3['moving_average'] = tot_tr3['Transaction'].rolling(window=3, center=False).mean()

plt.figure(figsize=(20,10))

plt.plot(tot_tr3.Date, tot_tr3.Transaction,'-',color='black',alpha=0.3)

plt.plot(tot_tr3.Date, tot_tr3.moving_average,color='b')

plt.title('Transaction and Moving Average Smoothening')

plt.legend()

plt.show()
y_hat_avg = test.copy()

y_hat_avg['moving_avg_forecast'] = train['Transaction'].rolling(60).mean().iloc[-1]

plt.figure(figsize=(18,12))

plt.plot(train['Date'], train['Transaction'], label='Train')

plt.plot(test['Date'], test['Transaction'], label='Test')

plt.plot(y_hat_avg['Date'],y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')

plt.legend(loc='best')

plt.show()
from math import sqrt

from sklearn.metrics import mean_squared_error

rms = sqrt(mean_squared_error(test.Transaction, y_hat_avg.moving_avg_forecast))

print(rms)
tot_tr3['ewma'] = tot_tr3['Transaction'].ewm(halflife=3, ignore_na=False,min_periods=0, adjust=True).mean()

plt.figure(figsize=(20,10))

plt.plot(tot_tr3.Transaction,'-',color='black',alpha=0.3)

plt.plot(tot_tr3.ewma,color='g')

plt.title('Transaction and Exponential Smoothening')

plt.legend()

plt.show()
#from statsmodels.tsa.api import SimpleExpSmoothing, Holt

#from statsmodels.tsa.api import ExponentialSmoothing

#import statsmodels.tsa.holtwinters.ExponentialSmoothing

#y_hat_avg = test.copy()

#fit2 = SimpleExpSmoothing(np.asarray(train['Transaction'])).fit(smoothing_level=0.6,optimized=False)

#y_hat_avg['SES'] = fit2.forecast(len(test))

#plt.figure(figsize=(16,8))

#plt.plot(train['Transaction'], label='Train')

#plt.plot(test['Transaction'], label='Test')

#plt.plot(y_hat_avg['SES'], label='SES')

#plt.legend(loc='best')

#plt.show()
# Dickey Fuller test for Stationarity

    

from statsmodels.tsa.stattools import adfuller

def ad_fuller_test(ts):

    dftest = adfuller(ts, autolag='AIC')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value','#Lags Used', 'Number of Observations Used'])

    for key,value in dftest[4].items():

        dfoutput['Critical Value (%s)'%key] = value

        print(dfoutput)
# Plot rolling stats

    

def plot_rolling_stats(ts):

    a = tot_tr3['Transaction']

    ts_log = (a) 

    rolling_mean = ts_log.rolling(12).mean()

    rolling_std = ts_log.rolling(12).std()

    orig = plt.plot(ts, color='blue',label='Original')

    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')

    std = plt.plot(rolling_std, color='black', label = 'Rolling Std')

    plt.legend(loc='best')

    plt.title('Rolling Mean & Standard Deviation')

    plt.show(block=False)
log_series =  (tot_tr3.Transaction.values)

log_series1 = np.log(tot_tr3.Transaction.values)

tot_tr3['log_series1'] = np.log(tot_tr3.Transaction.values)

ad_fuller_test(log_series)
ad_fuller_test(log_series1)
plt.figure(figsize=(18,12))

plt.plot(log_series1, 'blue', label='normal')

plt.plot(log_series, 'red', label='log')



plt.legend(loc='best')

plt.show()
from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 12

a = tot_tr3['Transaction']

plot_rolling_stats(a)
log_series_shift = log_series1[1:] - log_series1[:-1]

log_series_shift = log_series_shift[~np.isnan(log_series_shift)]
ad_fuller_test(log_series_shift)
plt.plot(log_series_shift)
plot_rolling_stats(log_series_shift)
import statsmodels.api as sm

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

from statsmodels.tsa.arima_model import ARIMA, ARMA



from pandas.plotting import autocorrelation_plot

fig = plt.figure(figsize=(12,12))

ax1 = fig.add_subplot(211)

fig = sm.graphics.tsa.plot_acf(log_series_shift , lags=20, ax=ax1)

ax2 = fig.add_subplot(212)

fig = sm.graphics.tsa.plot_pacf(log_series_shift, lags=20, ax=ax2)
def auto_arima(param_max=1,series=pd.Series(),verbose=True):

    # Define the p, d and q parameters to take any value 

    # between 0 and param_max

    p = d = q = range(0, param_max+1)



    # Generate all different combinations of seasonal p, d and q triplets

    pdq = [(x[0], x[1], x[2]) for x in list(itertools.product(p, d, q))]

    

    model_resuls = []

    best_model = {}

    min_aic = 10000000

    for param in pdq:

        try:

            mod = sm.tsa.ARIMA(series, order=param)



            results = mod.fit()

            

            if verbose:

                print('ARIMA{}- AIC:{}'.format(param, results.aic))

            model_resuls.append({'aic':results.aic,

                                 'params':param,

                                 'model_obj':results})

            if min_aic>results.aic:

                best_model={'aic':results.aic,

                            'params':param,

                            'model_obj':results}

                min_aic = results.aic

        except Exception as ex:

            print(ex)

    if verbose:

        print("Best Model params:{} AIC:{}".format(best_model['params'],

              best_model['aic']))  

        

    return best_model, model_resuls
import itertools

import matplotlib.dates as mdates

from sklearn.model_selection import TimeSeriesSplit



def arima_gridsearch_cv(series, cv_splits=2,verbose=True,show_plots=True):

    # prepare train-test split object

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    

    # initialize variables

    splits = []

    best_models = []

    all_models = []

    i = 1

    

    # loop through each CV split

    for train_index, test_index in tscv.split(series):

        print("*"*20)

        print("Iteration {} of {}".format(i,cv_splits))

        i = i + 1

        

        # print train and test indices

        if verbose:

            print("TRAIN:", train_index, "TEST:", test_index)

        splits.append({'train':train_index,'test':test_index})

        

        # split train and test sets

        train_series = series.ix[train_index]

        test_series = series.ix[test_index]

        

        print("Train shape:{}, Test shape:{}".format(train_series.shape,

              test_series.shape))

        

        # perform auto arima

        _best_model, _all_models = auto_arima(series=train_series)

        best_models.append(_best_model)

        all_models.append(_all_models)

        

        # display summary for best fitting model

        if verbose:

            print(_best_model['model_obj'].summary())

        results = _best_model['model_obj']

        

        if show_plots:

            # show residual plots

            residuals = pd.DataFrame(results.resid)

            residuals.plot()

            plt.title('Residual Plot')

            plt.show()

            residuals.plot(kind='kde')

            plt.title('KDE Plot')

            plt.show()

            print(residuals.describe())

        

            # show forecast plot

            fig, ax = plt.subplots(figsize=(18, 4))

            fig.autofmt_xdate()

            ax = train_series.plot(ax=ax)

            test_series.plot(ax=ax)

            fig = results.plot_predict(test_series.index.min(), 

                                       test_series.index.max(), 

                                       dynamic=True,ax=ax,

                                       plot_insample=False)

            plt.title('Forecast Plot ')

            plt.legend()

            plt.show()



            # show error plot

            insample_fit = list(results.predict(train_series.index.min()+1, 

                                                train_series.index.max(),

                                                typ='levels')) 

            plt.plot((np.exp(train_series.ix[1:].tolist())-\

                             np.exp(insample_fit)))

            plt.title('Error Plot')

            plt.show()

    return {'cv_split_index':splits,

            'all_models':all_models,

            'best_models':best_models}
tot_tr3c = tot_tr3.copy()

tot_tr3c = tot_tr3c.set_index('Date')

pd.to_datetime(tot_tr3c.index)

results_dict = arima_gridsearch_cv(tot_tr3c.log_series1,cv_splits=5)
model = ARIMA(log_series_shift, order=(1,0,0))  

results_AR = model.fit()

plt.plot(log_series1)

plt.plot(results_AR.fittedvalues, color='red')
transactions = pd.DataFrame(tot_tr)

transactions.columns = ['ds', 'y']
from fbprophet import Prophet



m = Prophet()

m.fit(transactions)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)

forecast.head(10)
py.iplot([

    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')

])
m = Prophet(changepoint_prior_scale=2.5)

m.fit(transactions)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
# Calculate root mean squared error.



print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )

py.iplot([

    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')

])
m = Prophet(changepoint_prior_scale=2.5)

m.add_seasonality(name='monthly', period=30.5, fourier_order=5)

m.fit(transactions)

future = m.make_future_dataframe(periods=365)

forecast = m.predict(future)
# Calculate root mean squared error.



print('RMSE: %f' % np.sqrt(np.mean((forecast.loc[:1682, 'yhat']-transactions['y'])**2)) )

py.iplot([

    go.Scatter(x=transactions['ds'], y=transactions['y'], name='y'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='yhat'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], fill='tonexty', mode='none', name='upper'),

    go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='none', name='lower'),

    go.Scatter(x=forecast['ds'], y=forecast['trend'], name='Trend')

])
item_cnt = data['Item'].value_counts().reset_index()

item_cnt.columns = ['Item', 'Count']

item_cnt = item_cnt[item_cnt.Item != 'NONE']
objects = (list(item_cnt['Item'].head(n=20)))

y_pos = np.arange(len(objects))

performance = list(item_cnt['Count'].head(n=20))

plt.bar(y_pos, performance, align='center', alpha=0.5)

plt.xticks(y_pos, objects, rotation='vertical')

plt.ylabel('Item count')

plt.title('Item Sales distribution')
total_item_count = item_cnt['Item'].count()

item_cnt['item_perc'] = item_cnt['Count']/total_item_count

item_cnt['total_perc'] = item_cnt.item_perc.cumsum()

ict = item_cnt.head(10)
import plotly.graph_objs as go



trace = go.Table(

    header=dict(values=['Item', 'Count', 'Item Perc', 'Total Perc'],

                line = dict(color='#7D7F80'),

                fill = dict(color='#a1c3d1'),

                align = ['left'] * 5),

    cells=dict(values=[ict['Item'].values.tolist(),

                       ict['Count'].values.tolist(),

                       ict['item_perc'].values.tolist(),

                       ict['total_perc'].values.tolist()],

               line = dict(color='#7D7F80'),

               fill = dict(color='#EDFAFF'),

               align = ['left'] * 5))



layout = dict(width=600, height=600)

data1 = [trace]

fig = dict(data=data1, layout=layout)

py.iplot(fig, filename = 'styled_table')
from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules
items = []

for i in data['Transaction'].unique():

    itemlist = list(set(data[data["Transaction"]==i]['Item']))

    if len(itemlist) > 0:

        items.append(itemlist)
### We need to one-hot encode the items as the library we are working with accepts only 1,0, True or False



from mlxtend.preprocessing import TransactionEncoder



oht = TransactionEncoder()

oht_item = oht.fit(items).transform(items)

df1 = pd.DataFrame(oht_item, columns=oht.columns_)

frequent_itemset = apriori(df1, use_colnames=True, min_support=0.02)

rules = association_rules(frequent_itemset, metric="lift", min_threshold=0.5)
rules.head(10)
plt.figure(figsize=(12,12))

plt.scatter(rules['support'],rules['confidence'],marker='*',edgecolors='grey',s=100,c=rules['lift'])

plt.colorbar(label='Lift')

plt.xlabel('support')

plt.ylabel('confidence')
def draw_graph(rules, rules_to_show):

    import networkx as nx  

    plt.figure(figsize=(10,8))

    G1 = nx.DiGraph()

    

    color_map=[]

    N = 400

    colors = np.random.rand(N)    

    strs=[]

    for i in range(rules_to_show):

        strs.append('R'+str(i))

    

    for i in range (rules_to_show):      

        G1.add_nodes_from(["R"+str(i)])

         

        for a in rules.iloc[i]['antecedents']:

                

            G1.add_nodes_from([a])        

            G1.add_edge(a, "R"+str(i), color=colors[i] , weight = 1.5)

        

        for c in rules.iloc[i]['consequents']:         

            G1.add_nodes_from([c])

            G1.add_edge("R"+str(i), c, color=colors[i],  weight=1.5)

    

    for node in G1:

        if node in strs:

            color_map.append('black')

        else:

            color_map.append('red')

            

    edges = G1.edges()

    colors = [G1[u][v]['color'] for u,v in edges]

    weights = [G1[u][v]['weight'] for u,v in edges]



    pos = nx.spring_layout(G1, k=16, scale=1)

    nx.draw(G1, pos, edges=edges, node_color = color_map, edge_color=colors, width=weights, font_size=14, with_labels=False)            

    for p in pos:  # raise text positions

        pos[p][1] += 0.08

    nx.draw_networkx_labels(G1, pos)
draw_graph(rules,len(rules))