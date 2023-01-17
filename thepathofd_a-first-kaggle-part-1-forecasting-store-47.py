import warnings
warnings.filterwarnings("ignore")

# DATA MANIPULATION
import numpy as np # linear algebra
import random as rd # generating random numbers
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime # manipulating date formats
from operator import add # elementwise addition

# VIZUALIZATION
import matplotlib.pyplot as plt # basic plotting
import seaborn # for prettier plots
import folium # plotting data on interactive maps

# UNSUPERVISED LEARNING
from sklearn.cluster import AgglomerativeClustering as AggClust # Hierarchical Clustering
from scipy.cluster.hierarchy import ward,dendrogram # Hierarchical Clustering + Dendograms

# TIME SERIES
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
# Reading daily transfers per store
sales=pd.read_csv('../input/transactions.csv')

# Reading store list
stores=pd.read_csv('../input/stores.csv')
stores.type=stores.type.astype('category')

# Adding information about the stores
sales=pd.merge(sales,stores,how='left')

# Reading the holiday and events schedule
holidays=pd.read_csv('../input/holidays_events.csv')

# Formatting the dates properly
sales['date']=sales.date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))
holidays['date']=holidays.date.apply(lambda x:datetime.datetime.strptime(x, '%Y-%m-%d'))

# Isolating events that do not correspond to holidays
events=holidays.loc[holidays.type=='Event']
holidays=holidays.loc[holidays.type!='Event']

# Extracting year, week and day
sales['year'],sales['week'],sales['day']=list(zip(*sales.date.apply(lambda x: x.isocalendar())))

# Creating a categorical variable showing weekends
sales['dayoff']=[x in [6,7] for x in sales.day]

# Adjuusting this variable to show all holidays
for (d,t,l,n) in zip(holidays.date,holidays.type,holidays.locale,holidays.locale_name):
  if t!='Work Day':
    if l=='National':
      sales.loc[sales.date==d,'dayoff']=True
    elif l=='Regional':
      sales.loc[(sales.date==d)&(sales.state==n),'dayoff']=True
    else:
      sales.loc[(sales.date==d)&(sales.city==n),'dayoff']=True
  else:
    sales.loc[(sales.date==d),'dayoff']=False

sales.info()
sales.head(20)
ts=sales.loc[sales['store_nbr']==47,['date','transactions']].set_index('date')
ts=ts.transactions.astype('float')
plt.figure(figsize=(12,12))
plt.title('Daily transactions in store #47')
plt.xlabel('time')
plt.ylabel('Number of transactions')
plt.plot(ts);
plt.figure(figsize=(12,12))
plt.plot(ts.rolling(window=30,center=False).mean(),label='Rolling Mean');
plt.plot(ts.rolling(window=30,center=False).std(),label='Rolling sd');
plt.legend();
def test_stationarity(timeseries):
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)

test_stationarity(ts)
plt.figure(figsize=(12,6))
autocorrelation_plot(ts);
plt.figure(figsize=(12,6))
autocorrelation_plot(ts);
plt.xlim(xmax=100);
plt.figure(figsize=(12,6))
autocorrelation_plot(ts);
plt.xlim(xmax=10);

result = arma_order_select_ic(ts,max_ar=10, max_ma=10, ic=['aic','bic'], trend='c', fit_kw=dict(method='css',maxiter=500))
print('The bic prescribes these (p,q) parameters : {}'.format(result.bic_min_order))
print('The aic prescribes these (p,q) parameters : {}'.format(result.aic_min_order))
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title('bic results')
seaborn.heatmap(result.bic);
plt.subplot(1,2,2)
plt.title('aic results')
seaborn.heatmap(result.aic);
pdq=(5,0,5)
model = ARIMA(ts, order = pdq, freq='W')
model_fit = model.fit(disp=False,method='css',maxiter=100)
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig,axes = plt.subplots(nrows=1, ncols=2,figsize=(12,6))
residuals.plot(ax=axes[0])
residuals.plot(kind='kde',ax=axes[1]);
residuals.describe().T
plt.figure(figsize=(12,6))
plt.subplot
plt.plot(ts);
plt.plot(model_fit.fittedvalues,alpha=.7);
forecast_len=30
size = int(len(ts)-forecast_len)
train, test = ts[0:size], ts[size:len(ts)]
history = [x for x in train]
predictions = list()

print('Starting the ARIMA predictions...')
print('\n')
for t in range(len(test)):
    model = ARIMA(history, order = pdq, freq='W');
    model_fit = model.fit(disp=0);
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(float(yhat))
    obs = test[t]
    history.append(obs)
print('Predictions finished.\n')
    

predictions_series = pd.Series(predictions, index = test.index)
plt.figure(figsize=(12,12))
plt.title('Store 47 : Transactions')
plt.xlabel('Date')
plt.ylabel('Transactions')
plt.plot(ts[-2*forecast_len:], 'o', label='observed');
plt.plot(predictions_series, '-o',label='rolling one-step out-of-sample forecast');
plt.legend(loc='upper right');
plt.figure(figsize=(12,12))
x=abs(ts[-forecast_len:]-predictions_series)
seaborn.distplot(x,norm_hist=False,rug=True,kde=False);
# The function takes a store number, an integer for the rolling means window and a bool
# for whether or not to split the working days and off days
def plot_store_transactions(store_viz,n=30,split=False):
    temp=sales.loc[sales.store_nbr==store_viz].set_index('date')
    plt.figure(figsize=(12,6))
    if split:
        ax1=plt.subplot(1,2,1)
        plt.scatter(temp.loc[~temp.dayoff].index,
                    temp.loc[~temp.dayoff].transactions,label='working days')
        plt.scatter(temp.loc[temp.dayoff].index,
                    temp.loc[temp.dayoff].transactions,label='off days')
        plt.legend()
        plt.title('Daily transactions. Store {}, Type {}, Cluster {}'.format(store_viz,
                                                                        list(stores.loc[stores.store_nbr==store_viz,'type'])[0],
                                                                        list(stores.loc[stores.store_nbr==store_viz,'cluster'])[0])
                 )
        ax2=plt.subplot(1,2,2,sharey=ax1,sharex=ax1)
        plt.plot(temp.loc[~temp.dayoff,'transactions'].rolling(window=n).mean(),label='working days')
        plt.plot(temp.loc[temp.dayoff,'transactions'].rolling(window=n).mean(),label='off days')
        plt.legend()
        plt.title('Store {}: {} day rolling means'.format(store_viz,n))
        plt.setp(ax2.get_yticklabels(), visible=False)
    else:
        ax1=plt.subplot(1,2,1)
        plt.scatter(temp.index,temp.transactions)
        plt.title('Daily transactions. Store {}, Type {}, Cluster {}'.format(store_viz,
                                                                        list(stores.loc[stores.store_nbr==store_viz,'type'])[0],
                                                                        list(stores.loc[stores.store_nbr==store_viz,'cluster'])[0])
                 )
        ax2=plt.subplot(1,2,2,sharey=ax1)
        plt.plot(temp.transactions.rolling(window=n).mean())
        plt.title('Store {}: {} day rolling means'.format(store_viz,n))
        plt.setp(ax2.get_yticklabels(), visible=False)
plt.show()
plot_store_transactions(47,30,True)
def plot_store_transactions_type(typ):
    typ_stores=stores.loc[stores.type==typ,'store_nbr']
    n=len(typ_stores)
    m=1
    for x in range(1,6):
        if (n-1) in range((x-1)**2,x**2):
            m=x
    plt.figure(figsize=(15,15))
    for x in range(n):
        nbr=typ_stores.iloc[x]
        ax1 = plt.subplot(m,m,x+1)
        ax1.scatter(sales.loc[(~sales.dayoff)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(~sales.dayoff)&(sales.store_nbr==nbr),'transactions'])
        ax1.scatter(sales.loc[(sales.dayoff)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(sales.dayoff)&(sales.store_nbr==nbr),'transactions'])
        plt.title('Store {}, Type {}, Cluster {}'.format(nbr,
                                                         list(stores.loc[stores.store_nbr==nbr,'type'])[0],
                                                         list(stores.loc[stores.store_nbr==nbr,'cluster'])[0])
             )
        plt.suptitle(' Type {} stores'.format(typ),fontsize=25)
    plt.show()
plot_store_transactions_type('A')
plot_store_transactions_type('B')
plot_store_transactions_type('C')
plot_store_transactions_type('D')
plot_store_transactions_type('E')
def plot_store_transactions_cluster(clust):
    clust_stores=stores.loc[stores.cluster==clust,'store_nbr']
    n=len(clust_stores)
    m=1
    for x in range(1,6):
        if (n-1) in range((x-1)**2,x**2):
            m=x
    plt.figure(figsize=(15,15))
    for x in range(n):
        nbr=clust_stores.iloc[x]
        ax1 = plt.subplot(m,m,x+1)
        ax1.scatter(sales.loc[(~sales.dayoff)&(sales.cluster==clust)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(~sales.dayoff)&(sales.cluster==clust)&(sales.store_nbr==nbr),'transactions'])
        ax1.scatter(sales.loc[(sales.dayoff)&(sales.cluster==clust)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(sales.dayoff)&(sales.cluster==clust)&(sales.store_nbr==nbr),'transactions'])
        plt.title('Store {}, Type {}, Cluster {}'.format(nbr,
                                                         list(stores.loc[stores.store_nbr==nbr,'type'])[0],
                                                         list(stores.loc[stores.store_nbr==nbr,'cluster'])[0])
             )
        plt.suptitle(' Cluster {} stores'.format(clust),fontsize=25)
    plt.show()
plot_store_transactions_cluster(13)
#creating a table with means and std of transaction volume per type of day per store
Means1=sales.groupby(['store_nbr','dayoff']).transactions.agg(['mean','std']).unstack(level=1)

#creating a table with means and std of transaction volume per day of the week per store
Means2=sales.groupby(['store_nbr','day']).transactions.agg(['mean','std']).unstack(level=1)

# Creating a table  with the daily average of transaction volume per store 
sales_by_store=sales.groupby(['store_nbr']).transactions.sum()/sales.groupby(['store_nbr']).transactions.count()
# Creating a new columns with ratio of transactions of the day / daily average
sales['normalized']=[v/sales_by_store[s] for (s,v) in zip(sales.store_nbr,sales.transactions)]

#creating a table with means and std of normalized transaction volume per type of day per store
Means1_norm=sales.groupby(['store_nbr','dayoff']).normalized.agg(['mean','std']).unstack(level=1)
#creating a table with means and std of normalized transaction volume per day of the week per store
Means2_norm=sales.groupby(['store_nbr','day']).normalized.agg(['mean','std']).unstack(level=1)
plt.figure(figsize=(12,16))
plt.subplot(2,2,1)
plt.scatter(Means1.iloc[:,0],Means1.iloc[:,1])
plt.xlabel('Means of working days')
plt.ylabel('Means of holidays')
plt.plot([0,5000],[0,5000])
plt.title('Comparing mean by type of day')
plt.subplot(2,2,2)
plt.scatter(Means1.iloc[:,2],Means1.iloc[:,3])
plt.xlabel('Standard dev. of working days')
plt.ylabel('Standard dev. of holidays')
plt.plot([0,1000],[0,1000])
plt.title('Comparing std by type of day');
plt.subplot(2,2,3)
plt.scatter(Means1.iloc[:,0],Means1.iloc[:,2])
plt.xlabel('Means of working days')
plt.ylabel('Standard dev. of working days')
plt.plot([0,5000],[0,500])
plt.title('Comparing mand and std for working days')
plt.subplot(2,2,4)
plt.scatter(Means1.iloc[:,1],Means1.iloc[:,3])
plt.xlabel('Means of holidays days')
plt.ylabel('Standard dev. of holidays')
plt.plot([0,5000],[0,1000]);
plt.title('Comparing mand and std for holidays');
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
seaborn.heatmap(Means2.iloc[:,0:7],cmap='Oranges');
plt.subplot(1,2,2)
seaborn.heatmap(Means2.iloc[:,7:14],cmap='Oranges');
plt.figure(figsize=(12,16))
plt.subplot(2,2,1)
plt.scatter(Means1_norm.iloc[:,0],Means1_norm.iloc[:,1])
plt.xlabel('Means of working days')
plt.ylabel('Means of holidays')
plt.title('Comparing mean by type of day')
plt.subplot(2,2,2)
plt.scatter(Means1_norm.iloc[:,2],Means1_norm.iloc[:,3])
plt.xlabel('Standard dev. of working days')
plt.ylabel('Standard dev. of holidays')
plt.title('Comparing std by type of day');
plt.subplot(2,2,3)
plt.scatter(Means1_norm.iloc[:,0],Means1_norm.iloc[:,2])
plt.xlabel('Means of working days')
plt.ylabel('Standard dev. of working days')
plt.title('Comparing mean and std for working days')
plt.subplot(2,2,4)
plt.scatter(Means1_norm.iloc[:,1],Means1_norm.iloc[:,3])
plt.xlabel('Means of holidays days')
plt.ylabel('Standard dev. of holidays')
plt.title('Comparing mean and std for holidays');
plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
seaborn.heatmap(Means2_norm.iloc[:,0:7],cmap='Oranges');
plt.subplot(1,2,2)
seaborn.heatmap(Means2_norm.iloc[:,7:14],cmap='Oranges');
fig=plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 1, 1)
dendrogram(ward(Means2_norm),ax=ax)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=8)
plt.show()
clustering=AggClust(n_clusters=6)
cluster=clustering.fit_predict(Means2_norm)
stores['new_cluster']=cluster

def plot_store_transactions_new_cluster(clust):
    clust_stores=stores.loc[stores['new_cluster']==clust,'store_nbr']
    n=len(clust_stores)
    m=1
    for x in range(1,10):
        if (n-1) in range((abs(x-1))**2,x**2):
            m=x
    plt.figure(figsize=(15,15))
    for x in range(n):
        nbr=clust_stores.iloc[x]
        ax1 = plt.subplot(m,m,x+1)
        ax1.scatter(sales.loc[(~sales.dayoff)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(~sales.dayoff)&(sales.store_nbr==nbr),'transactions'])
        ax1.scatter(sales.loc[(sales.dayoff)&(sales.store_nbr==nbr),'date'].values,
                sales.loc[(sales.dayoff)&(sales.store_nbr==nbr),'transactions'])
        plt.title('Store {}, Type {}, Cluster {}'.format(nbr,
                                                         list(stores.loc[stores.store_nbr==nbr,'type'])[0],
                                                         list(stores.loc[stores.store_nbr==nbr,'cluster'])[0])
             )
    plt.show()
plot_store_transactions_new_cluster(1)
plot_store_transactions_new_cluster(2)
plot_store_transactions_new_cluster(3)
store_locations={
 'Ambato' : [-1.2543408,-78.6228504],
 'Babahoyo' : [-1.801926,-79.53464589999999],
 'Cayambe' : [0.025,-77.98916659999998],
 'Cuenca' : [-2.9001285,-79.0058965],
 'Daule' : [-1.86218,-79.97766899999999],
 'El Carmen' : [-0.266667, -79.4333],
 'Esmeraldas' : [0.9681788999999998,-79.6517202],
 'Guaranda' : [-1.5904721,-78.9995154],
 'Guayaquil' : [-2.1709979,-79.92235920000002],
 'Ibarra' : [0.3391763,-78.12223360000002],
 'Latacunga' : [-0.7754954,-78.52064999999999],
 'Libertad' : [-2.2344458,-79.91122430000001],
 'Loja' : [-4.0078909,-79.21127690000003],
 'Machala' : [-3.2581112,-79.9553924],
 'Manta' : [-0.9676533,-80.70891010000003],
 'Playas' : [-2.6284683,-80.38958860000002],
 'Puyo' : [-1.4923925,-78.00241340000002],
 'Quevedo' : [-1.0225124,-79.46040349999998],
 'Quito' : [-0.1806532,-78.46783820000002],
 'Riobamba' : [-1.6635508,-78.65464600000001],
 'Salinas' : [-2.2233633,-80.958462],
 'Santo Domingo' : [-0.2389045,-79.17742679999998]
}

# Defining a color dictionary
col={'A':'red','B':'blue','C':'green','D':'pink','E':'beige',
     0:'red',1:'blue',2:'green',3:'darkblue',4:'pink',5:'beige'}

#
def add_city_map(name,typ):
    folium.Marker(
         location=list(map(add,store_locations.get(name),[(0.5-rd.random())/20,(0.5-rd.random())/20])),
         icon=folium.Icon(color=col.get(typ), icon='shopping-cart'),
    ).add_to(map_Ecuador)

map_Ecuador=folium.Map(location=[-1.233333, -78.516667],zoom_start=7)

# Enabling clustering (also replace map_ecuador by store_cluster in the add_city_map function)
# from folium.plugins import MarkerCluster
#store_cluster=MarkerCluster().add_to(map_Ecuador)

[add_city_map(x,y) for x,y in zip(stores.city,stores.type)]
map_Ecuador
help(folium.Icon)
map_Ecuador=folium.Map(location=[-1.233333, -78.516667],zoom_start=7)

# Enabling clustering (also replace map_ecuador by store_cluster in the add_city_map function)
# from folium.plugins import MarkerCluster
#store_cluster=MarkerCluster().add_to(map_Ecuador)

[add_city_map(x,y) for x,y in zip(stores.city,stores.new_cluster)]
map_Ecuador