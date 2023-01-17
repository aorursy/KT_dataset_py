# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

from datetime import timedelta

# Data Visualization Liraries
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go
from IPython.display import display, Markdown

#hide warnings
import warnings
warnings.filterwarnings('ignore')
pyo.init_notebook_mode()

#display max columns of pandas dataframe
pd.set_option('display.max_columns', None)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
cov_dash = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-over-time.csv')
cov_dash_country = pd.read_csv('../input/uncover/UNCOVER_v4/UNCOVER/johns_hopkins_csse/johns-hopkins-covid-19-daily-dashboard-cases-by-country.csv')
cov_dash_country.head(3)
# Helper Function - Missing data check
def missing_data(data):
    missing = data.isnull().sum()
    available = data.count()
    total = (missing + available)
    percent = (data.isnull().sum()/data.isnull().count()*100).round(4)
    return pd.concat([missing, available, total, percent], axis=1, keys=['Missing', 'Available', 'Total', 'Percent']).sort_values(['Missing'], ascending=False)
# missing data check
missing_data(cov_dash_country)
cov_dash_country = cov_dash_country.drop(['people_tested','people_hospitalized','iso3'],axis = 1)
cov_dash_country[cov_dash_country.lat.isnull()]
covid_country = cov_dash_country.dropna()
covid_country.describe()
new_df = pd.DataFrame(covid_country[["confirmed","deaths","recovered","active"]].sum()).transpose()
new_df['mortality_rate'] = covid_country['mortality_rate'].mean()
new_df['incident_rate'] = covid_country['incident_rate'].mean()
new_df
# Create a world map to show distributions of users 
import folium
from folium.plugins import MarkerCluster
#empty map
world_map= folium.Map(tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(world_map)
#for each coordinate, create circlemarker of user percent
for i in range(len(covid_country)):
        lat = covid_country.iloc[i]['lat']
        long = covid_country.iloc[i]['long']
        radius=5
        popup_text = """Country : {}<br>
                    Confimed : {}<br>
                    Deaths : {}<br>
                    Recovered : {}<br>"""
        popup_text = popup_text.format(covid_country.iloc[i]['country_region'],
                                   covid_country.iloc[i]['confirmed'],
                                       covid_country.iloc[i]['deaths'],
                                       covid_country.iloc[i]['recovered']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
world_map
fig = px.choropleth(covid_country, locations="country_region",
                    color=covid_country["confirmed"], 
                    hover_name="country_region", 
                    hover_data=["deaths"],
                    locationmode="country names")

fig.update_layout(title_text="Confirmed Cases Heat Map (Log Scale)")
fig.update_coloraxes(colorscale="blues")

fig.show()
# Top 20 countries with highest confirmed cases
covid_country_top20=covid_country.sort_values("confirmed",ascending=False).head(20)

fig = px.bar(covid_country_top20, 
             x="country_region",
             y="confirmed",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Confirmed Cases',
            color='country_region')
fig.show()
# Top 20 countries with highest deaths
covid_country_top20=covid_country.sort_values("deaths",ascending=False).head(20)
fig = px.bar(covid_country_top20, 
             x="country_region",
             y="deaths",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Deaths',
            color='country_region')
fig.show()
# Top 20 countries with highest active cases
covid_country_top20=covid_country.sort_values("active",ascending=False).head(20)
fig = px.bar(covid_country_top20, 
             x="country_region",
             y="active",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Active Cases',
            color='country_region')
fig.show()
# Top 20 countries with highest recovered cases
covid_country_top20=covid_country.sort_values("recovered",ascending=False).head(20)
fig = px.bar(covid_country_top20, 
             x="country_region",
             y="recovered",
             orientation='v',
             height=800,
             title='Top 20 countries with COVID19 Recovered Cases',
            color='country_region')
fig.show()
corr= covid_country.corr()
plt.figure(figsize=(16,16))
sns.heatmap(corr,cmap="YlGnBu",annot=True)
#age_group = pd.read_csv('../input/covid19-in-india/AgeGroupDetails.csv')
india_covid19 = pd.read_csv('../input/covid19-in-india/covid_19_india.csv')
#hospital_beds = pd.read_csv('../input/covid19-in-india/HospitalBedsIndia.csv')
#individual_details = pd.read_csv('../input/covid19-in-india/IndividualDetails.csv')
#ICMR_details = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
#ICMR_labs = pd.read_csv('../input/covid19-in-india/ICMRTestingLabs.csv')
state_testing = pd.read_csv('../input/covid19-in-india/StatewiseTestingDetails.csv')
latlong = pd.read_csv('../input/latlong/LatLong.csv')
india_covid19.info()
india_covid19['State/UnionTerritory'].unique()
# Data Cleaning
india_covid19.rename(columns={'State/UnionTerritory': 'State', 'Cured': 'Recovered'}, inplace=True)
india_covid19['State'] = india_covid19['State'].replace({"Nagaland#": "Nagaland","Jharkhand#":"Jharkhand","Madhya Pradesh#":"Madhya Pradesh",
                                                        "Chandigarh":"Punjab", "Cases being reassigned to states":"Other", "Unassigned":"Other"})
india_covid19['State'].unique()
india_covid19['Confirmed'] = pd.to_numeric(india_covid19['Confirmed'], errors='coerce')
india_covid19['Confirmed']=india_covid19['Confirmed'].fillna(0)
india_covid19['Confirmed']=india_covid19['Confirmed'].astype('int')
india_covid19['Deaths'] = pd.to_numeric(india_covid19['Deaths'], errors='coerce')
india_covid19['Deaths']=india_covid19['Deaths'].fillna(0)
india_covid19['Deaths']=india_covid19['Deaths'].astype('int')
india_covid19['Recovered'] = pd.to_numeric(india_covid19['Recovered'], errors='coerce')
india_covid19['Recovered']=india_covid19['Recovered'].fillna(0)
india_covid19['Recovered']=india_covid19['Recovered'].astype('int')
india_covid19['ConfirmedIndianNational']= india_covid19['ConfirmedIndianNational'].replace("-", 0)
#india_covid19['ConfirmedIndianNational']=india_covid19['ConfirmedIndianNational'].fillna(0)
india_covid19['ConfirmedIndianNational']=india_covid19['ConfirmedIndianNational'].astype('int')
india_covid19['ConfirmedForeignNational']= india_covid19['ConfirmedForeignNational'].replace("-", 0)
#india_covid19['ConfirmedForeignNational']=india_covid19['ConfirmedForeignNational'].fillna(0)
india_covid19['ConfirmedForeignNational']=india_covid19['ConfirmedForeignNational'].astype('int')
#Cleaning up mixed date formats in Date column

# new data frame with split value columns 
new = india_covid19["Date"].str.split("/", n = 2, expand = True) 
  
# making separate first name column from new data frame 
india_covid19["Day"]= new[0] 
india_covid19['Day']=india_covid19['Day'].astype('int') 
# making separate last name column from new data frame 
india_covid19["Month"]= new[1]
india_covid19['Month']=india_covid19['Month'].astype('int') 
# making separate last name column from new data frame 
india_covid19["Year"]= 2020
#india_covid19['Year']=india_covid19['Year'].astype('int') 
india_covid19.tail()
india_covid19.describe()
#dropping original date column and creating a new cleaned date column
india_covid19 = india_covid19.drop(['Date'],axis = 1)
india_covid19['Date'] = india_covid19['Year'].map(str) + '-' + india_covid19['Month'].map(str) + '-' + india_covid19['Day'].map(str)
india_covid19 = india_covid19.drop(['Year','Month','Day'],axis = 1)
# Changing data types to datetime format
india_covid19["Date"]=pd.to_datetime(india_covid19["Date"],format='%Y%m%d', errors='ignore')
india_covid19["Time"]=pd.to_datetime(india_covid19["Time"], format='%H%M', errors='ignore')
india_covid19.info()
india_covid_final = india_covid19.merge(latlong)
india_covid_final.columns
india_covid_final

statewise = pd.pivot_table(india_covid_final, values=['Confirmed','Deaths','Recovered'], index='State', aggfunc='max')
statewise['Recovery Rate'] = statewise['Recovered']*100 / statewise['Confirmed']
statewise['Mortality Rate'] = statewise['Deaths']*100 /statewise['Confirmed']
statewise = statewise.sort_values(by='Confirmed', ascending= False)
statewise.style.background_gradient(cmap='YlOrRd')
state = pd.pivot_table(india_covid_final, values=['Confirmed','Deaths','Recovered','Latitude','Longitude'], index='State', aggfunc='max')
state['Recovery Rate'] = state['Recovered']*100 / state['Confirmed']
state['Mortality Rate'] = state['Deaths']*100 /state['Confirmed']
state = state.sort_values(by='Confirmed', ascending= False)
state.reset_index(level=0, inplace=True)
state.head()
#empty map
india_map= folium.Map(location=[21, 78], zoom_start=5,tiles="cartodbpositron")
marker_cluster = MarkerCluster().add_to(india_map)
#for each coordinate, create circlemarker of user percent
for i in range(len(state)):
        lat = state.iloc[i]['Latitude']
        long = state.iloc[i]['Longitude']
        radius=5
        popup_text = """State : {}<br>
                    Confimed : {}<br>
                    Deaths : {}<br>
                    Recovered : {}<br>"""
        popup_text = popup_text.format(state.iloc[i]['State'],
                                   state.iloc[i]['Confirmed'],
                                       state.iloc[i]['Deaths'],
                                       state.iloc[i]['Recovered']
                                   )
        folium.CircleMarker(location = [lat, long], radius=radius, popup= popup_text, fill =True).add_to(marker_cluster)
#show the map
india_map
# Data Cleaning
state_testing['TotalSamples']=state_testing['TotalSamples'].fillna(0)
state_testing['TotalSamples']=state_testing['TotalSamples'].astype('int')
state_testing['Positive']=state_testing['Positive'].fillna(0)
state_testing['Positive']=state_testing['Positive'].astype('int')
state_testing['Negative']=state_testing['Negative'].fillna(0)
#state_testing['Negative']=state_testing['Negative'].astype('int')
statewise_testing = pd.pivot_table(state_testing, values=['TotalSamples','Positive','Negative'], index='State', aggfunc='max')
statewise_testing['Positive_Case_Rate'] = statewise_testing['Positive']*100 / statewise_testing['TotalSamples']
statewise_testing['Positive_Case_Rate']=round(statewise_testing['Positive_Case_Rate'].astype('int'),2)
statewise_testing = statewise_testing.sort_values(by='TotalSamples', ascending= False)
statewise_testing.style.background_gradient(cmap='YlOrRd')
testing=state_testing.groupby('State')['TotalSamples'].max().sort_values(ascending=False).reset_index()
fig = px.bar(testing, 
             x="TotalSamples",
             y="State", 
             orientation='h',
             height=800,
             title='Statewise Testing',
            color='State')
fig.show()
#india_covid_final.to_csv('newcovid.csv',index=False)
plt.figure(figsize = (18,10))
figure = px.line(india_covid_final, x='Date', y='Confirmed', color='State')
figure.update_xaxes(rangeslider_visible=True)
pyo.iplot(figure)
statewise.columns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
std=StandardScaler()
#pd.set_option('display.float_format', lambda x: '%.6f' % x)
X=statewise[["Mortality Rate","Recovery Rate"]]
#Standard Scaling since K-Means Clustering is a distance based alogrithm
X=std.fit_transform(X) 
wcss=[]
sil=[]
for i in range(2,10):
    clf=KMeans(n_clusters=i,init='k-means++',random_state=64)
    clf.fit(X)
    labels=clf.labels_
    centroids=clf.cluster_centers_
    sil.append(silhouette_score(X, labels, metric='euclidean'))
    wcss.append(clf.inertia_)
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(16,5))
x=np.arange(2,10)
ax1.plot(x,wcss,marker='o')
ax1.set_xlabel("Number of Clusters")
ax1.set_ylabel("Within Cluster Sum of Squares (WCSS)")
ax1.set_title("Elbow Method")
x=np.arange(2,10)
ax2.plot(x,sil,marker='o')
ax2.set_xlabel("Number of Clusters")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score Method")
clf_final=KMeans(n_clusters=4,init='k-means++',random_state=32)
clf_final.fit(X)
statewise["Clusters"]=clf_final.predict(X)
cluster_summary = statewise.sort_values(by='Clusters', ascending= False)
cluster_summary.style.background_gradient(cmap='Purples').format("{:.2f}")
table1 = pd.pivot_table(statewise, values=['Confirmed', 'Deaths','Recovered'], 
                       index=['Clusters'], aggfunc=np.sum)
table1.style.background_gradient(cmap='Greens').format("{:.2f}")
table2 = pd.pivot_table(statewise, values=['Recovery Rate','Mortality Rate'], 
                       index=['Clusters'], aggfunc=np.mean)
table2.style.background_gradient(cmap='Blues').format("{:.2f}")
statewise["ClusterNo"] = statewise["Clusters"].astype(str)
fig = px.scatter(statewise, x="Recovery Rate", y="Mortality Rate", color="ClusterNo",
                 size='Deaths', hover_data=['Confirmed','Deaths','Recovered'])
fig.show()
# Day wise summary
df = india_covid_final.copy()
df['Date'] = pd.to_datetime(df['Date'],format='%Y/%m/%d')
india_covid_date = pd.pivot_table(df, values=['Confirmed','Deaths','Recovered'], index='Date', aggfunc='sum')
india_covid_date['Recovery Rate'] = india_covid_date['Recovered']*100 / india_covid_date['Confirmed']
india_covid_date['Mortality Rate'] = india_covid_date['Deaths']*100 /india_covid_date['Confirmed']
india_covid_date.reset_index(level=0, inplace=True)
pd.set_option('display.max_rows', india_covid_date.shape[0]+1)
india_covid_date

plt.figure(figsize = (18,10))

# Plot 
fig = px.line(india_covid_date, x='Date', y='Confirmed')

# Add one more plot
fig.add_scatter(x=india_covid_date['Date'], y=india_covid_date['Recovered'], mode='lines')

# Add one more plot
fig.add_scatter(x=india_covid_date['Date'], y=india_covid_date['Deaths'], mode='lines')

# Show plot w/ range slider
fig.update_xaxes(rangeslider_visible=True)
fig.show()
plt.figure(figsize = (18,10))

# Plot 
fig = px.line(india_covid_date, x='Date', y='Recovery Rate')

# Show plot w/ range slider
fig.update_xaxes(rangeslider_visible=True)
fig.show()
plt.figure(figsize = (18,10))

# Plot 
fig = px.line(india_covid_date, x='Date', y='Mortality Rate')

# Show plot w/ range slider
fig.update_xaxes(rangeslider_visible=True)
fig.show()
#india_covid_final.to_csv('newcovid.csv',index=False)
#india_covid_date.to_csv('covdate.csv',index=False)
from fbprophet import Prophet
fb_data = india_covid_date.copy()
fb_confirm = fb_data[['Date', 'Confirmed']]
fb_confirm = fb_confirm.rename(columns={'Date': 'ds',
                        'Confirmed': 'y'})

fb_confirm.head()
# Time Series Forecasting with Prophet
# set the uncertainty interval to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)
my_model.fit(fb_confirm)
# Creating a new dataframe
# Prophet provides the make_future_dataframe helper function
future_dates = my_model.make_future_dataframe(periods=2, freq='MS')
future_dates.tail()
forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
my_model.plot(forecast,
              uncertainty=True)
my_model.plot_components(forecast)
##Run all below code before using parametric or logistic prediction
! pip install pmdarima
! pip install arch
## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt

## for stationarity test
import statsmodels.api as sm

## for outliers detection
from sklearn import preprocessing, svm

## for autoregressive models
import pmdarima
import statsmodels.tsa.api as smt
import arch

## for deep learning
from tensorflow.keras import models, layers, preprocessing as kprocessing

## for prophet
from fbprophet import Prophet
pd.plotting.register_matplotlib_converters()

## for parametric fit
from scipy import optimize



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
'''
Plot ts with rolling mean and 95% confidence interval with rolling std.
:parameter
    :param ts: pandas Series
    :param window: num for rolling stats
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(15,5)):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(ts.name)
    plt.plot(ts[window:], label='Actual values', color="black")
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window), color="red")
    if plot_intervals:
        #mean_absolute_error = np.mean(np.abs((ts[window:] - rolling_mean[window:]) / ts[window:])) * 100
        #deviation = np.std(ts[window:] - rolling_mean[window:])
        #lower_bound = rolling_mean - (mean_absolute_error + 1.96 * deviation)
        #upper_bound = rolling_mean + (mean_absolute_error + 1.96 * deviation)
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        #plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        #plt.plot(lower_bound, 'r--')
        plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound, color='lightskyblue', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
        


'''
Test stationarity by:
    - running Augmented Dickey-Fuller test wiht 95%
    - plotting mean and variance of a sample from data
    - plottig autocorrelation and partial autocorrelation
'''
def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(15,10)):
    with plt.style.context(style='bmh'):
        ## set figure
        fig = plt.figure(figsize=figsize)
        ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
        acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))
        
        ## plot ts with mean/std of a sample from the first x% 
        dtf_ts = ts.to_frame(name="ts")
        sample_size = int(len(ts)*sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red", linestyle="--", linewidth=0.7)
        ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(ax=ts_ax, legend=False, color="red", linewidth=0.9)
        ts_ax.fill_between(x=dtf_ts.head(sample_size).index, y1=dtf_ts['lower'].head(sample_size), y2=dtf_ts['upper'].head(sample_size), color='lightskyblue')
        
        ## test stationarity (Augmented Dickey-Fuller)
        adfuller_test = sm.tsa.stattools.adfuller(ts, maxlag=maxlag, autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+' (p-value: '+str(p)+')')
        
        ## pacf (for AR) e acf (for MA) 
        smt.graphics.plot_pacf(ts, lags=maxlag, ax=pacf_ax, title="Partial Autocorrelation (for AR component)")
        smt.graphics.plot_acf(ts, lags=maxlag, ax=acf_ax, title="Autocorrelation (for MA component)")
        plt.tight_layout()    
   


'''
Defferenciate ts.
:parameter
    :param ts: pandas Series
    :param lag: num - diff[t] = y[t] - y[t-lag]
    :param order: num - how many times it has to differenciate: diff[t]^order = diff[t] - diff[t-lag] 
    :param drop_na: logic - if True Na are dropped, else are filled with last observation
'''
def diff_ts(ts, lag=1, order=1, drop_na=True):
    for i in range(order):
        ts = ts - ts.shift(lag)
    ts = ts[(pd.notnull(ts))] if drop_na is True else ts.fillna(method="bfill")
    return ts



'''
'''
def undo_diff(ts, first_y, lag=1, order=1):
    for i in range(order):
        (24168.04468 - 18256.02366) + a.cumsum()
        ts = np.r_[ts, ts[lag:]].cumsum()
    return ts



'''
Run Granger test on 2 series
'''
def test_2ts_casuality(ts1, ts2, maxlag=30, figsize=(15,5)):
    ## prepare
    dtf = ts1.to_frame(name=ts1.name)
    dtf[ts2.name] = ts2
    dtf.plot(figsize=figsize, grid=True, title=ts1.name+"  vs  "+ts2.name)
    plt.show()
    ## test casuality (Granger test) 
    granger_test = sm.tsa.stattools.grangercausalitytests(dtf, maxlag=maxlag, verbose=False)
    for lag,tupla in granger_test.items():
        p = np.mean([tupla[0][k][1] for k in tupla[0].keys()])
        p = round(p, 3)
        if p < 0.05:
            conclusion = "Casuality with lag "+str(lag)+" (p-value: "+str(p)+")"
            print(conclusion)
        


'''
Decompose ts into
    - trend component = moving avarage
    - seasonality
    - residuals = y - (trend + seasonality)
:parameter
    :param s: num - number of observations per season (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
'''
def decompose_ts(ts, s=250, figsize=(20,13)):
    decomposition = smt.seasonal_decompose(ts, freq=s)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid   
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(ts)
    ax[0].set_title('Original')
    ax[0].grid(True) 
    ax[1].plot(trend)
    ax[1].set_title('Trend')
    ax[1].grid(True)  
    ax[2].plot(seasonal)
    ax[2].set_title('Seasonality')
    ax[2].grid(True)  
    ax[3].plot(residual)
    ax[3].set_title('Residuals')
    ax[3].grid(True)
    return {"trend":trend, "seasonal":seasonal, "residual":residual}



'''
Find outliers using sklearn unsupervised support vetcor machine.
:parameter
    :param ts: pandas Series
    :param perc: float - percentage of outliers to look for
:return
    dtf with raw ts, outlier 1/0 (yes/no), numeric index
'''
def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda x: 1 if x==-1 else 0)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found "+str(sum(dtf_outliers["outlier"]==1)))
    ax.plot(dtf_outliers["index"], dtf_outliers["ts"], color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1]["index"], y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'], color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers



'''
Interpolate outliers in a ts.
'''
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5, title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black", label="interpolated", legend=True)
    plt.show()
    return ts_clean



###############################################################################
#                 MODEL DESIGN & TESTING - FORECASTING                        #
###############################################################################
'''
Split train/test from any given data point.
:parameter
    :param ts: pandas Series
    :param exog: array len(ts) x n regressors
    :param test: num or str - test size (ex. 0.20) or index position (ex. "yyyy-mm-dd", 1000)
:return
    ts_train, ts_test, exog_train, exog_test
'''
def split_train_test(ts, exog=None, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|", ts.index[split], "| test size:", perc, " ---")
    
    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=figsize)
        ts_train.plot(ax=ax[0], grid=True, title="Train", color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test", color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()
        
    ## split exog
    if exog is not None:
        exog_train = exog[0:split] 
        exog_test = exog[split:]
        return ts_train, ts_test, exog_train, exog_test
    else:
        return ts_train, ts_test
    


'''
Evaluation metrics for predictions.
:parameter
    :param dtf: DataFrame with columns raw values, fitted training values, predicted test values
:return
    dataframe with raw ts and forecast
'''
def utils_evaluate_forecast(dtf, title, plot=True, figsize=(20,13)):
    try:
        ## residuals
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        
        ## kpi
        residuals_mean = dtf["residuals"].mean()  #errore medio nel training
        residuals_std = dtf["residuals"].std()    #standard dev dell'errore nel training
        error_mean = dtf["error"].mean()   #errore medio nel test
        error_std = dtf["error"].std()     #standard dev dell'errore nel test
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["error"].apply(lambda x: x**2).mean() # mean squared error
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std
        
        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[["ts","model","residuals","conf_int_low","conf_int_up", 
                    "forecast","error","pred_int_low","pred_int_up"]]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
    


'''
Generate dates to index predictions.
:parameter
    :param start: str - "yyyy-mm-dd"
    :param end: str - "yyyy-mm-dd"
    :param n: num - length of index
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
'''
def utils_generate_indexdate(start, end=None, n=None, freq="D"):
    if end is not None:
        index = pd.date_range(start=start, end=end, freq=freq)
    else:
        index = pd.date_range(start=start, periods=n, freq=freq)
    index = index[1:]
    print("--- generating index date --> start:", index[0], "| end:", index[-1], "| len:", len(index), "---")
    return index



'''
Plot unknown future forecast.
'''
def utils_plot_forecast(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    ## entire series
    dtf[["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax[0], title="History + Future")
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3) 
          
    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:][["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax[1], title="Zoom on the last "+str(zoom)+" observations")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    plt.show()
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]



###############################################################################
#                           RANDOM WALK                                       #
###############################################################################
'''
Generate a Random Walk process.
:parameter
    :param y0: num - starting value
    :param n: num - length of process
    :param ymin: num - limit
    :param ymax: num - limit
'''
def utils_generate_rw(y0, n, sigma, ymin=None, ymax=None):
    rw = [y0]
    for t in range(1, n):
        yt = rw[t-1] + np.random.normal(0,sigma)
        if (ymax is not None) and (yt > ymax):
            yt = rw[t-1] - abs(np.random.normal(0,sigma))
        elif (ymin is not None) and (yt < ymin):
            yt = rw[t-1] + abs(np.random.normal(0,sigma))
        rw.append(yt)
    return rw
        

 
'''
Simulate Random Walk from params of a given ts: 
    y[t+1] = y[t] + wn~(0,σ)
'''
def simulate_rw(ts_train, ts_test, figsize=(15,10)):
    ## simulate train
    diff_ts = ts_train - ts_train.shift(1)
    rw = utils_generate_rw(y0=ts_train[0], n=len(ts_train), sigma=diff_ts.std(), ymin=ts_train.min(), ymax=ts_train.max())
    dtf_train = ts_train.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts_train.index, columns=["model"]), how='left', left_index=True, right_index=True)
    
    ## test
    rw = utils_generate_rw(y0=ts_train[-1], n=len(ts_test), sigma=diff_ts.std(), ymin=ts_train.min(), ymax=ts_train.max())
    dtf_test = ts_test.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts_test.index, columns=["forecast"]), 
                                                 how='left', left_index=True, right_index=True)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Random Walk Simulation")
    return dtf



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_rw(ts, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    diff_ts = ts - ts.shift(1)
    sigma = diff_ts.std()
    rw = utils_generate_rw(y0=ts[0], n=len(ts), sigma=sigma, ymin=ts.min(), ymax=ts.max())
    dtf = ts.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts.index, columns=["model"]), 
                                       how='left', left_index=True, right_index=True)
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = utils_generate_rw(y0=ts[-1], n=len(index), sigma=sigma, ymin=ts.min(), ymax=ts.max())
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf
    


###############################################################################
#                        AUTOREGRESSIVE                                       #
###############################################################################
'''
Fits Holt-Winters Exponential Smoothing: 
    y[t+i] = (level[t] + i*trend[t]) * seasonality[t]
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param trend: str - "additive" (linear), "multiplicative" (non-linear)
    :param seasonal: str - "additive" (ex. +100 every 7 days), "multiplicative" (ex. x10 every 7 days)
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param alpha: num - the alpha value of the simple exponential smoothing (ex 0.94)
:return
    dtf with predictons and the model
'''
def fit_expsmooth(ts_train, ts_test, trend="additive", seasonal="multiplicative", s=None, alpha=0.94, figsize=(15,10)):
    ## checks
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal is None) & (s is None) else "Seasonal parameters: "+str(seasonal)+" Seasonality every "+str(s)+" observations"
    print(check_seasonality)
    
    ## train
    #alpha = alpha if s is None else 2/(s+1)
    model = smt.ExponentialSmoothing(ts_train, trend=trend, seasonal=seasonal, seasonal_periods=s).fit(smoothing_level=alpha)
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Holt-Winters ("+str(alpha)+")")
    return dtf, model



'''
Fits SARIMAX (Seasonal ARIMA with External Regressors):  
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] + b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d: degree of differencing (to remove trend), q: order of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param exog_train: pandas dataframe or numpy array
    :param exog_test: pandas dataframe or numpy array
:return
    dtf with predictons and the model
'''
def fit_sarimax(ts_train, ts_test, order=(1,0,1), seasonal_order=(0,0,0,0), exog_train=None, exog_test=None, figsize=(15,10)):
    ## checks
    check_trend = "Trend parameters: No differencing" if order[1] == 0 else "Trend parameters: d="+str(order[1])
    print(check_trend)
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal_order[3] == 0) & (np.sum(seasonal_order[0:3]) == 0) else "Seasonal parameters: Seasonality every "+str(seasonal_order[3])+" observations"
    print(check_seasonality)
    check_exog = "Exog parameters: Not given" if (exog_train is None) & (exog_test is None) else "Exog parameters: number of regressors="+str(exog_train.shape[1])
    print(check_exog)
    
    ## train
    model = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit()
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1, exog=exog_test)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model


    
'''
Find best Seasonal-ARIMAX parameters.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    best model
'''
def find_best_sarimax(ts, seasonal=True, stationary=False, s=1, exog=None,
                      max_p=10, max_d=3, max_q=10,
                      max_P=10, max_D=3, max_Q=10):
    best_model = pmdarima.auto_arima(ts, exogenous=exog,
                                     seasonal=seasonal, stationary=stationary, m=s, 
                                     information_criterion='aic', max_order=20,
                                     max_p=max_p, max_d=max_d, max_q=max_q,
                                     max_P=max_P, max_D=max_D, max_Q=max_Q,
                                     error_action='ignore')
    print("best model --> (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
    return best_model.summary()



'''
Fits GARCH (Generalized Autoregressive Conditional Heteroskedasticity):  
    y[t+1] = m + e[t+1]
    e[t+1] = σ[t+1] * wn~(0,1)
    σ²[t+1] = c + (a0*σ²[t] + a1*σ²[t-1] +...+ ap*σ²[t-p]) + (b0*e²[t] + b1*e[t-1] + b2*e²[t-2] +...+ bq*e²[t-q])
:parameter
    :param ts: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p:lag order (AR), d:degree of differencing (to remove trend), q:order of moving average (MA)
'''
def fit_garch(ts_train, ts_test, order=(1,0,1), seasonal_order=(0,0,0,0), exog_train=None, exog_test=None, figsize=(15,10)):
    ## train
    arima = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit()
    garch = arch.arch_model(arima.resid, p=order[0], o=order[1], q=order[2], x=exog_train, dist='StudentsT', power=2.0, mean='Constant', vol='GARCH')
    model = garch.fit(update_freq=seasonal_order[3])
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.conditional_volatility
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.forecast(horizon=len(ts_test))

    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "GARCH ("+str(order[0])+","+str(order[2])+")" if order[0] != 0 else "ARCH ("+str(order[2])+")"
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_arima(ts, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    model = model.fit()
    dtf = ts.to_frame(name="ts")
    dtf["model"] = model.fittedvalues
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = model.forecast(len(index))
    dtf = dtf.append(preds.to_frame(name="forecast"))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf



###############################################################################
#                            RNN                                              #
###############################################################################
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metrics = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    
    ## training
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='black')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax11.plot(training.history[metric], label=metric)
    ax11.set_ylabel("Score", color='steelblue')
    ax11.legend()
    
    ## validation
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='black')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='black')
    for metric in metrics:
        ax22.plot(training.history['val_'+metric], label=metric)
    ax22.set_ylabel("Score", color="steelblue")
    plt.show()
    
    
    
'''
Preprocess a ts partitioning into X and y.
:parameter
    :param ts: pandas timeseries
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param scaler: sklearn scaler object - if None is fitted
    :param exog: pandas dataframe or numpy array
:return
    X, y, scaler
'''
def utils_preprocess_ts(ts, s, scaler=None, exog=None):
    ## scale
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ts_preprocessed = scaler.fit_transform(ts.values.reshape(-1,1)).reshape(-1)        
    
    ## create X,y for train
    ts_preprocessed = kprocessing.sequence.TimeseriesGenerator(data=ts_preprocessed, 
                                                               targets=ts_preprocessed, 
                                                               length=s, batch_size=1)
    lst_X, lst_y = [], []
    for i in range(len(ts_preprocessed)):
        xi, yi = ts_preprocessed[i]
        lst_X.append(xi)
        lst_y.append(yi)
    X = np.array(lst_X)
    y = np.array(lst_y)
    return X, y, scaler



'''
Get fitted values.
'''
def utils_fitted_lstm(ts, model, scaler, exog=None):
    ## scale
    ts_preprocessed = scaler.fit_transform(ts.values.reshape(-1,1)).reshape(-1) 
    
    ## create Xy, predict = fitted
    s = model.input_shape[-1]
    lst_fitted = [np.nan]*s
    for i in range(len(ts_preprocessed)):
        end_ix = i + s
        if end_ix > len(ts_preprocessed)-1:
            break
        X = ts_preprocessed[i:end_ix]
        X = np.array(X)
        X = np.reshape(X, (1,1,X.shape[0]))
        fit = model.predict(X)
        fit = scaler.inverse_transform(fit)[0][0]
        lst_fitted.append(fit)
    return np.array(lst_fitted)



'''
Predict ts using previous predictions.
'''
def utils_predict_lstm(ts, model, scaler, pred_ahead, exog=None):
    ## scale
    s = model.input_shape[-1]
    ts_preprocessed = list(scaler.fit_transform(ts[-s:].values.reshape(-1,1))) 
    
    ## predict, append, re-predict
    lst_preds = []
    for i in range(pred_ahead):
        X = np.array(ts_preprocessed[len(ts_preprocessed)-s:])
        X = np.reshape(X, (1,1,X.shape[0]))
        pred = model.predict(X)
        ts_preprocessed.append(pred)
        pred = scaler.inverse_transform(pred)[0][0]
        lst_preds.append(pred)
    return np.array(lst_preds)



'''
Fit Long short-term memory neural network.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    generator, scaler 
'''
def fit_lstm(ts_train, ts_test, model, exog=None, s=20, figsize=(15,5)):
    ## check
    print("Seasonality: using the last", s, "observations to predict the next 1")
    
    ## preprocess train
    X_train, y_train, scaler = utils_preprocess_ts(ts_train, scaler=None, exog=exog, s=s)
    
    ## lstm
    if model is None:
        model = models.Sequential()
        model.add( layers.LSTM(input_shape=X_train.shape[1:], units=50, activation='relu', return_sequences=False) )
        model.add( layers.Dense(1) )
        model.compile(optimizer='adam', loss='mean_absolute_error')
    
    ## train
    print(model.summary())
    training = model.fit(x=X_train, y=y_train, batch_size=1, epochs=100, shuffle=True, verbose=0, validation_split=0.3)
    utils_plot_keras_training(training)
    
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = utils_fitted_lstm(ts_train, training.model, scaler, exog)
    dtf_train["model"] = dtf_train["model"].fillna(method='bfill')
    
    ## test
    preds = utils_predict_lstm(ts_train[-s:], training.model, scaler, pred_ahead=len(ts_test), exog=None)
    dtf_test = ts_test.to_frame(name="ts").merge(pd.DataFrame(data=preds, index=ts_test.index, columns=["forecast"]),
                                                 how='left', left_index=True, right_index=True)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="LSTM (memory:"+str(s)+")")
    return dtf, training.model



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_lstm(ts, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    s = model.input_shape[-1]
    X, y, scaler = utils_preprocess_ts(ts, scaler=None, exog=None, s=s)
    training = model.fit(x=X, y=y, batch_size=1, epochs=100, shuffle=True, verbose=0, validation_split=0.3)
    dtf = ts.to_frame(name="ts")
    dtf["model"] = utils_fitted_lstm(ts, training.model, scaler, None)
    dtf["model"] = dtf["model"].fillna(method='bfill')
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = utils_predict_lstm(ts[-s:], training.model, scaler, pred_ahead=len(index), exog=None)
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf



###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet on Business Data:
    y = trend + seasonality + holidays
:parameter
    :param dtf_train: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param dtf_test: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param lst_exog: list - names of variables
    :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS" monthly start ...
:return
    dtf with predictons and the model
'''
def fit_prophet(dtf_train, dtf_test, lst_exog=None, model=None, freq="D", figsize=(15,10)):
    ## setup prophet
    if model is None:
        model = Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                holidays=None)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)
    
    ## train
    model.fit(dtf_train)
    
    ## test
    dtf_prophet = model.make_future_dataframe(periods=len(dtf_test), freq=freq, include_history=True)
    
    if model.growth == "logistic":
        dtf_prophet["cap"] = dtf_train["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(dtf_train[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-len(dtf_test):][lst_exog] = dtf_test[lst_exog].values
    
    dtf_prophet = model.predict(dtf_prophet)
    dtf_train = dtf_train.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    dtf_test = dtf_test.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'forecast', 'y':'ts'}).set_index("ds")
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Prophet")
    return dtf, model
    


'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param model: model object
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_prophet(dtf, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    model.fit(dtf)
    
    ## index
    index = utils_generate_indexdate(start=dtf["ds"].values[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    dtf_prophet = model.make_future_dataframe(periods=len(index), freq=freq, include_history=True)
    dtf_prophet = model.predict(dtf_prophet)
    dtf = dtf.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    preds = pd.DataFrame(data=index, columns=["ds"])
    preds = preds.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'forecast'}).set_index("ds")
    dtf = dtf.append(preds)
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf



###############################################################################
#                    PARAMETRIC CURVE FITTING                                 #
###############################################################################
'''
Fits a custom function.
:parameter
    :param X: array
    :param y: array
    :param f: function to fit (ex. logistic: f(X) = capacity / (1 + np.exp(-k*(X - midpoint)))
                                or gaussian: f(X) = a * np.exp(-0.5 * ((X-mu)/sigma)**2)   )
    :param kind: str - "logistic", "gaussian" or None
    :param p0: array or list of initial parameters (ex. for logistic p0=[np.max(ts), 1, 1])
:return
    optimal params
'''
def fit_curve(X, y, f=None, kind=None, p0=None):
    ## define f(x) if not specified
    if f is None:
        if kind == "logistic":
            f = lambda p,X: p[0] / (1 + np.exp(-p[1]*(X-p[2])))
        elif find == "gaussian":
            f = lambda p,X: p[0] * np.exp(-0.5 * ((X-p[1])/p[2])**2)
    
    ## find optimal parameters
    model, cov = optimize.curve_fit(f, X, y, maxfev=10000, p0=p0)
    return model
    


'''
Predict with optimal parameters.
'''
def utils_predict_curve(model, f, X):
    fitted = f(X, model[0], model[1], model[2])
    return fitted



'''
Plot parametric fitting.
'''
def utils_plot_parametric(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    ## entire series
    dtf["ts"].plot(marker=".", linestyle='None', ax=ax[0], title="Parametric Fitting", color="black")
    dtf["model"].plot(ax=ax[0], color="green", label="model", legend=True)
    dtf["forecast"].plot(ax=ax[0], grid=True, color="red", label="forecast", legend=True)
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)
   
    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:]["ts"].plot(marker=".", linestyle='None', ax=ax[1], color="black", 
                                  title="Zoom on the last "+str(zoom)+" observations")
    dtf.loc[zoom_idx:]["model"].plot(ax=ax[1], color="green")
    dtf.loc[zoom_idx:]["forecast"].plot(ax=ax[1], grid=True, color="red")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    plt.show()
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]



'''
Forecast unknown future.
:parameter
    :param ts: pandas series
    :param f: function
    :param model: list of optim params
    :param pred_ahead: number of observations to forecast (ex. pred_ahead=30)
    :param end: string - date to forecast (ex. end="2016-12-31")
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param zoom: for plotting
'''
def forecast_curve(ts, f, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    fitted = utils_predict_curve(model, f, X=np.arange(len(ts)))
    dtf = ts.to_frame(name="ts")
    dtf["model"] = fitted
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = utils_predict_curve(model, f, X=np.arange(len(ts)+1, len(ts)+1+len(index)))
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    utils_plot_parametric(dtf, zoom=zoom)
    return dtf
## For parametric fitting
from scipy import optimize
df_new = india_covid_date.copy()
dtf = df_new[['Date','Confirmed']]
dtf = dtf.set_index('Date')
## create new cases column
# new cases(t) = total(t) — total(t-1)
## create new cases column
dtf["new"] = dtf["Confirmed"] - dtf["Confirmed"].shift(1)
dtf["new"] = dtf["new"].fillna(method='bfill')
dtf.head()
dtf.tail()
#Various functions with random parameters

'''
Linear function: f(x) = a + b*x
'''
def f(x):
    return 1 + 28000*x

y_linear = f(x=np.arange(len(dtf)))
'''
Exponential function: f(x) = a + b^x
'''
def f(x):
    return 1 + 1.066**x

y_exponential = f(x=np.arange(len(dtf)))
'''
Logistic function: f(x) = a / (1 + e^(-b*(x-c)))
'''
def f(x): 
    return 6800000 / (1 + np.exp(-0.3*(x-150)))

y_logistic = f(x=np.arange(len(dtf)))
fig, ax = plt.subplots(figsize=(13,5))
ax.scatter(dtf["Confirmed"].index, dtf["Confirmed"].values, color="black")
ax.plot(dtf["Confirmed"].index, y_linear, label="linear", color="red")
ax.plot(dtf["Confirmed"].index, y_exponential, label="exponential", color="green")
ax.plot(dtf["Confirmed"].index, y_logistic, label="logistic", color="blue")
ax.legend()
plt.show()
#dtf.to_csv('param.csv',index=True)
#Guassian function with random parameters

'''
Gaussian function: f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)
'''
def f(x):
    return 100000 * np.exp(-0.015 * ((x-225)/6)**2)

y_gaussian = f(x=np.arange(len(dtf)))
fig, ax = plt.subplots(figsize=(13,5))
ax.bar(dtf["new"].index, dtf["new"].values, color="black")
ax.plot(dtf["new"].index, y_gaussian, color="red")
plt.show()
from scipy import optimize
## Logistic Function
# https://docs.scipy.org/doc/scipy/reference/optimize.html
'''
Function to fit. In this case logistic function:
    f(x) = capacity / (1 + e^-k*(x - midpoint) )
'''
def f(X, c, k, m):
    y = c / (1 + np.exp(-k*(X-m)))
    return y
## Fit
model_l = fit_curve(X=np.arange(len(dtf["Confirmed"])), y=dtf["Confirmed"].values, f=f, p0=[np.max(dtf["Confirmed"]), 1, 1])
model_l
## Forecast - Logistic Function
preds = forecast_curve(dtf["Confirmed"], f, model_l, pred_ahead=30, end=None, freq="D", zoom=14, figsize=(20,8))
# 120 days forecase to observe the peak/saturation point
preds90 = forecast_curve(dtf["Confirmed"], f, model_l, pred_ahead=120, end=None, freq="D", zoom=14, figsize=(20,8))
'''
Function to fit. In this case gaussian function:
    f(x) = a * e^(-0.5 * ((x-μ)/σ)**2)
'''
def f(X, a, b, c):
    y = a * np.exp(-0.4 * ((X-b)/c)**2)
    return y
model = fit_curve(X=np.arange(len(dtf["new"])), y=dtf["new"].values, f=f, p0=[1, np.mean(dtf["new"]), np.std(dtf["new"])])
model
## Forecast
preds = forecast_curve(dtf["new"], f, model, pred_ahead=120, end=None, freq="D", zoom=15, figsize=(20,8))
# 90 days forecast to observe the recovery phase
preds90p = forecast_curve(dtf["new"], f, model, pred_ahead=270, end=None, freq="D", zoom=14, figsize=(20,8))
sars = pd.read_csv("../input/sars-outbreak-2003-complete-dataset/sars_2003_complete_dataset_clean.csv")
sars.head()
# Data Cleaning
sars.rename(columns={'Cumulative number of case(s)': 'Total_Cases', 'Number of deaths': 'Deaths', 'Number recovered':'Recovered'}, inplace=True)
sars['Country'] = sars['Country'].replace({"Hong Kong SAR, China": "HongKong","Taiwan, China":"Taiwan","Republic of Ireland":"Ireland",
                                                        "Republic of Korea":"Korea", "Macao SAR, China":"Macao", "Russian Federation":"Russia","Viet Nam":"Vietnam"})
sars.columns
# Country level aggregation
sars_country = pd.pivot_table(sars, values=['Total_Cases', 'Deaths', 'Recovered'], index='Country', aggfunc='sum')
sars_country = sars_country.sort_values(by='Total_Cases', ascending= False)
sars_country.style.background_gradient(cmap='YlOrRd')
# Day wise summary
sars_df = sars.copy()
sars_df['Date'] = pd.to_datetime(sars_df['Date'],format='%Y/%m/%d')
sars_df_date = pd.pivot_table(sars_df, values=['Total_Cases', 'Deaths', 'Recovered'], index='Date', aggfunc='sum')
sars_df_date['Recovery Rate'] = sars_df_date['Recovered']*100 / sars_df_date['Total_Cases']
sars_df_date['Mortality Rate'] = sars_df_date['Deaths']*100 /sars_df_date['Total_Cases']
sars_df_date.reset_index(level=0, inplace=True)
## create new cases column
# new cases(t) = total(t) — total(t-1)
## create new cases column
sars_new = sars_df_date[['Date','Total_Cases']]
sars_new = sars_new.set_index('Date')
sars_new["new"] = sars_new["Total_Cases"] - sars_new["Total_Cases"].shift(1)
sars_new["new"] = sars_new["new"].fillna(method='bfill')
sars_new.head()
# Plot 
fig = px.bar(sars_new, x=sars_new.index, y='new')
# Show plot w/ range slider
fig.update_xaxes(rangeslider_visible=True)
fig.show()
plt.figure(figsize = (18,10))

# Plot 
fig = px.line(sars_df_date, x='Date', y='Total_Cases')

# Add one more plot
fig.add_scatter(x=sars_df_date['Date'], y=sars_df_date['Recovered'], mode='lines')

# Add one more plot
fig.add_scatter(x=sars_df_date['Date'], y=sars_df_date['Deaths'], mode='lines')

# Show plot w/ range slider
fig.update_xaxes(rangeslider_visible=True)
fig.show()
class Params:
    def __init__(self, c, n, sigma, gamma, r_zero):
        self.c = c
        self.N = n
        self.sigma = sigma
        self.gamma = gamma
        self.r_zero = r_zero
# Helper functions - Calculations

def seir_function(t, y, params):
    """
    dS / dt = -beta * S * I / N
    dE / dt = +beta * S * I / N - sigma * E
    dI / dt = +sigma * E - gamma * I + c * R * I / N
    dR / dt = gamma * I - c * R * I / N
    yprime = [dS / dt  dE / dt dI / dt   dRdt]
    input:
      t current time
      y vector of current soln values
      y(1) = S, y(2) = E, y(3) = I, y(4) = R
    parameters in "params"
      beta, N, sigma, gamma, c, R_zero_array(table of values)
    output: (col vector)
      yprime(1) = dS / dt
      yprime(2) = dE / dt
      yprime(3) = dI / dt
      yprime(4) = dR / dt
    """
    R_zero_array = params.r_zero
    
    min_t = np.min(R_zero_array[:, 0])
    max_t = np.max(R_zero_array[:, 0])
    t_val = max(min_t, min(t, max_t))
    
    R_zero = np.interp(t_val, R_zero_array[:, 0], R_zero_array[:, 1])
    
    gamma = params.gamma
    
    beta = R_zero * gamma
    
    N = params.N
    sigma = params.sigma
    c = params.c
    
    S = y[0]
    E = y[1]
    I = y[2]
    R = y[3]
    
    yprime = np.zeros(4)
    
    yprime[0] = -beta * S * I / N
    yprime[1] = +beta * S * I / N - sigma * E
    yprime[2] = +sigma * E - gamma * I + c * R * I / N
    yprime[3] = gamma * I - c * R * I / N
    return yprime
pd.set_option('display.max_rows', india_covid_date.shape[0]+1)
india_covid_date
"""
This code is a python version of the oringal Matlab/Octave code from Peter Forsyth
(see: https://cs.uwaterloo.ca/~paforsyt/SEIR.html)
parameters.py for SEIR model
S = susceptible population
E = Exposed (infected, not yet infectious)
I = Infectious (now can infect others)
R = Removed (got sick, now recovered and immune, or died :( )
N = total population = (S + E + I + R)
note: added cRI/N term:  disease
mutates, can cause reinfection, or immunity lost
This assumes that mutated form jumps to Infected population
Can also assume that mutated form jumps to Exposed population
For now, we assume c=0 (no mutation has been observed)
dS/dt = -beta*S*I/N
dE/dt = +beta*S*I/N - sigma*E
dI/dt = +sigma*E -gamma*I + c*R*I/N
dR/dt = gamma*I -c*R*I/N
this file passes seir_function in the calculations module to the ode solver
ode systen is specified in the calculations module
"""
import numpy as np
#import parameters as parameters
from scipy import integrate
#from calculations_module import seir_function
import matplotlib.pyplot as plt

#Reference:
# https://towardsdatascience.com/infection-modeling-part-1-87e74645568a
# https://en.wikipedia.org/wiki/COVID-19_pandemic_in_India
# https://sites.me.ucsb.edu/~moehlis/APC514/tutorials/tutorial_seasonal/node4.html
# https://www.idmod.org/docs/hiv/model-seir.html

S_0 = 168690 #India, excluding initial infected, exposed population,

I_0 = 3  # initial infected

E_0 = 27. * I_0  # initial exposed

R_0 = 3  # initial recovered (not to be confused with R_zero, below)
# initially, no one has recovered

c = 0.0  # no mutation (yet)
# maybe later...still a mystery



N = S_0 + I_0 + E_0 + R_0  # N = total population

sigma = 1. / 5.1  # https://doi.org/10.1056/NEJMoa2001316 (2020).

gamma = 1. / 18.  # https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/newsâ€“wuhan-coronavirus
"""
 R_zero = number of people infected by each infectious person
          this has nothing to do with "R" = removed above
          or R_0 (initial value of recovered)
          but is common terminology (confusing, but usual notation)
     time dependent, starts offf large, than drops with
         time due to public health actions (i.e. quarantine, social distancing)
    R_zero > 1, cases increase
    R_zero < 1 cases peak and then drop off 
      R_zero declining with time https://www.nature.com/articles/s41421-020-0148-0
      beta = R_zero*gammma (done in "seir.m" )
 
     table of:   time(days)  R_zero
                  ....     ....
                  ....     ....
                  ....     ....
       linearly interpolate between times
       Note: this is different from Wang et al (2020), which assumes
             piecewise constant values for R_zero
"""
r_zero_array = np.zeros([6, 2])
r_zero_array[0, :] = [0.0,  3.0]# t=0 days    R_zero = 3.0
r_zero_array[1, :] = [20.0,  2.6]# t = 60 days R_zero = 2.6
r_zero_array[2, :] = [70.0,  1.9]# t = 70 days R_zero = 1.9
r_zero_array[3, :] = [84.0,  1.0]# t = 84 days R_zero = 1.0
r_zero_array[4, :] = [90.0,  .50]# t = 90 days R_zero = .50
r_zero_array[5, :] = [1000, .50]# t = 1000 days R_zero =.50

params = Params(c, N, sigma, gamma, r_zero_array)

t_0 = 0
tspan = np.linspace(t_0, 181, 180)  # time in days

y_init = np.zeros(4)
y_init[0] = S_0
y_init[1] = E_0
y_init[2] = I_0
y_init[3] = R_0


def seir_with_params(t, y):
    return seir_function(t, y, params)


r = integrate.ode(seir_with_params).set_integrator("dopri5")
r.set_initial_value(y_init, t_0)
y = np.zeros((len(tspan), len(y_init)))
y[0, :] = y_init  # array for solution
for i in range(1, 180):
    y[i, :] = r.integrate(tspan[i])
    if not r.successful():
        raise RuntimeError("Could not integrate")


fig, axes = plt.subplots(ncols=2)
axes[0].plot(tspan, y[:, 0], color="b", label="S: susceptible")
axes[1].plot(tspan, y[:, 1], color="r", label="E: exposed")
axes[0].set(xlabel="time (days)", ylabel="S: susceptible")
axes[1].set(xlabel="time (days)", ylabel="E: exposed")

axes[0].legend()
axes[1].legend()
plt.show()

fig, axes = plt.subplots(ncols=2)
axes[0].plot(tspan, y[:, 2], color="b", label="I: infectious")
axes[1].plot(tspan, y[:, 3], color="r", label="R: recovered")
axes[0].set(xlabel="time (days)", ylabel="I: infectious")
axes[1].set(xlabel="time (days)", ylabel="R: recovered")
axes[0].legend()
axes[1].legend()
plt.show()

total_cases = y[:, 1] + y[:, 2] + y[:, 3]
total_cases_active = y[:, 1] + y[:, 2]

fig, ax = plt.subplots()
ax.plot(tspan, total_cases, color="b", label="E+I+R: Total cases")
ax.plot(tspan, total_cases_active, color="r", label="E+I: Active cases")
ax.set(xlabel="time (days)", ylabel="Patients", title='Cumulative and active cases')
plt.legend()
plt.show()

nsteps = np.size(tspan)
S_end = y[nsteps - 1, 0]
E_end = y[nsteps - 1, 1]
I_end = y[nsteps - 1, 2]
R_end = y[nsteps - 1, 3]

total = S_end + E_end + I_end + R_end

from datetime import datetime, timedelta
specific_date = datetime(2020, 1, 30)
new_date = specific_date + timedelta(tspan[nsteps-1])


print('time (days): % 2d' %tspan[nsteps-1])

print('total population: % 2d' %total)

print('initial infected: % 2d' %I_0)

print('total cases (E+I+R) at t= % 2d : % 2d' %(tspan[nsteps-1], E_end + I_end + R_end))

print('Recovered at t=  % 2d : % 2d \n' %(tspan[nsteps-1], R_end))
print('Infected (infectious) at t= % 2d : % 2d \n' %(tspan[nsteps-1],I_end))
print('Exposed (non-infectious) at t= % 2d : % 2d \n ' %(tspan[nsteps-1], E_end))
print('Susceptable at t= % 2d : % 2d \n ' %(tspan[nsteps-1], S_end))
print ('Data for t= ' ,new_date)
def fit_newcurve(X, y, f=None, kind=None, p0=None):
    ## define f(x) if not specified
    if f is None:
        if kind == "logistic":
            f = lambda p,X: p[0] / (1 + np.exp(-p[1]*(X-p[2])))
        elif find == "gaussian":
            f = lambda p,X: p[0] * np.exp(-0.5 * ((X-p[1])/p[2])**2)
    
    ## find optimal parameters
    model, cov = optimize.curve_fit(f, X, y, maxfev=10000, p0=p0)
    return model
    
