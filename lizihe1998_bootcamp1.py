from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
!ls ../input/corona-virus-report
country_wise= pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')

country_wise= country_wise.replace('',np.nan).fillna(0)
country_wise.head()
full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
full_grouped.head()
japan=full_grouped[full_grouped['Country/Region']=='Japan']
japan.head()
japan["New active"] = japan["Active"].diff()
temp = japan.melt(id_vars="Date", value_vars=['New cases', 'New deaths'],
                 var_name='Case', value_name='Count')
fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=1200,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
temp = japan.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
fig = px.area(temp, x="Date", y="Count", color='Case', height=600, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

import numpy as np

# to store and process data in dataframe
import pandas as pd

# to interface with operating system
import os

# for offline ploting
import matplotlib.pyplot as plt

# interactive visualization
import plotly.express as px
import seaborn as sns; sns.set()


from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

import plotly.graph_objs as go
import plotly.figure_factory as ff

from plotly.subplots import make_subplots

# for trendlines
import statsmodels
# Create an empty list
files = []

# Fill the list with the file names of the CSV files in the Kaggle folder
for dirname, _, filenames in os.walk('../input/econfin'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))

# Sort the file names
files = sorted(files)

# Output the list of sorted file names
files
series = [pd.read_csv(f, na_values=['.']) for f in files]

# Define series name, which becomes the dictionary key
series_name = ['btc','cpi','gold','snp','high_yield_bond','inv_grade_bond','moderna','employment','tesla_robinhood','trea_20y_bond','trea_10y_yield','tesla','fed_bs','wti']

# series name = dictionary key, series = dictionary value
series_dict = dict(zip(series_name, series))

#1. Snp
snp = series_dict['snp']
snp['Date'] = pd.to_datetime(snp['Date']) #转换成date
snp.rename(columns={'Adj Close':'snp'}, inplace=True)
snp['snp_return'] = snp['snp'].pct_change()
snp['snp_volatility_1m'] = (snp['snp_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation
snp['snp_volatility_1y'] = (snp['snp_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year
snp = snp[['Date','snp','snp_return','snp_volatility_1m','snp_volatility_1y']]
# Calculate 1-month forward cumulative returns
snp['one_month_forward_snp_return'] = snp['snp_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 2. Bitcoin
btc = series_dict['btc']
btc['Date'] = pd.to_datetime(btc['Date'])
btc.rename(columns={'Adj Close':'btc'}, inplace=True) #换名称
btc['btc_return'] = btc['btc'].pct_change() #相较前一天的增长
btc['btc_volatility_1m'] = (btc['btc_return'].rolling(20).std())*(20)**(1/2) 
btc['btc_volatility_1y'] = (btc['btc_return'].rolling(252).std())*(252)**(1/2) 
btc = btc[['Date','btc','btc_return','btc_volatility_1m','btc_volatility_1y']]
btc['one_month_forward_btc_return'] = btc['btc_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 3. Gold
gold = series_dict['gold']
gold['Date'] = pd.to_datetime(gold['DATE'])
gold.rename(columns={'GOLDPMGBD228NLBM':'gold'}, inplace=True)
gold['gold_lag1'] = gold['gold'].shift(1) #前一天的
gold['gold_lag2'] = gold['gold'].shift(2)
gold['gold'] = gold['gold'].fillna(gold['gold_lag1']) #如果有na,用前一天的来换
gold['gold'] = gold['gold'].fillna(gold['gold_lag2'])#如果还有，用两天前的来换
gold["gold"] = gold["gold"].astype('float64')
gold['gold_return'] = gold['gold'].pct_change()
gold['gold_volatility_1m'] = (gold['gold_return'].rolling(20).std())*(20)**(1/2) 
gold['gold_volatility_1y'] = (gold['gold_return'].rolling(252).std())*(252)**(1/2) 
gold = gold[['Date','gold','gold_return','gold_volatility_1m','gold_volatility_1y']]
gold['one_month_forward_gold_return'] = gold['gold_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 4. Crude Oil WTI
wti = series_dict['wti']
wti['Date'] = pd.to_datetime(wti['DATE'])
wti.rename(columns={'WTISPLC':'wti'}, inplace=True)
wti['wti_return'] = wti['wti'].pct_change()
wti['wti_volatility_1m'] = wti['wti_return'].rolling(20).std()*(20)**(1/2)
wti['wti_volatility_1y'] = wti['wti_return'].rolling(252).std()*(252)**(1/2)
wti = wti[['Date','wti','wti_return','wti_volatility_1m','wti_volatility_1y']]
wti['one_month_forward_wti_return'] = wti['wti_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

#5. Inflation
cpi = series_dict['cpi']
cpi['Date'] = pd.to_datetime(cpi['DATE'])
cpi.rename(columns={'CUUR0000SEHE':'cpi'}, inplace=True)
cpi = cpi[['Date','cpi']]
nk=pd.read_csv('../input/nikkie225/N225.csv')
nk.head()
snp.head()
nk['Date'] = pd.to_datetime(nk['Date']) #change to date
nk.rename(columns={'Adj Close':'nk'}, inplace=True)
nk['nk_return'] = nk['nk'].pct_change()
nk['nk_volatility_1m'] = (nk['nk_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation
nk['nk_volatility_1y'] = (nk['nk_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year
nk = nk[['Date','nk','nk_return','nk_volatility_1m','nk_volatility_1y']]
# Calculate 1-month forward cumulative returns
nk['one_month_forward_nk_return'] = nk['nk_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]
nk.head()
jpy=pd.read_csv('../input/jpyusd/JPY.csv')

jpy['Date'] = pd.to_datetime(jpy['Date']) #change to date
jpy.rename(columns={'Adj Close':'jpy'}, inplace=True)
jpy['jpy_return'] = jpy['jpy'].pct_change()
jpy['jpy_volatility_1m'] = (jpy['jpy_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation
jpy['jpy_volatility_1y'] = (jpy['jpy_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year
jpy = jpy[['Date','jpy','jpy_return','jpy_volatility_1m','jpy_volatility_1y']]
# Calculate 1-month forward cumulative returns
jpy['one_month_forward_jpy_return'] = jpy['jpy_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]
jpy.head()
# Import datasets with Pandas method read_csv
nber_recession_indicator_day = pd.read_csv('../input/jpnrecp/JPNRECP.csv')
nber_recession_indicator_day.head()

# Convert data types
nber_recession_indicator_day["Date"] = pd.to_datetime(nber_recession_indicator_day["DATE"])
nber_recession_indicator_day.rename(columns={'JPNRECP':'recession'}, inplace=True)

# Subset data columns
nber_recession_indicator_day = nber_recession_indicator_day[["Date","recession"]]
nber_recession_indicator_day.head()
# Merge datasets together
asset_classes = [btc,cpi,gold,wti,jpy]

baseline = pd.merge(nk,nber_recession_indicator_day,how='left',left_on='Date', right_on="Date") #how=left把snp的东西都留下

for asset_class in asset_classes:
    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on="Date")

# Backfilling missing values,  
baseline.loc[baseline.Date >= '2020-03-01', "recession"] = 1 # if the date is > 3/1, make resession =1
baseline["recession"] = baseline["recession"].fillna(0).astype(bool)

baseline.info()
# Index Date
baseline.set_index('Date', inplace=True)
baseline.tail()
baseline_yearly_return = baseline[["nk_return", "btc_return", "gold_return", "wti_return",'jpy_return']].dropna().resample('Y').sum().reset_index()
#resample('Y')is to make the data to yearly basis
print(baseline_yearly_return['Date'].min()) # 2010-12-31
baseline_yearly_return.tail()
# Re-sample the dataset every year and calculate the mean of 1-year volatility
baseline_yearly_volatility_1y = baseline[["nk_volatility_1y", "btc_volatility_1y", "gold_volatility_1y", 
                                          "wti_volatility_1y","jpy_volatility_1y"]].dropna().resample('Y').mean().reset_index()

baseline_yearly = baseline_yearly_return.merge(baseline_yearly_volatility_1y, left_on='Date', right_on='Date')

baseline_yearly.head()
baseline_returns = baseline[["nk_return", "btc_return", "gold_return",  "wti_return", "jpy_return","recession"]]

sns.pairplot(baseline_returns, hue="recession")
baseline_corr = baseline[['nk_return', 'nk_volatility_1y', 'btc_return', 'btc_volatility_1y',
                         'gold_return', 'gold_volatility_1y', 'wti_return', 'wti_volatility_1y','jpy_return', 'jpy_volatility_1y',
                         'recession']].dropna().corr()

fig, ax = plt.subplots(figsize=(20,10)) 
sns.heatmap(baseline_corr, annot=True, ax = ax)
fig = px.scatter(baseline[baseline['nk_return'].notnull()], x='nk_volatility_1m', 
                     color='recession', y='one_month_forward_nk_return', 
                     trendline = 'ols')
fig.update_layout(title= ' nk volatility vs one-month forward return', xaxis_title='', yaxis_title='')
fig.show()
print(pd.Timestamp(1586957400000))
print(pd.Timestamp(1595597400000))

