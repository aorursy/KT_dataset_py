# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# datetime operations
from datetime import timedelta

# for numerical analyiss
import numpy as np

# to store and process data in dataframe
import pandas as pd

# basic visualization package
import matplotlib.pyplot as plt

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# color pallette
# Hexademical code RRGGBB (True Black #000000, True White #ffffff)
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 
# list files
!ls ../input/corona-virus-report
# Grouped by day, country
full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped.info()
full_grouped.head(10)

# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
full_grouped.info()
full_grouped.head(10)

uae = full_grouped[full_grouped['Country/Region']=='United Arab Emirates']
uae.info()
uae= uae.reset_index()
uae.head(10)
# Collapse Country, Date observations to Date observations and reindex
uae1 = uae.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()
uae1.head(50)
# Melt the data by the value_vars
uae1 = uae.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],
                 var_name='Case', value_name='Count')
uae1.head()
# Plot
fig = px.area(uae1, x="Date", y="Count", color='Case', height=600, width=700,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
uae["New active"] = uae["Active"].diff()
uae.info()
uae.head(10)
uae2 = uae.melt(id_vars="Date", value_vars=['New cases', 'New deaths'],
                 var_name='Case', value_name='Count')
uae2.head()

fig = px.area(uae2, x="Date", y="Count", color='Case', height=600, width=800,
             title='Cases over time', color_discrete_sequence = [rec, dth, act])
fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()
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
# Read the CSV files through list comprehension, which can be broken into three parts
# 1. OUTPUT EXPRESSION [pd.read_csv(f, na_values=['.'])] --- Note: this turns character '.' values into missing value
# 2. INPUT SEQUENCE [for f] 
# 3. CONDITION (OPTIONAL) [in files] 
series = [pd.read_csv(f, na_values=['.']) for f in files]

# Define series name, which becomes the dictionary key
series_name = ['btc','cpi','gold','snp','high_yield_bond','inv_grade_bond','moderna','employment','tesla_robinhood','trea_20y_bond','trea_10y_yield','tesla','uae','fed_bs','wti']

# series name = dictionary key, series = dictionary value
series_dict = dict(zip(series_name, series))
# 1. S&P 
snp = series_dict['snp']
snp['Date'] = pd.to_datetime(snp['Date'])
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
btc.rename(columns={'Adj Close':'btc'}, inplace=True)
btc['btc_return'] = btc['btc'].pct_change()
btc['btc_volatility_1m'] = (btc['btc_return'].rolling(20).std())*(20)**(1/2) 
btc['btc_volatility_1y'] = (btc['btc_return'].rolling(252).std())*(252)**(1/2) 
btc = btc[['Date','btc','btc_return','btc_volatility_1m','btc_volatility_1y']]
btc['one_month_forward_btc_return'] = btc['btc_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 3. Gold
gold = series_dict['gold']
gold['Date'] = pd.to_datetime(gold['DATE'])
gold.rename(columns={'GOLDPMGBD228NLBM':'gold'}, inplace=True)
gold['gold_lag1'] = gold['gold'].shift(1)
gold['gold_lag2'] = gold['gold'].shift(2)
gold['gold'] = gold['gold'].fillna(gold['gold_lag1'])
gold['gold'] = gold['gold'].fillna(gold['gold_lag2'])
gold["gold"] = gold["gold"].astype('float64')
gold['gold_return'] = gold['gold'].pct_change()
gold['gold_volatility_1m'] = (gold['gold_return'].rolling(20).std())*(20)**(1/2) 
gold['gold_volatility_1y'] = (gold['gold_return'].rolling(252).std())*(252)**(1/2) 
gold = gold[['Date','gold','gold_return','gold_volatility_1m','gold_volatility_1y']]
gold['one_month_forward_gold_return'] = gold['gold_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 4. High Yield Bond
high_yield_bond = series_dict['high_yield_bond']
high_yield_bond['Date'] = pd.to_datetime(high_yield_bond['Date'])
high_yield_bond.rename(columns={'Adj Close':'high_yield_bond'}, inplace=True)
high_yield_bond['high_yield_bond_return'] = high_yield_bond['high_yield_bond'].pct_change()
high_yield_bond['high_yield_bond_volatility_1m'] = (high_yield_bond['high_yield_bond_return'].rolling(20).std())*(20)**(1/2)
high_yield_bond['high_yield_bond_volatility_1y'] = (high_yield_bond['high_yield_bond_return'].rolling(252).std())*(252)**(1/2)
high_yield_bond = high_yield_bond[['Date','high_yield_bond','high_yield_bond_return','high_yield_bond_volatility_1m',
                                   'high_yield_bond_volatility_1y']]
high_yield_bond['one_month_forward_high_yield_bond_return'] = high_yield_bond['high_yield_bond_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 5. Investment Grade Bond
inv_grade_bond = series_dict['inv_grade_bond']
inv_grade_bond['Date'] = pd.to_datetime(inv_grade_bond['Date'])
inv_grade_bond.rename(columns={'Adj Close':'inv_grade_bond'}, inplace=True)
inv_grade_bond['inv_grade_bond_return'] = inv_grade_bond['inv_grade_bond'].pct_change()
inv_grade_bond['inv_grade_bond_volatility_1m'] = (inv_grade_bond['inv_grade_bond_return'].rolling(20).std())*(20)**(1/2)
inv_grade_bond['inv_grade_bond_volatility_1y'] = (inv_grade_bond['inv_grade_bond_return'].rolling(252).std())*(252)**(1/2)
inv_grade_bond = inv_grade_bond[['Date','inv_grade_bond','inv_grade_bond_return','inv_grade_bond_volatility_1m',
                                 'inv_grade_bond_volatility_1y']]
inv_grade_bond['one_month_forward_inv_grade_bond_return'] = inv_grade_bond['inv_grade_bond_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 6. Crude Oil WTI
wti = series_dict['wti']
wti['Date'] = pd.to_datetime(wti['DATE'])
wti.rename(columns={'WTISPLC':'wti'}, inplace=True)
wti['wti_return'] = wti['wti'].pct_change()
wti['wti_volatility_1m'] = wti['wti_return'].rolling(20).std()*(20)**(1/2)
wti['wti_volatility_1y'] = wti['wti_return'].rolling(252).std()*(252)**(1/2)
wti = wti[['Date','wti','wti_return','wti_volatility_1m','wti_volatility_1y']]
wti['one_month_forward_wti_return'] = wti['wti_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

uae.tail(10)
# 7. MSCI UAE  
uae = series_dict['uae']
uae['Date'] = pd.to_datetime(uae['Date'])
uae.rename(columns={'Price':'uae'}, inplace=True)
uae['uae_return'] = uae['uae'].pct_change()
uae['uae_volatility_1m'] = (uae['uae_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation
uae['uae_volatility_1y'] = (uae['uae_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year
uae = uae[['Date','uae','uae_return','uae_volatility_1m','uae_volatility_1y']]
# Calculate 1-month forward cumulative returns
uae['one_month_forward_uae_return'] = uae['uae_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]
uae.tail(10)
# Merge datasets together
asset_classes = [btc,gold,high_yield_bond,inv_grade_bond,wti]

baseline = pd.merge(uae,snp,how='left',left_on='Date', right_on="Date")

for asset_class in asset_classes:
    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on="Date")

baseline.info()
baseline.tail(100)
# Index Date
baseline.set_index('Date', inplace=True)
baseline.tail()
# Plot a jointplot with a regression line
sns.jointplot(x = 'gold_return', y = 'uae_return', data = baseline, kind='reg')
# Plot pairplot
baseline_returns = baseline[["uae_return","snp_return", "btc_return", "gold_return", "high_yield_bond_return", "inv_grade_bond_return", "wti_return"]]

sns.pairplot(baseline_returns)
def plot_chart_vol_ret(series):
    fig = px.scatter(baseline[baseline[series+'_return'].notnull()], x=series + '_volatility_1m', 
                     y='one_month_forward_' + series + '_return', width=800,
                     trendline = 'ols')
    fig.update_layout(title=str(series) + ' volatility vs one-month forward return', xaxis_title='', yaxis_title='')
    fig.show()
    
def plot_chart_vol_ret_by_recession(series):
    fig = px.scatter(baseline[baseline[series+'_return'].notnull()], x=series + '_volatility_1m', \
                     color='recession', y='one_month_forward_' + series + '_return', 
                     color_discrete_sequence=['#636EFA', '#FFA15A'], width=800,
                     trendline = 'ols')
    fig.update_layout(title=str(series) + ' volatility vs one-month forward return', xaxis_title='', yaxis_title='')
    fig.show()
plot_chart_vol_ret('uae')