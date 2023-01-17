# for numerical analysis
import numpy as np

# to store and process data in dataframe
import pandas as pd

# to interface with operating system
import os

# for basic visualization
import matplotlib.pyplot as plt

# for advanced visualization
import seaborn as sns; sns.set()

# for interactive visualization
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# for offline interactive visualization
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# for trendlines
import statsmodels

# data manipulation
from datetime import datetime as dt
from scipy.stats.mstats import winsorize
files = []

for dirname, _, filenames in os.walk('../input/econfin'):
    for filename in filenames:
        files.append(os.path.join(dirname, filename))
        
files = sorted(files)
files
series = [pd.read_csv(f, na_values=['.']) for f in files]
series_name = ['btc', 'cpi', 'gold', 'snp', 'high_yield_bond', 'inv_grade_bond', 'moderna', 'employment', 'tesla_robinhood', 
               'trea_20y_bond', 'trea_10y_yield', 'tesla_stock', 'fed_bs', 'wti']
series_dict = dict(zip(series_name, series))
# Grouped by day, country
# =======================

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped.info()
full_grouped.head(10)

# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
us_covid = full_grouped[full_grouped['Country/Region']=="US"]
us_covid.info()
us_covid.tail()
import warnings
warnings.filterwarnings("ignore")
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

# 7. Inflation
cpi = series_dict['cpi']
cpi['Date'] = pd.to_datetime(cpi['DATE'])
cpi.rename(columns={'CUUR0000SEHE':'cpi'}, inplace=True)
cpi = cpi[['Date','cpi']]

# 8. Employment
employment = series_dict['employment']
employment['Date'] = pd.to_datetime(employment['DATE'])
employment.rename(columns={'PAYEMS_CHG':'employment'}, inplace=True)
employment = employment[['Date','employment']]

# 9. US Fed's Balance Sheet
fed_bs = series_dict['fed_bs']
fed_bs['Date'] = pd.to_datetime(fed_bs['DATE'])
fed_bs.rename(columns={'WALCL':'fed_bs'}, inplace=True)
fed_bs = fed_bs[['Date','fed_bs']]

# 10. Moderna
moderna = series_dict['moderna']
moderna['Date'] = pd.to_datetime(moderna['Date'])
moderna.rename(columns={'Adj Close':'moderna'}, inplace=True)
moderna['moderna_return'] = moderna['moderna'].pct_change()
moderna['moderna_volatility_1m'] = (moderna['moderna_return'].rolling(20).std())*(20)**(1/2)
moderna['moderna_volatility_1y'] = (moderna['moderna_return'].rolling(252).std())*(252)**(1/2)
moderna = moderna[['Date','moderna','moderna_return','moderna_volatility_1m', 'moderna_volatility_1y']]
moderna['one_month_forward_moderna_return'] = moderna['moderna_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

# 11. Tesla
tesla = series_dict['tesla_stock']
tesla['Date'] = pd.to_datetime(tesla['Date'])
tesla.rename(columns={'Adj Close':'tesla'}, inplace=True)
tesla['tesla_return'] = tesla['tesla'].pct_change()
tesla['tesla_volatility_1m'] = (tesla['tesla_return'].rolling(20).std())*(20)**(1/2)
tesla['tesla_volatility_1y'] = (tesla['tesla_return'].rolling(252).std())*(252)**(1/2)
tesla = tesla[['Date','tesla','tesla_return','tesla_volatility_1m', 'tesla_volatility_1y']]
tesla['one_month_forward_tesla_return'] = tesla['tesla_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]

robinhood = series_dict['tesla_robinhood']
robinhood['Date'] = pd.to_datetime(robinhood['timestamp'], format='%Y-%m-%d %H:%M:%S')
robinhood.set_index('Date', inplace=True)
robinhood_daily = robinhood[['users_holding']].dropna().resample('D').mean().reset_index()
robinhood_daily['users_holding_pct_change'] = robinhood_daily["users_holding"].pct_change()

tesla_robinhood = pd.merge(robinhood_daily,tesla,how='left',on='Date')

nber_recession_indicator_month = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECM.csv')
nber_recession_indicator_day = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')

nber_recession_indicator_day["Date"] = pd.to_datetime(nber_recession_indicator_day["date"])
nber_recession_indicator_day["value"] = nber_recession_indicator_day["value"].astype('bool')
nber_recession_indicator_day.rename(columns={'value':'recession'}, inplace=True)
nber_recession_indicator_day = nber_recession_indicator_day[["Date","recession"]]
baseline = pd.merge(snp, nber_recession_indicator_day, how='left', on='Date')
baseline = pd.merge(baseline, btc, how='left', on='Date')
baseline = pd.merge(baseline, cpi, how='left', on='Date')
baseline = pd.merge(baseline, gold, how='left', on='Date')
baseline = pd.merge(baseline, high_yield_bond, how='left', on='Date')
baseline = pd.merge(baseline, inv_grade_bond, how='left', on='Date')
baseline = pd.merge(baseline, wti, how='left', on='Date')
baseline = pd.merge(baseline, employment, how='left', on='Date')
baseline = pd.merge(baseline, fed_bs, how='left', on='Date')

baseline.loc[baseline.Date >= '2020-03-01', "recession"] = 1
baseline["recession"] = baseline["recession"].fillna(0)
#baseline["recession"] = baseline["recession"].astype(int)

baseline.info()

#2020 covid19 period
baseline2020 = baseline[baseline['Date'] >= '2020-01-01']
baseline2020 = pd.merge(baseline2020,us_covid, how='left', on='Date')
baseline2020['New cases'] = baseline2020['New cases'].fillna(0)
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces to create subplots
fig.add_trace(
    go.Scatter(x=baseline2020['Date'], y=baseline2020['snp'], name = 'S&P500'),  
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=baseline2020['Date'], y=baseline2020['New cases'], name = 'New COVID19 Cases'), 
    secondary_y=True,
)

# Add figure title
fig.update_layout(
    title_text="S&P500 and New COVID19 Cases"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>S&P500</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>New COVID19 Cases</b>", secondary_y=True)

fig.show()
def plot_chart(series):
    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color='recession', width=1000)
    fig.update_traces(mode='markers', marker_size=4)
    fig.update_layout(title=series, xaxis_title="", yaxis_title="")
    fig.show()
baseline2020['snp_return'].describe()
baseline['snp_return'].describe()
print("The worst single-day return in 2020 is ", str(round(abs(baseline2020['snp_return'].min()/baseline['snp_return'].std()),2)), 
      " X standard deviations of S&P500 historical returns!")
# Output the range of S&P500 historical daily returns from 1928-01-03 to 2020-07-01
print("S&P500 historical daily returns from " + str(baseline[baseline['snp_return'].notnull()]['Date'].min().date()) + ' to '
       + str(baseline[baseline['snp_return'].notnull()]['Date'].max().date()))

fig = px.histogram(baseline, x="snp_return")
fig.show()
plot_chart("employment")
print("This is ", str(round(abs(baseline['employment'].min()/baseline['employment'].std()),2)), 
      " X standard deviations of the historical monthly change in employment!")
sns.jointplot(x = 'New cases', y = 'snp_return', data = baseline2020, kind='reg')
sns.jointplot(x = 'New deaths', y = 'snp_return', data = baseline2020, kind='reg')
# Draw scatter of asset returns during Covid19 pandemic
baseline_returns = baseline2020[["snp_return", "btc_return", "gold_return", "high_yield_bond_return", "inv_grade_bond_return", 
                  "wti_return", "New deaths", "New cases"]]

sns.pairplot(baseline_returns)
# Draw heatmap of correlation strength across asset classes (returns and volatilities) and Covid19 new cases and deaths during the pandemic period 
baseline_corr = baseline2020[['snp_return', 'snp_volatility_1y', 'btc_return', 'btc_volatility_1y',
                         'gold_return', 'gold_volatility_1y', 'high_yield_bond_return', 'high_yield_bond_volatility_1y',
                         'inv_grade_bond_return', 'inv_grade_bond_volatility_1y', 'wti_return', 'wti_volatility_1y',
                         'New deaths', 'New cases']].corr()

fig, ax = plt.subplots(figsize=(16,5)) 
sns.heatmap(baseline_corr, annot=True, ax = ax)
# Let's see how Federal Reserves's balance sheet has changed over time?
plot_chart('fed_bs')
# Identify key milestone dates in vaccine developments by Moderna
dates = pd.to_datetime(['2020-7-27', '2020-5-6', '2020-5-1', '2020-4-27', '2020-4-16', '2020-3-16', '2020-1-13'])


moderna['vaccine_milestone_announced']  = moderna['Date'].isin(dates)
baseline2020['vaccine_milestone_announced'] = baseline2020['Date'].isin(dates)
# Let's inspect the moderna dataset
moderna
# Let's create a function to plot graphs with vaccine milestones highlighted.
def plot_return_vaccine_milestone(data, asset):
    fig = px.scatter(data, x='Date', y=asset, color='vaccine_milestone_announced', width=1000)
    fig.update_traces(mode='markers', marker_size=4)
    fig.update_layout(title=str(asset), xaxis_title='Date', yaxis_title=str(asset))
    fig.show()
# Draw a scatterplot of Moderna's historical stock returns
plot_return_vaccine_milestone(moderna, 'moderna_return')
# Draw a scatterplot of S&P500's historical stock returns
plot_return_vaccine_milestone(baseline2020, 'snp_return')
# Filter Tesla return series and robinhood users who invest in Tesla for the year 2020
tesla_robinhood2020 = tesla_robinhood[tesla_robinhood['Date'] >= '2020-01-01']
tesla_robinhood2020.info()
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

# Add traces
fig.add_trace(
    go.Scatter(x=tesla_robinhood2020['Date'], y=tesla_robinhood2020['tesla'], name = 'Tesla Price'),  
    secondary_y=False,
)

fig.add_trace(
    go.Scatter(x=tesla_robinhood2020['Date'], y=tesla_robinhood2020['users_holding'], name = 'Robinhood Users\' Holdings'), 
    secondary_y=True,
)


# Add figure title
fig.update_layout(
    title_text="Tesla Price and Robinhood Users\' Holdings"
)

# Set x-axis title
fig.update_xaxes(title_text="Date")

# Set y-axes titles
fig.update_yaxes(title_text="<b>Tesla Price</b>", secondary_y=False)
fig.update_yaxes(title_text="<b>Robinhood Users\' Holdings</b>", secondary_y=True)

fig.show()
#Turn infinity values (due to division by zero) into NaN and then drop all the NaN for the column 'users_holding_pct_change'
tesla_robinhood2020 = tesla_robinhood2020.replace([np.inf, -np.inf], np.nan).dropna(subset=['users_holding_pct_change'], how="all")
#tesla_robinhood2020.describe()

# Draw jointplot for testa's return and users_holding_pct_change
sns.jointplot(x = 'tesla_return', y = 'users_holding_pct_change', data = tesla_robinhood2020, kind='reg')