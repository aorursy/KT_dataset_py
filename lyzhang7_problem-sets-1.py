# for numerical analyiss

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

for dirname, _, filenames in os.walk('../input/xyz123'):

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

series_name = ['btc','cpi','gold','cac','high_yield_bond','inv_grade_bond','moderna','employment','tesla_robinhood','trea_20y_bond','trea_10y_yield','tesla','fed_bs','wti']



# series name = dictionary key, series = dictionary value

series_dict = dict(zip(series_name, series))
# 1. CAC40 

cac = series_dict['cac']

cac['Date'] = pd.to_datetime(cac['Date'])

cac.rename(columns={'Adj Close':'cac'}, inplace=True)

cac['cac_return'] = cac['cac'].pct_change()

cac['cac_volatility_1m'] = (cac['cac_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation

cac['cac_volatility_1y'] = (cac['cac_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year

cac = cac[['Date','cac','cac_return','cac_volatility_1m','cac_volatility_1y']]

# Calculate 1-month forward cumulative returns

cac['one_month_forward_cac_return'] = cac['cac_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]
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
#7. Inflation

cpi = series_dict['cpi']

cpi['Date'] = pd.to_datetime(cpi['DATE'])

cpi.rename(columns={'CUUR0000SEHE':'cpi'}, inplace=True)

cpi = cpi[['Date','cpi']]
#8. Employment

employment = series_dict['employment']

employment['Date'] = pd.to_datetime(employment['DATE'])

employment.rename(columns={'PAYEMS_CHG':'employment'}, inplace=True)

employment = employment[['Date','employment']]
#9. US Fed's Balance Sheet

fed_bs = series_dict['fed_bs']

fed_bs['Date'] = pd.to_datetime(fed_bs['DATE'])

fed_bs.rename(columns={'WALCL':'fed_bs'}, inplace=True)

fed_bs = fed_bs[['Date','fed_bs']]
cac.tail(10)
# Import datasets with Pandas method read_csv

nber_recession_indicator_month = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECM.csv')

nber_recession_indicator_day = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')



# Convert data types

nber_recession_indicator_day["Date"] = pd.to_datetime(nber_recession_indicator_day["date"])

nber_recession_indicator_day["recession"] = nber_recession_indicator_day["value"].astype('bool')



# Subset data columns

nber_recession_indicator_day = nber_recession_indicator_day[["Date","recession"]]
# Merge datasets together

asset_classes = [btc,cpi,gold,high_yield_bond,inv_grade_bond,employment,fed_bs,wti]



baseline = pd.merge(cac,nber_recession_indicator_day,how='left',left_on='Date', right_on="Date")



for asset_class in asset_classes:

    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on="Date")



# Backfilling missing values,  

baseline.loc[baseline.Date >= '2020-03-01', "recession"] = 1

baseline["recession"] = baseline["recession"].fillna(0).astype(bool)



baseline.info()
# Index Date

baseline.set_index('Date', inplace=True)

baseline.tail()
baseline.tail()
# Re-sample the dataset every year and calculate the sum of returns

baseline_yearly_return = baseline[["cac_return", "btc_return", "gold_return", "high_yield_bond_return",  

                            "inv_grade_bond_return", "wti_return"]].dropna().resample('Y').sum().reset_index()



print(baseline_yearly_return['Date'].min()) # 2010-12-31

baseline_yearly_return.head()
# Re-sample the dataset every year and calculate the mean of 1-year volatility

baseline_yearly_volatility_1y = baseline[["cac_volatility_1y", "btc_volatility_1y", "gold_volatility_1y", 

                                          "high_yield_bond_volatility_1y", "inv_grade_bond_volatility_1y", 

                                          "wti_volatility_1y"]].dropna().resample('Y').mean().reset_index()



baseline_yearly = baseline_yearly_return.merge(baseline_yearly_volatility_1y, left_on='Date', right_on='Date')



baseline_yearly.head()
# Reshape dataset wide to tall with method melt

baseline_yearly_reshaped = baseline_yearly.melt(id_vars='Date', var_name='key', value_name='value')

baseline_yearly_reshaped.head()
baseline_yearly_reshaped['metric'] = np.where(baseline_yearly_reshaped['key'].str.contains(pat = 'return'), 'return', 'volatility')

baseline_yearly_reshaped['position']= baseline_yearly_reshaped['key'].str.find('_') 

baseline_yearly_reshaped['asset_class']= baseline_yearly_reshaped['key'].str.slice(0,3,1)

baseline_yearly_reshaped = baseline_yearly_reshaped[['Date','metric','asset_class','value']]

baseline_yearly_reshaped.head()
# Display return and volatility for each asset class

print(baseline_yearly_reshaped[baseline_yearly_reshaped['metric'] == 'return'].groupby('asset_class').mean())

print(baseline_yearly_reshaped[baseline_yearly_reshaped['metric'] == 'volatility'].groupby('asset_class').mean())
baseline.tail()
# Reset index

baseline.reset_index(inplace=True)

baseline.tail()
# Output summary statistics

baseline[["cac_return", "cac_volatility_1y", "btc_return", "btc_volatility_1y", "gold_return", "gold_volatility_1y", 

                  "high_yield_bond_return", "high_yield_bond_volatility_1y", "inv_grade_bond_return", 

                  "inv_grade_bond_volatility_1y", "wti_return", "wti_volatility_1y"]].describe()
# Plot a jointplot with a regression line

sns.jointplot(x = 'gold_return', y = 'cac_return', data = baseline, kind='reg')
def plot_chart(series):

    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color="recession", color_discrete_sequence=['#636EFA', '#FFA15A'], width=1200)

    fig.update_traces(mode='markers', marker_size=4)

    fig.update_layout(title=series, xaxis_title="", yaxis_title="")

    fig.show()
plot_chart("cac")
plot_chart("gold")
plot_chart('btc')
# Plot pairplot

baseline_returns = baseline[["cac_return", "btc_return", "gold_return", "high_yield_bond_return", "inv_grade_bond_return", "wti_return", "recession"]]



sns.pairplot(baseline_returns, hue="recession")
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
plot_chart_vol_ret('cac')
plot_chart_vol_ret_by_recession('cac')
# Plot heatmap of the relationships across different asset classes

baseline_corr = baseline[['cac_return', 'cac_volatility_1y', 'btc_return', 'btc_volatility_1y',

                         'gold_return', 'gold_volatility_1y', 'high_yield_bond_return', 'high_yield_bond_volatility_1y',

                         'inv_grade_bond_return', 'inv_grade_bond_volatility_1y', 'wti_return', 'wti_volatility_1y',

                         'recession']].dropna().corr()



fig, ax = plt.subplots(figsize=(20,10)) 

sns.heatmap(baseline_corr, annot=True, ax = ax)