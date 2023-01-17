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
# Create an empty list for universal and germany specific data

files = []

files_G = []



# Fill the list with the file names of the CSV files in the Kaggle folder

for dirname, _, filenames in os.walk('../input/bootcamp1b'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))

        

for dirname, _, filenames in os.walk('../input/germany'):

    for filename in filenames:

        files_G.append(os.path.join(dirname, filename))



# Sort the file names

files = sorted(files)

files_G = sorted(files_G)



# Output the list of sorted file names

files

files_G
# Read the CSV files through list comprehension

series = [pd.read_csv(f, na_values=['.']) for f in files]



# Define series name, which becomes the dictionary key

series_name = ['btc','cpi_us','gold','snp_us','high_yield_bond_us','inv_grade_bond_us','moderna','employment','tesla_robinhood','trea_20y_bond','trea_10y_yield','tesla','fed_bs','wti']



# series name = dictionary key, series = dictionary value

series_dict = dict(zip(series_name, series))



#Making germany data into a dictionary

series_G = [pd.read_csv(f, na_values=['.']) for f in files_G]

series_name_G = ['10_yr_bond','cpi','DAX','inflation','recession_mon','recession','debt','ltbond','unemployment']

series_dict_G = dict(zip(series_name_G, series_G))
# 1.DAX (Daily and Monthly)

dax = series_dict_G['DAX']



#Convert date into the right format

dax['Date'] = pd.to_datetime(dax['Date'])



#Rename colume from SNP closing data to snp

dax.rename(columns={'Adj Close':'dax'}, inplace=True)



#Calculate % change in stock price

dax['dax_return'] = dax['dax'].pct_change()



# Annualize daily standard deviation * 20 trading days per month (Rolling means to look back X trading days from the recent to the past)

dax['dax_volatility_1m'] = (dax['dax_return'].rolling(20).std())*(20)**(1/2) 



# Annualize daily S.D * 252 trading days per year

dax['dax_volatility_1y'] = (dax['dax_return'].rolling(252).std())*(252)**(1/2) 



#Select he necessary columns

dax = dax[['Date','dax','dax_return','dax_volatility_1m','dax_volatility_1y']]



# Calculate 1-month forward cumulative returns

dax['one_month_forward_dax_return'] = dax['dax_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]



dax_month = dax.resample('MS',on = 'Date').mean()

dax_month.tail()



plt.plot(dax['Date'], dax['dax'], color = 'r')

plt.title('DAX index again Time')

plt.show()



#There's a quick bounce back of stock price during the covid outbreak

#Possibly due to this is not a typical recession but it's an pendamic-induced recession. Cash is not going to physical business so they are flooded into stock markets. 
# 2. Bitcoin (Daily and Monthly)

btc = series_dict['btc']

btc['Date'] = pd.to_datetime(btc['Date'])

btc.rename(columns={'Adj Close':'btc'}, inplace=True)

btc['btc_return'] = btc['btc'].pct_change()

btc['btc_volatility_1m'] = (btc['btc_return'].rolling(20).std())*(20)**(1/2) 

btc['btc_volatility_1y'] = (btc['btc_return'].rolling(252).std())*(252)**(1/2) 

btc = btc[['Date','btc','btc_return','btc_volatility_1m','btc_volatility_1y']]

btc['one_month_forward_btc_return'] = btc['btc_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]



#Compile to monthly

btc_month = btc.resample('MS',on = 'Date').mean()

btc_month.tail()
# 3. Gold (Daily and Monthly)

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



gold_month = gold.resample('MS',on = 'Date').mean()

gold_month.tail()



plt.plot(gold['Date'], gold['gold'],color = 'y')

plt.show()
# 4. Germany 10 year bond (MONTHLY)

ten_year_bond = series_dict_G['10_yr_bond']

ten_year_bond['Date'] = pd.to_datetime(ten_year_bond['DATE'])

ten_year_bond.rename(columns={'IRLTLT01DEM156N':'10_yr_bond'}, inplace=True)

ten_year_bond['10_yr_bond_return_1m'] = ten_year_bond['10_yr_bond'].pct_change()

ten_year_bond['10_yr_bond_volatility_1m'] = ten_year_bond['10_yr_bond_return_1m']

#ten_year_bond['10_yr_bond_volatility_1m'] = (ten_year_bond['10_yr_bond_return'].rolling(20).std())*(20)**(1/2)

ten_year_bond['10_yr_bond_volatility_1y'] = (ten_year_bond['10_yr_bond_return_1m'].rolling(12).std())*(12)**(1/2)

ten_year_bond = ten_year_bond[['Date','10_yr_bond','10_yr_bond_return_1m','10_yr_bond_volatility_1m',

                                   '10_yr_bond_volatility_1y']]

ten_year_bond['one_month_forward_10_yr_bond_return'] = ten_year_bond['10_yr_bond_return_1m'][::-1].rolling(window=1, min_periods=1).sum()[::-1]

ten_year_bond.tail()

#5. Germany long term bond (combined terms) (MONTHLY)

ltbond = series_dict_G['ltbond']



ltbond['Date'] = pd.to_datetime(ltbond['DATE'])

ltbond.rename(columns={'IRLTCT01DEM156N':'ltbond'}, inplace=True)

ltbond['ltbond_return_1m'] = ltbond['ltbond'].pct_change()

ltbond['ltbond_volatility_1m'] = ltbond['ltbond_return_1m'] 

#ltbond['ltbond_volatility_1m'] = (ltbond['ltbond_return'].rolling(20).std())*(20)**(1/2)

ltbond['ltbond_volatility_1y'] = (ltbond['ltbond_return_1m'].rolling(12).std())*(12)**(1/2)

ltbond['one_month_forward_inv_grade_bond_return'] = ltbond['ltbond_return_1m'][::-1].rolling(window=1, min_periods=1).sum()[::-1]



ltbond = ltbond[['Date','ltbond','ltbond_return_1m','ltbond_volatility_1m','ltbond_volatility_1y']]

ltbond
# 6. Crude Oil WTI (Daily and Monthly)

wti = series_dict['wti']

wti['Date'] = pd.to_datetime(wti['DATE'])

wti.rename(columns={'WTISPLC':'wti'}, inplace=True)

wti['wti_return'] = wti['wti'].pct_change()

wti['wti_volatility_1m'] = wti['wti_return'].rolling(20).std()*(20)**(1/2)

wti['wti_volatility_1y'] = wti['wti_return'].rolling(252).std()*(252)**(1/2)

wti = wti[['Date','wti','wti_return','wti_volatility_1m','wti_volatility_1y']]

wti['one_month_forward_wti_return'] = wti['wti_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]



wti_month = wti.resample('MS',on = 'Date').mean()

wti_month.tail()
#7. Inflation in Germany indicated by CPI (MONTHLY)

cpi = series_dict_G['cpi']

cpi['Date'] = pd.to_datetime(cpi['DATE'])

cpi.rename(columns={'DEUCPHPLA01GYM':'cpi'}, inplace=True)

cpi = cpi[['Date','cpi']]

cpi
#8. Employment  (MONTHLY)

#** This is unemployment rate!

unemployment = series_dict_G['unemployment']

unemployment['Date'] = pd.to_datetime(unemployment['DATE'])

unemployment.rename(columns={'LMUNRRTTDEM156S':'unemployment_rate'}, inplace=True)

unemployment = unemployment[['Date','unemployment_rate']]



plt.plot(unemployment['Date'], unemployment['unemployment_rate'])

plt.show()
#9. Germany's Debt Position 

#* The data is not sufficient to the current date

#This is an indicator of economic conditions, not an asset class



ger_debt = series_dict_G['debt']

ger_debt['Date'] = pd.to_datetime(ger_debt['DATE'])

ger_debt.rename(columns={'GGGDTADEA188N':'debt_percent'}, inplace=True)

ger_debt = ger_debt[['Date','debt_percent']]



plt.plot(ger_debt['Date'], ger_debt['debt_percent'])

plt.show()

# Import datasets with Pandas method read_csv (Daily and Monthly, but only monthly is used)



#recession_indicator_month = series_dict_G['recession_mon'] #Monthly data is unnecessary

#recession_indicator_day = series_dict_G['recession']



#import monthly recession indicator (1,0)

recession_indicator_month = series_dict_G['recession_mon']



#Convert data types and rename columns

recession_indicator_month["Date"] = pd.to_datetime(recession_indicator_month["DATE"], format = '%d/%m/%Y')

recession_indicator_month["Recession"] = recession_indicator_month["DEUREC"].astype('bool')

#recession_indicator_day["Date"] = pd.to_datetime(recession_indicator_day["DATE"])

#recession_indicator_day["Recession"] = recession_indicator_day["DEURECD"].astype('bool')



# Subset data columns

recession_indicator_month = recession_indicator_month[["Date","Recession"]]

#recession_indicator_day = recession_indicator_day[["Date","Recession"]]
# Merge datasets together (All monthly datasets)

asset_classes = [btc_month,cpi,gold_month,ten_year_bond,ltbond,unemployment,wti_month]



baseline = pd.merge(dax_month,recession_indicator_month,how='left',left_on='Date', right_on = 'Date')



for asset_class in asset_classes:

    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on = 'Date')



#Backfilling missing values, for covid outbreak, assume recession YES

baseline.loc[baseline.Date >= '2020-03-01', "Recession"] = 1

baseline["Recession"] = baseline["Recession"].fillna(0).astype(bool)



baseline.info()

baseline.tail(10)

# Index Date

baseline.set_index('Date', inplace=True)

baseline.tail(20)
# Re-sample the dataset every year and calculate the sum of returns

baseline_yearly_return = baseline[["dax_return", "btc_return", "gold_return", "ltbond_return_1m", "10_yr_bond_return_1m", "wti_return"]].dropna().resample('Y').sum().reset_index()



print(baseline_yearly_return['Date'].min())

baseline_yearly_return.tail()
# Re-sample the dataset every year and calculate the mean of 1-year volatility

baseline_yearly_volatility_1y = baseline[["dax_volatility_1y", "btc_volatility_1y", "gold_volatility_1y", 

                                          "ltbond_volatility_1y", '10_yr_bond_volatility_1y',

                                          "wti_volatility_1y"]].dropna().resample('Y').mean().reset_index()



baseline_yearly = baseline_yearly_return.merge(baseline_yearly_volatility_1y, on='Date')



baseline_yearly.head()
# Reshape dataset wide to tall with method melt

baseline_yearly_reshaped = baseline_yearly.melt(id_vars='Date', var_name='key', value_name='value')

baseline_yearly_reshaped.head(50)
baseline_yearly_reshaped['metric'] = np.where(baseline_yearly_reshaped['key'].str.contains(pat = 'return'), 'return', 'volatility')

baseline_yearly_reshaped['position']= baseline_yearly_reshaped['key'].str.find('_') 

baseline_yearly_reshaped['asset_class']= baseline_yearly_reshaped['key'].str.slice(0,3,1)

baseline_yearly_reshaped = baseline_yearly_reshaped[['Date','metric','asset_class','value']]

baseline_yearly_reshaped.head()
# Display return and volatility for each asset class

print(baseline_yearly_reshaped[baseline_yearly_reshaped['metric'] == 'return'].groupby('asset_class').mean())

print(baseline_yearly_reshaped[baseline_yearly_reshaped['metric'] == 'volatility'].groupby('asset_class').mean())
# Reset index

baseline.reset_index(inplace=True)

baseline.tail(10)
# Output summary statistics

baseline[["dax_return", "dax_volatility_1y", "btc_return", "btc_volatility_1y", "gold_return", "gold_volatility_1y", 

                  "10_yr_bond_return_1m",'10_yr_bond_volatility_1y', "ltbond_return_1m",'ltbond_volatility_1y',

                 "wti_return", "wti_volatility_1y"]].describe()
# Plot a jointplot with a regression line

sns.jointplot(x = 'gold_return', y = 'dax_return', data = baseline, kind='reg', color = 'r')
def plot_chart(series):

    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color="Recession", color_discrete_sequence=['#636EFA', '#FFA15A'], width=1200)

    fig.update_traces(mode='markers', marker_size=4)

    fig.update_layout(title=series, xaxis_title="", yaxis_title="")

    fig.show()
plot_chart("dax")

dax.tail(20)



#German Stocks in general is not a good hedge against recession
plot_chart("gold")
plot_chart('btc')
# Plot pairplot

baseline_returns = baseline[["dax_return", "btc_return", "gold_return", "ltbond_return_1m", "10_yr_bond_return_1m", "wti_return", "Recession"]]



sns.pairplot(baseline_returns, hue="Recession")
def plot_chart_vol_ret(series):

    fig = px.scatter(baseline[baseline[series+'_return'].notnull()], x=series + '_volatility_1m', 

                     y='one_month_forward_' + series + '_return', width=800,

                     trendline = 'ols')

    fig.update_layout(title=str(series) + ' volatility vs one-month forward return', xaxis_title='', yaxis_title='')

    fig.show()

    

def plot_chart_vol_ret_by_recession(series):

    fig = px.scatter(baseline[baseline[series+'_return'].notnull()], x=series + '_volatility_1m', \

                     color='Recession', y='one_month_forward_' + series + '_return', 

                     color_discrete_sequence=['#636EFA', '#FFA15A'], width=800,

                     trendline = 'ols')

    fig.update_layout(title=str(series) + ' volatility vs one-month forward return', xaxis_title='', yaxis_title='')

    fig.show()
plot_chart_vol_ret('dax')
plot_chart_vol_ret_by_recession('dax')
# Plot heatmap of the relationships across different asset classes

baseline_corr = baseline[['dax_return', 'dax_volatility_1y', 'btc_return', 'btc_volatility_1y',

                         'gold_return', 'gold_volatility_1y', '10_yr_bond_return_1m', '10_yr_bond_volatility_1y',

                         'ltbond_return_1m', 'ltbond_volatility_1y', 'wti_return', 'wti_volatility_1y',

                         'Recession']].dropna().corr()



fig, ax = plt.subplots(figsize=(20,10)) 

sns.heatmap(baseline_corr, annot=True, ax = ax)