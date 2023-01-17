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
files_G = []



for dirname, _, filenames in os.walk('../input/germany'):

    for filename in filenames:

        files_G.append(os.path.join(dirname, filename))

        

files_G = sorted(files_G)



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
# 2. Germany 10 year bond (MONTHLY)

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



plt.plot(ten_year_bond['Date'], ten_year_bond['10_yr_bond'], color = 'y')

plt.title('Germany 10-year Bond Return Against Time')

plt.show()
#3. Germany long term bond (combined terms) (MONTHLY)

ltbond = series_dict_G['ltbond']



ltbond['Date'] = pd.to_datetime(ltbond['DATE'])

ltbond.rename(columns={'IRLTCT01DEM156N':'ltbond'}, inplace=True)

ltbond['ltbond_return_1m'] = ltbond['ltbond'].pct_change()

ltbond['ltbond_volatility_1m'] = ltbond['ltbond_return_1m'] 

#ltbond['ltbond_volatility_1m'] = (ltbond['ltbond_return'].rolling(20).std())*(20)**(1/2)

ltbond['ltbond_volatility_1y'] = (ltbond['ltbond_return_1m'].rolling(12).std())*(12)**(1/2)

ltbond['one_month_forward_inv_grade_bond_return'] = ltbond['ltbond_return_1m'][::-1].rolling(window=1, min_periods=1).sum()[::-1]



ltbond = ltbond[['Date','ltbond','ltbond_return_1m','ltbond_volatility_1m','ltbond_volatility_1y']]



plt.plot(ltbond['Date'], ltbond['ltbond'], color = 'B')

plt.title('Germany Long-Term Bond Return Against Time')

plt.show()

#4. Inflation in Germany indicated by CPI (MONTHLY)

cpi = series_dict_G['cpi']

cpi['Date'] = pd.to_datetime(cpi['DATE'])

cpi.rename(columns={'DEUCPHPLA01GYM':'cpi'}, inplace=True)

cpi = cpi[['Date','cpi']]

cpi



plt.plot(cpi['Date'], cpi['cpi'], color = 'B')

plt.title('Germany CPI - Inflation Against Time')

plt.show()
#5. Unemployment  (MONTHLY)

#** This is unemployment rate!

unemployment = series_dict_G['unemployment']

unemployment['Date'] = pd.to_datetime(unemployment['DATE'])

unemployment.rename(columns={'LMUNRRTTDEM156S':'unemployment_rate'}, inplace=True)

unemployment = unemployment[['Date','unemployment_rate']]



plt.plot(unemployment['Date'], unemployment['unemployment_rate'], color = 'g')

plt.show()
#6. Germany's Debt Position 

#* The data is not sufficient as it's only until 2017

#This is an indicator of economic conditions, not an asset class



#ger_debt = series_dict_G['debt']

#ger_debt['Date'] = pd.to_datetime(ger_debt['DATE'])

#ger_debt.rename(columns={'GGGDTADEA188N':'debt_percent'}, inplace=True)

#ger_debt = ger_debt[['Date','debt_percent']]



#plt.plot(ger_debt['Date'], ger_debt['debt_percent'])

#plt.show()
# Import datasets with Pandas method read_csv (Daily and Monthly, but only monthly is used)



#import monthly recession indicator (1,0)

recession_indicator_month = series_dict_G['recession_mon']



#Convert data types and rename columns

recession_indicator_month["Date"] = pd.to_datetime(recession_indicator_month["DATE"], format = '%d/%m/%Y')

recession_indicator_month["Recession"] = recession_indicator_month["DEUREC"].astype('bool')



# Subset data columns

recession_indicator_month = recession_indicator_month[["Date","Recession"]]

recession_indicator_month.tail()

# Merge datasets together (All monthly datasets)

asset_classes = [cpi,ten_year_bond,ltbond,unemployment]



baseline = pd.merge(dax_month,recession_indicator_month,how='left',left_on='Date', right_on = 'Date')



for asset_class in asset_classes:

    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on = 'Date')



#Backfilling missing values, for covid outbreak, assume recession YES

baseline.loc[baseline.Date >= '2020-03-01', "Recession"] = 1

baseline["Recession"] = baseline["Recession"].fillna(0).astype(bool)



baseline.info()

baseline.tail(10)
# Grouped by Country, Filter Germany and Resample into Monthly Data

# =======================



full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

full_grouped.info()



# Convert Date from Dtype "Object" (or String) to Dtype "Datetime", filtering for Germany

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

g_covid = full_grouped[full_grouped['Country/Region']=="Germany"]

g_covid_month = g_covid.resample('MS', on = 'Date').mean()

g_covid_month.tail(10)



import warnings

warnings.filterwarnings("ignore")
#2020 covid19 period in Germany with assets

baseline2020 = baseline[baseline['Date'] >= '2019-12-01']

baseline2020 = pd.merge(baseline2020,g_covid_month, how='left', on='Date')

baseline2020['New cases'] = baseline2020['New cases'].fillna(0)



baseline2020.tail(10)
# Create figure with secondary y-axis

fig = make_subplots(specs=[[{"secondary_y": True}]])



# Add traces to create subplots

fig.add_trace(

    go.Scatter(x=baseline2020['Date'], y=baseline2020['dax'], name = 'DAX'),  

    secondary_y=False,

)



fig.add_trace(

    go.Scatter(x=baseline2020['Date'], y=baseline2020['New cases'], name = 'New COVID19 Cases'), 

    secondary_y=True,

)



# Add figure title

fig.update_layout(

    title_text="DAX and New COVID19 Cases in Germany"

)



# Set x-axis title

fig.update_xaxes(title_text="Date")



# Set y-axes titles

fig.update_yaxes(title_text="<b>DAX</b>", secondary_y=False)

fig.update_yaxes(title_text="<b>New COVID19 Cases</b>", secondary_y=True)



fig.show()
def plot_chart(series):

    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color='Recession', width=1000)

    fig.update_traces(mode='markers', marker_size=4)

    fig.update_layout(title=series, xaxis_title="", yaxis_title="")

    fig.show()
baseline['dax_volatility_1m'].describe()
print("The worst single-month return in 2020 is ", str(round(abs(baseline2020['dax_return'].min()/baseline['dax_return'].std()),2)), 

      " X standard deviations of DAX historical returns!")
# Output the range of DAX historical daily returns from 1928-01-03 to 2020-07-01

print("DAX historical monthly returns from " + str(baseline[baseline['dax_return'].notnull()]['Date'].min().date()) + ' to '

       + str(baseline[baseline['dax_return'].notnull()]['Date'].max().date()))



fig = px.histogram(baseline, x="dax_return")

fig.show()
plot_chart("unemployment_rate")
print("This is ", str(round(abs(baseline['unemployment_rate'].min()/baseline['unemployment_rate'].std()),2)), 

      " X standard deviations of the historical monthly change in unemployment!")
sns.jointplot(x = 'New cases', y = 'dax', data = baseline2020, kind='reg')
sns.jointplot(x = 'New deaths', y = 'dax_return', data = baseline2020, kind='reg')
# Draw scatter of asset returns during Covid19 pandemic

baseline2020.info()

baseline_returns = baseline2020[["dax_return", "10_yr_bond_return_1m", "ltbond_return_1m", "New deaths", "New cases"]]



sns.pairplot(baseline_returns)
# Draw heatmap of correlation strength across asset classes (returns and volatilities) and Covid19 new cases and deaths during the pandemic period 

baseline_corr = baseline2020[['dax_return', 'dax_volatility_1y', '10_yr_bond_return_1m', '10_yr_bond_volatility_1y',

                         'ltbond_return_1m', 'ltbond_volatility_1y','New deaths', 'New cases']].corr()



fig, ax = plt.subplots(figsize=(8,5)) 

sns.heatmap(baseline_corr, annot=True, ax = ax)