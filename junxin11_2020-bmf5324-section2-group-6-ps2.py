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
# color pallette
cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801'

# math operations
from numpy import inf

# time operations
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
# import plotly.figure_factory as ff
#from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# Worldometer data
# ================

worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

# Replace missing values '' with NAN and then 0
# What are the alternatives? Drop or impute. Do they make sense in this context?
worldometer_data = worldometer_data.replace('', np.nan).fillna(0)
worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases']/worldometer_data['TotalTests'],2)
worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths']/worldometer_data['TotalCases'],2)

# Case Positivity is infinity when there is zero TotalTests due to division by zero
worldometer_data[worldometer_data["Case Positivity"] == inf] = 0

# Qcut is quantile cut. Here we specify three equally sized bins and label them low, medium, and high, respectively.
worldometer_data ['Case Positivity Bin']= pd.qcut(worldometer_data['Case Positivity'], q=3, labels=["low", "medium", "high"])

# Population Structure
worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')

# Replace missing values with zeros
worldometer_pop_struc = worldometer_pop_struc.fillna(0)
#worldometer_pop_struc.info()

# Merge worldometer_data with worldometer_pop_struc
# Inner means keep only common key values in both datasets
worldometer_data = worldometer_data.merge(worldometer_pop_struc,how='inner',left_on='Country/Region', right_on='Country')

# Keep observations where column "Country/Region" is not 0
worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]

# Inspect worldometer_data's metadata
worldometer_data.info()

# Inspect Data
# worldometer_data.info()
# worldometer_data.tail(20)
# worldometer_data["Case Positivity"].describe()

worldometer_data["Case Positivity"].describe()
worldometer_data[worldometer_data['Country/Region']=='UAE']['Case Fatality']
worldometer_data[worldometer_data['Country/Region']=='UAE']
# Draw a joint plot to diagnose the relationship between fraction of population aged 65+ and case fatality rate
sns.jointplot(x = 'Fraction age 65+ years', y = "Case Fatality", data = worldometer_data, kind='reg')
plt.scatter(x=worldometer_data[worldometer_data['Country/Region']=='UAE']['Fraction age 65+ years'], y=worldometer_data[worldometer_data['Country/Region']=='UAE']['Case Fatality'], color='r')
worldometer_data[worldometer_data['Country/Region']=='UAE']['Case Positivity Bin']
# Show the descriptive statistics for case positivity bin (categorical variable)
worldometer_data.groupby('Case Positivity Bin')['Case Positivity Bin'].describe()
# Draw a Violin plot to diagnose the relationship betwen case positivity and case fatality rates
fig = go.Figure()

# Create a list of case positivity bin categories
bins = ['low', 'medium', 'high']

# Loop through each case positivity bin
for bin in bins:
    
    # worldometer_data['Case Positivity Bin'][worldometer_data['Case Positivity Bin'] == bin] means take the column 'Case Positivity Bin' and
    # filter the column, such that Case Positivity Bin equals 'low', 'medium', or 'high'
    fig.add_trace(go.Violin(x=worldometer_data['Case Positivity Bin'][worldometer_data['Case Positivity Bin'] == bin],
                            y=worldometer_data['Case Fatality'][worldometer_data['Case Positivity Bin'] == bin],
                            name=bin,
                            box_visible=True,
                            meanline_visible=True))
    
fig.update_layout(title='Case Fatality by Case Positivity Bins', 
                  yaxis_title="Case Fatality", xaxis_title="Case Positivity Bins", 
                  uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()
# Show the summary statistics of column case positivity
worldometer_data["Case Positivity"].describe()

# Filter countries with Case Positivity less than 1% (i.e., 1 confirmed case out of 100 tests)
# These are countries that go for rigorous testing regime
benchmark_countries = worldometer_data[worldometer_data["Case Positivity"]<=0.01]
benchmark_countries.info()
benchmark_countries.head(20)
# Assume that the number of confirmed cases are close to the true infections rates for countries with gold standard testing regimes 
# Thus, their case fatality rates are closer to the true infection fatality rates
infection_fatality_rate = benchmark_countries['TotalDeaths'].sum() / benchmark_countries['TotalCases'].sum()

# Calculate the fraction of total Covid19 deaths for the population aged 65+ among the benchmark countries
benchmark_death_65y_pct = sum(benchmark_countries['TotalDeaths'] * benchmark_countries['Fraction age 65+ years']) / sum(benchmark_countries['TotalDeaths'])

print(infection_fatality_rate)
print(benchmark_death_65y_pct)

print('Estimated Infection Fatality Rate for a benchmark country with %.1f%s of population older than 65 years old \
is %.2f%s' %(100 * benchmark_death_65y_pct,'%',100 * infection_fatality_rate,'%'))
# Estimate Infection Fatality Ratio using the estimated fraction of total Covid19 deaths for the population aged 65+
worldometer_data['Estimated Infection Fatality Ratio'] \
    = ((worldometer_data['TotalDeaths'] * worldometer_data['Fraction age 65+ years']
        /worldometer_data['TotalDeaths']) / benchmark_death_65y_pct) * infection_fatality_rate

# Show descriptive statistics of the columns Estimated Infection Fatality Ratio and Case Fatality
worldometer_data['Estimated Infection Fatality Ratio'].describe()
worldometer_data['Case Fatality'].describe()

# Plot histogram of Estimated Infection Fatality Ratio and Case Fatality
px.histogram(worldometer_data, x='Estimated Infection Fatality Ratio', barmode="overlay")
px.histogram(worldometer_data, x='Case Fatality', barmode="overlay")

# Overlay both histograms for comparison
fig = go.Figure()

fig.add_trace(go.Histogram(x=worldometer_data['Estimated Infection Fatality Ratio'], 
    name = 'Estimated Infection Fatality Rate'
))

fig.add_trace(go.Histogram(x=worldometer_data['Case Fatality'], 
    name = 'Case Fatality Rate'
))

fig.update_layout(barmode='overlay', 
    title = 'Estimated Infection Fatality Rate vs. Case Fatality Rate',
    xaxis_title_text='Value', # xaxis label
    yaxis_title_text='Count', # yaxis label
)
                  
fig.update_traces(opacity=0.75)

fig.show()
worldometer_data[worldometer_data['Country/Region']=='UAE']
worldometer_data_uae=worldometer_data
worldometer_data_uae[worldometer_data_uae['Country/Region']=='UAE']
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
nber_recession_indicator_month = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECM.csv')
nber_recession_indicator_day = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')

nber_recession_indicator_day["Date"] = pd.to_datetime(nber_recession_indicator_day["date"])
nber_recession_indicator_day["value"] = nber_recession_indicator_day["value"].astype('bool')
nber_recession_indicator_day.rename(columns={'value':'recession'}, inplace=True)
nber_recession_indicator_day = nber_recession_indicator_day[["Date","recession"]]
# Grouped by day, country
# =======================

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')
full_grouped.info()
full_grouped.head(10)

# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"
full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])
uae_covid = full_grouped[full_grouped['Country/Region']=="United Arab Emirates"]
uae_covid.info()
uae_covid.tail()

baseline = pd.merge(uae, nber_recession_indicator_day, how='left', on='Date')
baseline = pd.merge(baseline, btc, how='left', on='Date')
baseline = pd.merge(baseline, gold, how='left', on='Date')
baseline = pd.merge(baseline, high_yield_bond, how='left', on='Date')
baseline = pd.merge(baseline, inv_grade_bond, how='left', on='Date')
baseline = pd.merge(baseline, wti, how='left', on='Date')


baseline.loc[baseline.Date >= '2020-03-01', "recession"] = 1
baseline["recession"] = baseline["recession"].fillna(0)
#baseline["recession"] = baseline["recession"].astype(int)

baseline.info()

#2020 covid19 period
baseline2020 = baseline[baseline['Date'] >= '2020-01-01']
baseline2020 = pd.merge(baseline2020,uae_covid, how='left', on='Date')
baseline2020['New cases'] = baseline2020['New cases'].fillna(0)
uae_covid.tail()
def plot_chart(series):
    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color='recession', width=1000)
    fig.update_traces(mode='markers', marker_size=4)
    fig.update_layout(title=series, xaxis_title="", yaxis_title="")
    fig.show()
baseline2020['uae_return'].describe()
baseline['uae_return'].describe()
print("The worst single-day return in 2020 is ", str(round(abs(baseline2020['uae_return'].min()/baseline['uae_return'].std()),2)), 
      " X standard deviations of ADX General historical returns!")
# Output the range of AGX historical daily returns from 1928-01-03 to 2020-07-01
print("ADX General historical daily returns from " + str(baseline[baseline['uae_return'].notnull()]['Date'].min().date()) + ' to '
       + str(baseline[baseline['uae_return'].notnull()]['Date'].max().date()))

fig = px.histogram(baseline, x="uae_return")
fig.show()
sns.jointplot(x = 'New cases', y = 'uae_return', data = baseline2020, kind='reg')
sns.jointplot(x = 'New deaths', y = 'uae_return', data = baseline2020, kind='reg')
# Draw scatter of asset returns during Covid19 pandemic
baseline_returns = baseline2020[["uae_return", "btc_return", "gold_return", "high_yield_bond_return", "inv_grade_bond_return", 
                  "wti_return", "New deaths", "New cases"]]

sns.pairplot(baseline_returns)
# Draw heatmap of correlation strength across asset classes (returns and volatilities) and Covid19 new cases and deaths during the pandemic period 
baseline_corr = baseline2020[['uae_return', 'uae_volatility_1y', 'btc_return', 'btc_volatility_1y',
                         'gold_return', 'gold_volatility_1y', 'high_yield_bond_return', 'high_yield_bond_volatility_1y',
                         'inv_grade_bond_return', 'inv_grade_bond_volatility_1y', 'wti_return', 'wti_volatility_1y',
                         'New deaths', 'New cases']].corr()

fig, ax = plt.subplots(figsize=(16,5)) 
sns.heatmap(baseline_corr, annot=True, ax = ax)
