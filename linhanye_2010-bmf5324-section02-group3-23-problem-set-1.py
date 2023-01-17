# datetime operations

from datetime import timedelta



# for numerical analyiss

import numpy as np



# to store and process data in dataframe

import pandas as pd



# basic visualization package

import matplotlib.pyplot as plt



# advanced ploting

import seaborn as sns; sns.set()



# interactive visualization

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



# for offline ploting, must turn this plotly.offline on

from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# to interface with operating system

import os



# for trendlines

import statsmodels



# ------------------------------------ABOVE THIS LINE ARE PROBLEM SET ONE USAGE----------------------------------------#



# color pallette

# Hexademical code RRGGBB (True Black #000000, True White #ffffff)

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# ------------------------------------ABOVE THIS LINE ARE PROBLEM SET ONE USAGE----------------------------------------#
# 1ST FILE IMPUT (for COVID 19 Situation)

# list files, this has nothing to do with later codes but just display you the file inside the data

# In total 6 files: country_wise_latest.csv, day_wise.csv, usa_country_wise, covid19_clean_complete, full_group and woldometer

# First 5 files from John Hopkins Uni, and becasue we do SG only, we use file except usa_country_wise and woldometer file

!ls ../input/corona-virus-report



##############################################################################################################################################################



# SECOND FILE INPUT: Ecofin file (for macroeconomy test)

# Create an empty list

files = []



# Fill the list with the file names of the CSV files in the Kaggle folder

for dirname, _, filenames in os.walk('../input/ecofinpbs1'):

    for filename in filenames:

        files.append(os.path.join(dirname, filename))



# Sort the file names

files = sorted(files)



# Read the CSV files through list comprehension, which can be broken into three parts

# 1. OUTPUT EXPRESSION [pd.read_csv(f, na_values=['.'])] --- Note: this turns character '.' values into missing value

# 2. INPUT SEQUENCE [for f] 

# 3. CONDITION (OPTIONAL) [in files] 

series = [pd.read_csv(f, na_values=['.']) for f in files]



# Define series name, which becomes the dictionary key

series_name = ['btc','cpi','gold','snp','high_yield_bond','inv_grade_bond','moderna',\

               'employment','tesla_robinhood','trea_20y_bond','trea_10y_yield','tesla','fed_bs','wti']



# series name = dictionary key, series = dictionary value

series_dict = dict(zip(series_name, series))





##############################################################################################################################################################



# THIRD FILE INPUT: Ecofin file (for us recession test)

# Read the dataset CSV to an object

nber_recession_indicator_month = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECM.csv')

nber_recession_indicator_day = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')





##############################################################################################################################################################



# FORTH FILE INPUT: STIINDEX (for SG stock return)

# Read the dataset CSV to an object

STI = pd.read_csv('../input/stiidx/STI.csv')



##############################################################################################################################################################



# FIFTH FILE INPUT: STIINDEX (for SG stock return)

# Read the dataset CSV to an object

tbill = pd.read_csv('../input/tbill10yr/TBill10Yr.csv')





# ------------------------------------ABOVE THIS LINE ARE PROBLEM SET ONE USAGE----------------------------------------#





# PROBLEMSET1: for corona virus report



# 1st file: Country wise

# This CSV provides you with general look over the country cases (death etc),not daily, for pie and region general chart

country_wise = pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')



# Replace missing values '' with NAN and then 0

country_wise = country_wise.replace('', np.nan).fillna(0)

country_wise_sg_only = country_wise.loc[country_wise["Country/Region"].isin(['Singapore'])]





# Already check only one SG is there, to check simply just un# the following code

# list_of_country_name = []

# for i in range (0, 187):

#     list_of_country_name.append(country_wise.iloc[i, 0])

# print(list_of_country_name)



# if wanted to see the datastructure of this country wise (control + /, can do all # or un#)

# country_wise.info()





##########################################################################################################################

# 2nd file: full_group

# This file contains all information as accumulated, each country, everyday what is the case number and regions

# Grouped by day, country

full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')



# if want to see datastructure of the full group file (control + /, can do all # or un#)

# full_grouped.info()

# full_grouped.head(10)



# Checked, SG, Singpore etc wrong spelling does not appear in this file

# The way to check it is bascially the same as above, quary a list and find if wrong name such as SG, Singpore etc exist

# And double check in S start region (slice the list and check if no SG missing)



# Convert Date from Dtype "Object" (or String) to Dtype "Datetime"

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

# full_grouped.info(): to check the info of the changed dataset



# After check it, breakdown to SG only, create another dataframe called "full_grouped_SG_Only":

full_grouped_SG_Only = full_grouped.loc[full_grouped["Country/Region"].isin(['Singapore'])]

# full_grouped_SG_Only.head(10): to check the info of the changed dataset



##########################################################################################################################

# 3rd file: day_wise

# This file only have information on day wise case growth not the country specific

day_wise = pd.read_csv('../input/corona-virus-report/day_wise.csv')

day_wise['Date'] = pd.to_datetime(day_wise['Date'])



# if want to see datastructure of the day_wise file (control + /, can do all # or un#)

# day_wise.info()

# day_wise.head(10)



##########################################################################################################################

# 4th file: worldometer

# This file only have information on day wise case growth not the country specific

worldometer = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')



# Add some more statistics to worldmeter file

worldometer['InfectionRate'] = worldometer['TotalCases']/worldometer['Population']

worldometer['DeathRate'] = worldometer['TotalDeaths']/worldometer['TotalCases']

worldometer['SeriousRate'] = worldometer['Serious,Critical']/worldometer['TotalCases']

worldometer['TestRate'] = worldometer['TotalTests']/worldometer['Population']



# if want to see datastructure of the day_wise file (control + /, can do all # or un#)

# worldometer.info()

# worldometer.head(10)
# PROBLEMSET1: for ecofinpbs1



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

high_yield_bond = high_yield_bond[['Date','high_yield_bond','high_yield_bond_return','high_yield_bond_volatility_1m','high_yield_bond_volatility_1y']]

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



#10. STI Index

STI['Date'] = pd.to_datetime(STI['Date'])

STI.rename(columns={' Close':'sti'}, inplace=True) # remember there is a space before "Close"

STI['sti_return'] = STI['sti'].pct_change()

STI['sti_volatility_1m'] = (STI['sti_return'].rolling(20).std())*(20)**(1/2) # Annualize daily standard deviation

STI['sti_volatility_1y'] = (STI['sti_return'].rolling(252).std())*(252)**(1/2) # 252 trading days per year

STI = STI[['Date','sti','sti_return','sti_volatility_1m','sti_volatility_1y']]

# Calculate 1-month forward cumulative returns

STI['one_month_forward_sti_return'] = STI['sti_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]



#11. Tbill

tbill["help_drop_na"] = tbill["Close"]/2 # create nup.nan for future dropping and cleanning

tbill = tbill.dropna()

tbill['discount_rate'] = tbill ['Adj Close']/100 + 1

tbill['1plus_discount_rate'] = tbill ['discount_rate']**5

tbill['tbill_close_price'] = 100/tbill['1plus_discount_rate']

# try to change date

tbill['Date'] = pd.to_datetime(tbill['Date'])

tbill.rename(columns={'tbill_close_price':'tbill'}, inplace=True)

# do operation

tbill['tbill_return'] = tbill['tbill'].pct_change()

tbill['tbill_volatility_1m'] = (tbill['tbill_return'].rolling(20).std())*(20)**(1/2)

tbill['tbill_volatility_1y'] = (tbill['tbill_return'].rolling(252).std())*(252)**(1/2)

tbill = tbill[['Date','tbill','tbill_return','tbill_volatility_1m','tbill_volatility_1y']]

tbill['one_month_forward_tbill_return'] = tbill['tbill_return'][::-1].rolling(window=20, min_periods=1).sum()[::-1]
# PROBLEMSET1: for nber base recession indicator



# Import datasets with Pandas method read_csv

nber_recession_indicator_month = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECM.csv')

nber_recession_indicator_day = pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')



# Convert data types

nber_recession_indicator_day["Date"] = pd.to_datetime(nber_recession_indicator_day["date"])

nber_recession_indicator_day["recession"] = nber_recession_indicator_day["value"].astype('bool')



# Subset data columns

nber_recession_indicator_day = nber_recession_indicator_day[["Date","recession"]]



# PROBLEMSET1: merge 2nd and 3rd file together

# Merge datasets together

asset_classes = [btc,cpi,gold,high_yield_bond,inv_grade_bond,employment,fed_bs,wti,tbill]



baseline = pd.merge(snp,nber_recession_indicator_day,how='left',left_on='Date', right_on="Date")

baseline = pd.merge(STI,baseline,how='left',left_on='Date', right_on="Date")



for asset_class in asset_classes:

    baseline = pd.merge(baseline,asset_class,how='left',left_on='Date', right_on="Date")



# Backfilling missing values,  

baseline.loc[baseline.Date >= '2020-03-01', "recession"] = 1

baseline["recession"] = baseline["recession"].fillna(0).astype(bool)



baseline.info()
# Generate the global trend of active, recover and death chart

# Collapse Country, Date observations to Date observations and reindex

active_total_trend = full_grouped.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()



# Melt the data by the value_vars, bascially keep the date and make status as one column, cases become another column

active_total_trend = active_total_trend.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')



# Plot the general chart in the ways that as time goes by, what is the case situation

fig = px.area(active_total_trend, x="Date", y="Count", color='Case', height=600, width=700,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Try to plot the stacked chart over time by confirm case, and try to observe the global trend in another way



def plot_stacked(col):

    fig = px.bar(full_grouped, x="Date", y=col, color='Country/Region', 

                 height=600, title=col, color_discrete_sequence = px.colors.cyclical.mygbm)

    fig.update_layout(showlegend=True)

    fig.show()



plot_stacked('Confirmed')
# Try to look at top ten cases country now



def plot_hbar(df, col, n, hover_data=[]):

    fig = px.bar(df.sort_values(col).tail(n), 

                 x=col, y="Country/Region", color='WHO Region',  

                 text=col, orientation='h', width=700, hover_data=hover_data,

                 color_discrete_sequence = px.colors.qualitative.Dark2)

    fig.update_layout(title=col, xaxis_title="", yaxis_title="", 

                      yaxis_categoryorder = 'total ascending',

                      uniformtext_minsize=8, uniformtext_mode='hide')

    fig.show()



plot_hbar(country_wise, 'Confirmed', 10)
# Try to create a % wise country case chart, merge helps the percentage calculation

country_wise_perc = pd.merge(full_grouped[['Date', 'Country/Region', 'Confirmed', 'Deaths']], 

                day_wise[['Date', 'Confirmed', 'Deaths']], on='Date')

country_wise_perc['% Confirmed'] = round(country_wise_perc['Confirmed_x']/country_wise_perc['Confirmed_y'], 3)*100

country_wise_perc['% Deaths'] = round(country_wise_perc['Deaths_x']/country_wise_perc['Deaths_y'], 3)*100



def plot_bar(country_wise_perc, col): # change here as there is no feed in df, writing temp very confusing

    fig = px.bar(country_wise_perc, x='Date', y=col, color='Country/Region', 

             range_y=(0, 100), title='% of Confirmed Cases from each country', 

             color_discrete_sequence=px.colors.qualitative.Prism)

    fig.show()

    

plot_bar(country_wise_perc, '% Confirmed')
# Generate Current Screenshot of cases expansion in Singapore

# Graph out the chart

country_wise_sg_only = country_wise_sg_only.melt(value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(country_wise_sg_only, path=["variable"], values="value", height=225)

fig.data[0].textinfo = 'label+text+value'

fig.show()
# Generate the global trend of active, recover and death chart for Singapore

# Collapse Country, Date observations to Date observations and reindex

# Similar to world chart, can wtire a function to do this and just call, but notebook not scrip so make it clear, write it out

active_singapore_trend = full_grouped_SG_Only.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()



# Melt the data by the value_vars, bascially keep the date and make status as one column, cases become another column

active_singapore_trend = active_singapore_trend.melt(id_vars="Date", value_vars=['Deaths', 'Active', 'Recovered'],

                 var_name='Case', value_name='Count')



# Plot the general chart in the ways that as time goes by, what is the case situation

fig = px.area(active_singapore_trend, x="Date", y="Count", color='Case', height=600, width=700,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# Use world meter to see what are the test rate zone Singapore located

sg_testrate = worldometer[['Country/Region','WHO Region','TotalCases','TestRate']].dropna().sort_values('TestRate',ascending=False)



sg_testrate.reset_index(inplace=True)

sg_testrate.drop(['index'], axis=1,inplace=True)

print(sg_testrate.loc[sg_testrate['Country/Region'] == 'Singapore'])



fig = px.scatter(sg_testrate,x='Country/Region', y='TestRate',size='TotalCases',color='WHO Region',color_discrete_sequence = px.colors.qualitative.Dark2)

fig.update_layout(title='Test Rate', xaxis_title="", yaxis_title="TestRate",xaxis_categoryorder = 'total ascending',

                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)



fig.show()
# Use world meter to see what are the infection rate zone Singapore located



sg_infection_rate = worldometer[['Country/Region','WHO Region','TotalCases','InfectionRate']].dropna().sort_values('InfectionRate',ascending=False)



sg_infection_rate.reset_index(inplace=True)

sg_infection_rate.drop(['index'], axis=1,inplace=True)

print(sg_infection_rate.loc[sg_infection_rate['Country/Region'] == 'Singapore'])



fig = px.scatter(sg_infection_rate,x='Country/Region', y='InfectionRate',size='TotalCases',color='WHO Region',color_discrete_sequence = px.colors.qualitative.Dark2)

fig.update_layout(title='Infection Rate', xaxis_title="", yaxis_title="InfectionRate",xaxis_categoryorder = 'total ascending',

                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)

fig.show()



# Use worldometer to see the serious case rate for Singapore

sg_seriousrate = worldometer[['Country/Region','WHO Region','TotalCases','SeriousRate']].dropna().sort_values('SeriousRate',ascending=False)



sg_seriousrate.reset_index(inplace=True)

sg_seriousrate.drop(['index'], axis=1,inplace=True)

print(sg_seriousrate.loc[sg_seriousrate['Country/Region'] == 'Singapore'])

# Use world meter to see what are the death rate zone Singapore located



sg_death_rate = worldometer[['Country/Region','WHO Region','TotalCases','DeathRate']].dropna().sort_values('DeathRate',ascending=False)



sg_death_rate.reset_index(inplace=True)

sg_death_rate.drop(['index'], axis=1,inplace=True)

print(sg_death_rate.loc[sg_death_rate['Country/Region'] == 'Singapore'])



fig = px.scatter(sg_death_rate,x='Country/Region', y='DeathRate',size='TotalCases',color='WHO Region',color_discrete_sequence = px.colors.qualitative.Dark2)

fig.update_layout(title='Death Rate', xaxis_title="", yaxis_title="DeathRate",xaxis_categoryorder = 'total ascending',

                  uniformtext_minsize=8, uniformtext_mode='hide',xaxis_rangeslider_visible=True)

fig.show()
# Use Boolean indexing to generate a mask which is just a series of boolean values representing whether the column contains the specific element or not

selected = full_grouped['Country/Region'].str.contains('Singapore')



# Apply this mask to our original DataFrame to filter the required values.

singapore = full_grouped[selected]

singapore["New active"] = singapore["Active"].diff()

singapore["New recoverd"] = singapore["Recovered"].diff()



singapore_diff_case = singapore.melt(id_vars="Date", value_vars=['New deaths', 'New cases', 'New recovered'],

                 var_name='Case', value_name='Count')



fig = px.area(singapore_diff_case, x="Date", y="Count", color='Case', height=600, width=1200,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# basic discription of all return and volatility figures

baseline[["sti_return", "sti_volatility_1y", "snp_return", "snp_volatility_1y", "btc_return", "btc_volatility_1y", "gold_return", "gold_volatility_1y", 

                  "high_yield_bond_return", "high_yield_bond_volatility_1y", "inv_grade_bond_return", 

                  "inv_grade_bond_volatility_1y", "wti_return", "wti_volatility_1y", "tbill_return", "tbill_volatility_1y"]].describe()
# All return info no volatility info, then get sharp ratio

statestable = baseline[["sti_return", "snp_return", "btc_return", "gold_return", "high_yield_bond_return", "inv_grade_bond_return", 

                  "wti_return", "tbill_return"]].describe()



return_tbill = statestable.loc["mean","tbill_return"]

statestable.loc["tbill_return"] = return_tbill

statestable.loc["sharp_ratio"] = (statestable.loc["mean"]-statestable.loc["tbill_return"])/statestable.loc["std"]

print(statestable)
def plot_chart(series):

    fig = px.scatter(baseline[baseline[series].notnull()], x="Date", y=series, color="recession", color_discrete_sequence=['#636EFA', '#FFA15A'], width=1200)

    fig.update_traces(mode='markers', marker_size=4)

    fig.update_layout(title=series, xaxis_title="", yaxis_title="")

    fig.show()



plot_chart("sti")
baseline_corr = baseline[["sti_return", "sti_volatility_1y", "snp_return", "snp_volatility_1y", "btc_return", "btc_volatility_1y", "gold_return", "gold_volatility_1y", 

                  "high_yield_bond_return", "high_yield_bond_volatility_1y", "inv_grade_bond_return", 

                  "inv_grade_bond_volatility_1y", "wti_return", "wti_volatility_1y", "tbill_return", "tbill_volatility_1y"]].dropna().corr()



fig, ax = plt.subplots(figsize=(20,10)) 

sns.heatmap(baseline_corr, annot=True, ax = ax)
#Detail exam the correlation between STI and S&P

sns.jointplot(x = 'sti_return', y = 'snp_return', data = baseline, kind='reg')
#Detail exam the correlation between STI and wti

sns.jointplot(x = 'sti_return', y = 'wti_return', data = baseline, kind='reg')
#Detail exam the correlation between STI and high yield bond

sns.jointplot(x = 'sti_return', y = 'high_yield_bond_return', data = baseline, kind='reg')
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

    

# Show if STI volatility fortune tells return in general

plot_chart_vol_ret('sti')



# Show if STI volatility fortune tells return in recession or non-recession period

plot_chart_vol_ret_by_recession('sti')

    

    