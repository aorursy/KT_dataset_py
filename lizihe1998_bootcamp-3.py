# Data Management

from dateutil import relativedelta as rd

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import plotly as py

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo

pyo.init_notebook_mode()

import seaborn as sns



# Regression 

import statsmodels.api as sm

from statsmodels.formula.api import ols

import statsmodels.graphics.api as smg
cty_info = pd.read_csv('../input/countryinfo/covid19countryinfo.csv').rename(columns={'country':'Country'})



# Filter observations with aggregate country-level information

# The column region for region-level observations is populated

cty_info = cty_info[cty_info.region.isnull()]



# Convert string data type to floating data type

# Remove comma from the fields

cty_info['healthexp'] = cty_info[~cty_info['healthexp'].isnull()]['healthexp'].str.replace(',','').astype('float')

cty_info['gdp2019'] = cty_info[~cty_info['gdp2019'].isnull()]['gdp2019'].str.replace(',','').astype('float')



# Convert to date objects with to_datetime method

gov_actions = ['quarantine', 'schools', 'gathering', 'nonessential', 'publicplace']



for gov_action in gov_actions:

    cty_info[gov_action] = pd.to_datetime(cty_info[gov_action], format = '%m/%d/%Y')

    

# Filter columns of interest

# Note: feel free to explore other variables or datasets

cty_info = cty_info[['Country','avghumidity', 'avgtemp', 'fertility', 'medianage', 'urbanpop', 'quarantine', 'schools', \

                    'publicplace', 'gatheringlimit', 'gathering', 'nonessential', 'hospibed', 'smokers', \

                    'sex0', 'sex14', 'sex25', 'sex54', 'sex64', 'sex65plus', 'sexratio', 'lung', 'femalelung', \

                    'malelung', 'gdp2019', 'healthexp', 'healthperpop']]



# cty_info.describe()

cty_info.info()

#cty_info.head(20)

cty_info.head()

cty_info[cty_info['schools'].isna()==0].head(20)
# Worldometer data

# ================



worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')



# Replace missing values '' with NAN and then 0

worldometer_data = worldometer_data.replace('', np.nan).fillna(0)



# Transform variables and round them up to the two decimal points

# Note that there are instances of division by zero issue when there are either zero total tests or total cases

worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases']/worldometer_data['TotalTests'],2)

worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths']/worldometer_data['TotalCases'],2)



# Resolve the division by zero issue by replacing infinity value with zero

worldometer_data[worldometer_data["Case Positivity"] == np.inf] = 0

worldometer_data[worldometer_data["Case Fatality"] == np.inf] = 0



# Place case positivity into three bins

worldometer_data ['Case Positivity Bin']= pd.qcut(worldometer_data['Case Positivity'], q=3, labels=["low", "medium", "high"])



# Population Structure

worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')



# Replace missing values with zeros

worldometer_pop_struc = worldometer_pop_struc.fillna(0)



# Merge datasets by common key country

worldometer_data = worldometer_data.merge(worldometer_pop_struc,how='inner',left_on='Country/Region', right_on='Country')

worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]



# Country information

worldometer_data = worldometer_data.merge(cty_info, how='left', on='Country')



# Replace space in variable names with '_'

worldometer_data.columns = worldometer_data.columns.str.replace(' ', '_')



worldometer_data.describe()

worldometer_data.info()
# Full data

# =========



full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

full_table['Date'] = pd.to_datetime(full_table['Date'])



# Examine DataFrame (object type, shape, columns, dtypes)

full_table.info()



#type(full_table)

#full_table.shape

#full_table.columns

#full_table.dtypes



# Deep dive into the DataFrame

full_table.head()
# Grouped by day, country

# =======================



full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])

#full_grouped.loc[full_grouped['Country/Region'] == 'US', 'Country/Region'] = 'USA'

full_grouped.head()



# Correct country names in worldometer to make them consistent with dataframe full_grouped column Country/Region before merging 

worldometer_data['Country/Region'].replace({'USA':'US', 'UAE':'United Arab Emirates', 'S. Korea':'South Korea', \

                                           'UK':'United Kingdom'}, inplace=True)



# Draw population and country-level data

full_grouped = full_grouped.merge(worldometer_data[['Country/Region', 'Population']], how='left', on='Country/Region')

full_grouped = full_grouped.merge(cty_info, how = 'left', left_on = 'Country/Region', right_on = 'Country')

full_grouped['Confirmed per 1000'] = full_grouped['Confirmed'] / full_grouped['Population'] * 1000



# Backfill data

full_grouped = full_grouped.fillna(method='ffill')



# Create post-invention indicators

gov_actions = ['quarantine', 'schools', 'gathering', 'nonessential', 'publicplace']



for gov_action in gov_actions:

    full_grouped['post_'+gov_action] = full_grouped['Date'] >= full_grouped[gov_action]

    full_grouped['day_rel_to_' + gov_action] = (full_grouped['Date'] - full_grouped[gov_action]).dt.days



# Create percent changes in covid19 outcomes

covid_outcomes = ['Confirmed', 'Deaths', 'Recovered', 'Active', 'Confirmed per 1000']



for covid_outcome in covid_outcomes:

    full_grouped['pct_change_' + covid_outcome] = full_grouped.groupby(['Country/Region'])[covid_outcome].pct_change()

    full_grouped[full_grouped['pct_change_' + covid_outcome] == np.inf] = 0

    







# Replace space in variable names with '_'

full_grouped.columns = full_grouped.columns.str.replace(' ', '_')



for gov_action in gov_actions:

    full_grouped['Confirmed_per_1000_at_'+gov_action] = full_grouped[full_grouped['Date'] == full_grouped[gov_action]]['Confirmed_per_1000']

    

full_grouped['log_Confirmed_per_1000'] = np.log(full_grouped['Confirmed_per_1000']+1)

full_grouped.info()

#full_grouped.tail(20)

#print(full_grouped.iloc[0,0])

# full_grouped[full_grouped["quarantine"] != None]["Country/Region"].unique()

# full_grouped[full_grouped['Country/Region'] == 'Germany'][['quarantine','day_rel_to_quarantine']]

# list(full_grouped.columns.values)

# full_grouped.describe()
#plt.matshow(full_grouped.corr())

#plt.show()



f, ax = plt.subplots(figsize=(10, 8))

corr = full_grouped.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr, dtype=bool))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Plot heatmap

sns.heatmap(corr, mask=mask, cmap=cmap, square=True, ax=ax)
# Create interaction term

full_grouped['quarXurbanpop'] = full_grouped['post_quarantine'] * full_grouped['urbanpop']



# OLS regression

y = full_grouped['log_Confirmed_per_1000']

X = full_grouped[['day_rel_to_quarantine','avghumidity', 'avgtemp', 'urbanpop']]

X = sm.add_constant(X)



ols_model=sm.OLS(y,X.astype(float), missing='drop')

result=ols_model.fit()

print(result.summary2())
NZD= pd.read_csv('../input/nzdusd/NZDUSD.csv')

recession=pd.read_csv('../input/nber-based-recession-indicators-united-states/USRECD.csv')
NZD.head()
NZD['Date'] = pd.to_datetime(NZD['Date']) #转换成date

NZD.rename(columns={'Adj Close':'nzdusd'}, inplace=True)

NZD['nzd_return'] = NZD['nzdusd'].pct_change()

NZD = NZD[['Date','nzdusd','nzd_return']]



recession["Date"] = pd.to_datetime(recession["date"])

recession["recession"] = recession["value"].astype('bool')



# Subset data columns



recession = recession[["Date","recession"]]

baseline = pd.merge(NZD,recession,how='left',on='Date') #how=left把snp的东西都留下
baseline.head()
plt.figure(figsize=(8,6))

plt.hist(baseline[baseline['recession']==0]['nzdusd'], bins=100, label="non-recession")

plt.hist(baseline[baseline['recession']==1]['nzdusd'], bins=100, label="recession")



plt.legend(loc='upper right')

plt.show()

len(baseline)
from scipy.stats import t



mean_diff=baseline[baseline['recession']==0]['nzdusd'].mean()-baseline[baseline['recession']==1]['nzdusd'].mean()

#MSE=(S1^2+S2^2)/2

MSE=(baseline[baseline['recession']==0]['nzdusd'].var()+baseline[baseline['recession']==1]['nzdusd'].var())/2

S_diff=np.sqrt(4*MSE/len(baseline))

tt=mean_diff/S_diff

df=len(baseline)-2



pval=t.sf(np.abs(tt),df)*2

print("Point estimate of difference: " + str(mean_diff))

print("MSE : " + str(MSE))

print("Degree of freedom : " + str(df))

print("T-statistic : " + str(tt))



alpha = 0.05

print("P-value : " + str(pval))

if pval <= alpha: 

    print('There is a difference in NZD/USD in recession(reject H0)') 

else: 

    print('There is no difference in in NZD/USD in recession(fail to reject H0)') 

confidence_level = 0.95



confidence_interval = t.interval(confidence_level, df, mean_diff, S_diff)



print("Point estimate : " + str(mean_diff))

print("Confidence interval (0.025, 0.975) : " + str(confidence_interval))
stimulus = pd.read_csv('../input/stimulus1/CESI_3.csv')

stimulus.head()

stimulus.info()
# OLS regression

y = stimulus['CESI_3']

X = stimulus[['fiscal','ratecut', 'reserve requirement and buffer', 'macrofin']]

X = sm.add_constant(X)



ols_model=sm.OLS(y,X.astype(float), missing='drop')

result=ols_model.fit()

print(result.summary2())