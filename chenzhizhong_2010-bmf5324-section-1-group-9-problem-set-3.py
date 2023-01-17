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
# Read and rename column country
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
    
full_grouped.info()
#full_grouped.tail(20)
#print(full_grouped.iloc[0,0])
# full_grouped[full_grouped["quarantine"] != None]["Country/Region"].unique()
# full_grouped[full_grouped['Country/Region'] == 'Germany'][['quarantine','day_rel_to_quarantine']]
# list(full_grouped.columns.values)
# full_grouped.describe()
# Visualize the missingness isue in the dataset
sns.heatmap(cty_info.isnull(), cbar=False)
canada=full_grouped[full_grouped['Country/Region'].isin(['Canada','US','India','China','Australia','Russia','spain','Japen'])]

canada.head()
#reading and viewing the GDP by NAICS classified industries data set
gdp=pd.read_csv("../input/canada-covid19-impact-analysis-dataset/GDP by industry.csv")
gdp.head()
gdp.info()
gdp.columns
#taking transpose of the gdp dataframe for easier analysis
gdp_t=gdp.transpose()
gdp_t.columns=gdp_t.iloc[0]
gdp_t.drop('North American Industry Classification System (NAICS)',inplace=True,axis=0)
gdp_t.head()
#changing the datatypes from object to float 
for x in gdp_t.iloc[:,:]:
    gdp_t[x]=gdp_t[x].apply(lambda y: float(y.replace(',','')))
    
gdp_t.info()
#deriving a new feature which will represent the percentage of increase/decrease in GDP 
x=[]
for i in gdp_t.iloc[:,0:]:
    x.append(((gdp_t.loc['Mar-20',i]-gdp_t.loc['Nov-19',i])/gdp_t.loc['Nov-19',i])*100)
    
print(x)
#appending the newly created feature to the gdp_t dataframe
x=pd.Series(x,name='Increased/Decreased Percentage',index=gdp_t.columns)
gdp_t=gdp_t.append(x,ignore_index=False)

gdp_t.tail()
#plotting the decrease in GDP by industry
plt.figure(figsize=(15,10))
plt.bar(x=gdp_t.columns,height=-gdp_t.loc['Increased/Decreased Percentage'])
plt.xticks(rotation=90)
plt.xlabel('Industry')
plt.ylabel('% Decrease in GDP')
plt.title(' Percentage decrease in GDP by industry ')
plt.show()
#plotting some of the industry and their GDP trend
plt.figure(figsize=(15,10))
plt.plot(gdp_t.iloc[:-1,30:])
plt.xlabel('Months')
plt.ylabel('GDP')
plt.title('GDP trends for Nov-19 to Mar-20 industry wise')
plt.legend(gdp_t.iloc[:-1,28:].columns,loc='lower right')
plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)
plt.plot(gdp_t['Arts, entertainment and recreation  [71]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Arts, entertainment and recreation')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,2)
plt.plot(gdp_t['Accommodation and food services  [72]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Accommodation and food services')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,3)
plt.plot(gdp_t['Non-durable manufacturing industries  [T011]4'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Non-durable manufacturing industries')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.subplot(2,2,4)
plt.plot(gdp_t['Real estate and rental and leasing  [53]'].loc[['Nov-19','Dec-19','Jan-20','Feb-20','Mar-20']])
plt.title('GDP trend for Real estate and rental and leasing')
plt.xlabel('Months')
plt.ylabel('GDP')

plt.tight_layout()
plt.show()
def plot_gov_action (covid_outcome, gov_action):
    fig = px.scatter(canada[canada[gov_action] != None], x = 'day_rel_to_' + gov_action \
                     , y=covid_outcome, color='Country/Region', \
                     title='N days from ' + gov_action, height=600)
    fig.update_layout(yaxis=dict(range=[0,3]))
    fig.show()
# gov_actions = ['quarantine', 'schools', 'gathering', 'nonessential', 'publicplace']

plot_gov_action('pct_change_Confirmed_per_1000', 'quarantine')
plot_gov_action('pct_change_Confirmed_per_1000', 'schools')
plot_gov_action('pct_change_Confirmed_per_1000', 'gathering')
plot_gov_action('pct_change_Confirmed_per_1000', 'nonessential')
plot_gov_action('pct_change_Confirmed_per_1000', 'publicplace')
canada['Confirmed_per_1000'].describe()
canada['log_Confirmed_per_1000'] = np.log(canada['Confirmed_per_1000']+1)
canada['log_Confirmed_per_1000'].describe()
#plt.matshow(full_grouped.corr())
#plt.show()

f, ax = plt.subplots(figsize=(10, 8))
corr = canada.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Plot heatmap
sns.heatmap(corr, mask=mask, cmap=cmap, square=True, ax=ax)
canada.columns
# 'post_schools', 'post_gathering', 'post_nonessential', 'post_publicplace',

# Create interaction term
canada['nonesseXurbanpop'] = canada['post_nonessential'] * canada['urbanpop']

# OLS regression
y = canada['log_Confirmed_per_1000']
X=canada[['post_nonessential','avghumidity', 'avgtemp','urbanpop','nonesseXurbanpop']]
X = sm.add_constant(X)

ols_model=sm.OLS(y,X.astype(float), missing='drop')
result=ols_model.fit()
print(result.summary2())
import statsmodels.stats.api as sms
from statsmodels.compat import lzip

name = ['Breusch-Pagan Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(result.resid, result.model.exog)
lzip(name, test)