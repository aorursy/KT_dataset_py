# Data Management

from dateutil import relativedelta as rd ### PBS_3: ADDING: add the package

import datetime as dt

import pandas as pd

import numpy as np



# Visualization

import matplotlib.pyplot as plt

import plotly as py ### PBS_3: ADDING: add the package

import plotly.express as px

import plotly.graph_objects as go

import plotly.offline as pyo ### PBS_3: CHANGING,import entire package, other PBS only use partial function inside

pyo.init_notebook_mode()

import seaborn as sns



# Regression 

import statsmodels.api as sm ### PBS_3: ADDING: add the package

from statsmodels.formula.api import ols ### PBS_3: ADDING: add the package

import statsmodels.graphics.api as smg ### PBS_3: ADDING: add the package

from scipy.stats import t ### PBS_3: ADDING: add the package

from scipy.stats import chi2_contingency ### PBS_3: ADDING: add the package



!pip install yfinance ### PBS_3: ADDING: pip install package

import yfinance as yf ### PBS_3: ADDING: add the package
### PBS_3: ADDING: ADD THIS SECTION



# FIRST FILE



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



# can use the lines below to check on table inforamtion

# cty_info.describe()

# cty_info.info()

# cty_info.head(20)

# SECOND FILE INPUT



worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')



# Replace missing values '' with NAN and then 0

worldometer_data = worldometer_data.replace('', np.nan).fillna(0)



# Transform variables and round them up to the two decimal points

# Note that there are instances of division by zero issue when there are either zero total tests or total cases

worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases']/worldometer_data['TotalTests'],2)

worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths']/worldometer_data['TotalCases'],2)



# Resolve the division by zero issue by replacing infinity value with zero

worldometer_data[worldometer_data["Case Positivity"] == np.inf] = 0

worldometer_data[worldometer_data["Case Fatality"] == np.inf] = 0 ### PBS_3: ADDING: add this item



### PBS_3: ADDING: add the BINs

# Place case positivity into three bins

worldometer_data ['Case Positivity Bin']= pd.qcut(worldometer_data['Case Positivity'], q=3, labels=["low", "medium", "high"])



# Population Structure

worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')



# Replace missing values with zeros

worldometer_pop_struc = worldometer_pop_struc.fillna(0)



# Merge datasets by common key country

worldometer_data = worldometer_data.merge(worldometer_pop_struc,how='inner',left_on='Country/Region', right_on='Country')

worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]



### PBS_3: ADDING: merge one more item and replace the item below (below all adding)

# Country information

worldometer_data = worldometer_data.merge(cty_info, how='left', on='Country')



### PBS_3: ADDING: merge one more item and replace the item below (below all adding)

# Replace space in variable names with '_'

worldometer_data.columns = worldometer_data.columns.str.replace(' ', '_')



# can use the lines below to check on table inforamtion

# worldometer_data.describe()

# worldometer_data.info()
# THIRD FILE INPUT 



full_table = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

### PBS_3: ADDING: only in PBS3 this line to change date appears

full_table['Date'] = pd.to_datetime(full_table['Date'])



# Examine DataFrame (object type, shape, columns, dtypes)

# full_table.info()



#type(full_table)

#full_table.shape

#full_table.columns

#full_table.dtypes



# Deep dive into the DataFrame

# full_table.head()
# FOURTH FILE INPUT

# Grouped by day, country



full_grouped = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

full_grouped['Date'] = pd.to_datetime(full_grouped['Date'])



### PBS_3: ADDING: the rest are problem set adding



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

full_grouped_SG_Only = full_grouped.loc[full_grouped["Country/Region"].isin(['Singapore'])]



# full_grouped.info()

#full_grouped.tail(20)

#print(full_grouped.iloc[0,0])

# list(full_grouped.columns.values)

# full_grouped.describe()
### PBS_3: ADDING: the rest are problem set adding

# Visualize the missingness isue in the dataset

sns.heatmap(cty_info.isnull(), cbar=False)

# We found lots of governmental item missing

### PBS_3: ADDING: the rest are problem set adding

# try singapore, missing all the data for quarantine etc, therefore we could not test SG

cty_info_sg = cty_info.loc[cty_info["Country"].isin(['Singapore'])]

sns.heatmap(cty_info_sg.isnull(), cbar=False)
# Then we adding Singapore quarantine day (knowing the index is 162 and row is those four)

cty_info.iloc[162,6:12] = dt.date(2020,4,7) # quarantine start from 07042020
# This function is from previous bootcamp, copy paste the one to exam SG here

# Create a function to plot (reusing from the previous BootCamp)



def gt_n(n):

    countries = full_grouped[full_grouped['Confirmed']>n]['Country/Region'].unique()

    temp = full_table[full_table['Country/Region'].isin(countries)]

    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

    temp = temp[temp['Confirmed']>n]

    temp['Log Confirmed'] = np.log(1 + temp['Confirmed'])

    # print(temp.head())



    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()

    min_date.columns = ['Country/Region', 'Min Date']

    # print(min_date.head())



    from_nth_case = pd.merge(temp, min_date, on='Country/Region')

    from_nth_case['Date'] = pd.to_datetime(from_nth_case['Date'])

    from_nth_case['Min Date'] = pd.to_datetime(from_nth_case['Min Date'])

    from_nth_case['N days'] = (from_nth_case['Date'] - from_nth_case['Min Date']).dt.days

    # print(from_nth_case.head())



    fig = px.line(from_nth_case, x='N days', y='Confirmed', color='Country/Region', \

                  title='N days from '+str(n)+' case', height=600)

    fig.show()

    

    fig = px.line(from_nth_case, x='N days', y='Log Confirmed', color='Country/Region', \

                  title='N days from '+str(n)+' case', height=600)

    fig.show()



    

# Function to fatch SG similar countries

def gt_n_sg(minimal,maximum,days_from_n_case):

    # Identify countries with confirmed cases greater than days_from_n_case

    # Then among these countries choose the unique set of countries

    countries = full_grouped[full_grouped['Confirmed']>minimal]['Country/Region'].unique()

    countries_maxlimit = full_grouped[full_grouped['Confirmed']>=maximum]['Country/Region'].unique()

    

    # Filter countries that are in the unique set of countries with confirmed cases greater than minimal

    temp = full_table[full_table['Country/Region'].isin(countries)]

    temp = temp[~ temp['Country/Region'].isin(countries_maxlimit)] # not in the region, so we filter out those very big number countries such as us

    similar_country_sample_defined = temp

    

    # Aggregate (i.e., sum up) confirmed cases by Country/Region and Date

    # Reset the index (it is no longer in running order)

    temp = temp.groupby(['Country/Region', 'Date'])['Confirmed'].sum().reset_index()

    

    # Filter observations starting from the day n case is recorded

    temp = temp[temp['Confirmed']>days_from_n_case]

    # print(temp.head())



    # Filter observations with confirmed cases more than minimal

    temp = temp[temp['Confirmed']<maximum]

    # print(temp.head())

    

    # Identify the start date when confirmed cases exceed minimal for each country

    min_date = temp.groupby('Country/Region')['Date'].min().reset_index()

    

    # Name the columns in the dataframe min_date

    min_date.columns = ['Country/Region', 'Min Date']

    # print(min_date.head())



    # Merge dataframe temp with dataframe min_date by 'Country/Region'

    from_nth_case = pd.merge(temp, min_date, on='Country/Region')

    

    # Convert data type to datetime object

    from_nth_case['Date'] = pd.to_datetime(from_nth_case['Date'])

    from_nth_case['Min Date'] = pd.to_datetime(from_nth_case['Min Date'])

    

    # Create a variable that counts the number of days relative to the day when confirmed cases exceed N

    from_nth_case['N days'] = (from_nth_case['Date'] - from_nth_case['Min Date']).dt.days

    # print(from_nth_case.head())



    # Plot a line graph from dataframe from_nth_case with column 'N days' and 'Confirmed' mapped to x-axis and y-axis, respectively.

    # Distinguish each country by color (system-determined color)

    # str converts n integer into string and "'minimal days from '+ str(n) +' case'" is the title 

    fig = px.line(from_nth_case, x='N days', y='Confirmed', color='Country/Region', 

                  title='N days from '+ str(days_from_n_case) +' case', height=600)

    fig.show()

    

    return similar_country_sample_defined





# Singapore has case = 50000, so we create this first function

# So we compare countries with 10-100k case



# Calling

# similar_country_sample_defined = gt_n_sg(40000,60000,1000)
def plot_gov_action (covid_outcome, gov_action):

    fig = px.scatter(full_grouped[full_grouped[gov_action] != None], x = 'day_rel_to_' + gov_action \

                     , y=covid_outcome, color='Country/Region', \

                     title='N days from ' + gov_action, height=600)

    fig.update_layout(yaxis=dict(range=[0,10]))

    fig.show()



# gov_actions = ['quarantine', 'schools', 'gathering', 'nonessential', 'publicplace']

plot_gov_action('pct_change_Confirmed_per_1000', 'quarantine')
def plot_gov_action_singapore (covid_outcome, gov_action):

    fig = px.scatter(full_grouped_SG_Only[full_grouped_SG_Only[gov_action] != None], x = 'day_rel_to_' + gov_action \

                     , y=covid_outcome, color='Country/Region', \

                     title='N days from ' + gov_action, height=600)

    fig.update_layout(yaxis=dict(range=[0,1]))

    fig.show()



# gov_actions = ['quarantine', 'schools', 'gathering', 'nonessential', 'publicplace']

plot_gov_action_singapore('pct_change_Confirmed_per_1000', 'quarantine')
# For Singapore, according to test on urban population etc, we do not need to worry about the correlation test

# What we could do as we mentioned above, the correlation formula residual is not convincible, therefore, we are not apply it here
# full_grouped['Confirmed_per_1000'].describe()

full_grouped['log_Confirmed_per_1000'] = np.log(full_grouped['Confirmed_per_1000']+1) # avoid 0

# full_grouped['log_Confirmed_per_1000'].describe()



#Plot pairplot with countries organized by WHO regions, show if anything matters

g = sns.pairplot(full_grouped[["log_Confirmed_per_1000", "avghumidity", "avgtemp", "urbanpop", "WHO_Region"]], hue="WHO_Region")
# investigate if there are any relionship on two variables we input in

# plt.matshow(full_grouped.corr())

# plt.show()



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

X = full_grouped[['post_quarantine', 'avghumidity', 'avgtemp', 'urbanpop', 'quarXurbanpop']]

X = sm.add_constant(X)



ols_model=sm.OLS(y,X.astype(float), missing='drop')

result=ols_model.fit()

print(result.summary2())
from statsmodels.graphics.gofplots import ProbPlot



model_norm_residuals = result.get_influence().resid_studentized_internal



QQ = ProbPlot(model_norm_residuals)

plot_lm_2 = QQ.qqplot(line='45', alpha=0.5, color='#4C72B0', lw=1)

plot_lm_2.axes[0].set_title('Normal Q-Q')

plot_lm_2.axes[0].set_xlabel('Theoretical Quantiles')

plot_lm_2.axes[0].set_ylabel('Standardized Residuals');



# annotations

abs_norm_resid = np.flip(np.argsort(np.abs(model_norm_residuals)), 0)

abs_norm_resid_top_3 = abs_norm_resid[:3]

for r, i in enumerate(abs_norm_resid_top_3):

    plot_lm_2.axes[0].annotate(i,

                               xy=(np.flip(QQ.theoretical_quantiles, 0)[r],

                                   model_norm_residuals[i]));
from statsmodels.graphics.regressionplots import plot_leverage_resid2



fig, ax = plt.subplots(figsize=(8,6))

fig = plot_leverage_resid2(result, ax = ax)
import statsmodels.stats.api as sms

from statsmodels.compat import lzip



name = ['Breusch-Pagan Lagrange multiplier statistic', 'p-value',

        'f-value', 'f p-value']

test = sms.het_breuschpagan(result.resid, result.model.exog)

lzip(name, test)
## Function to get the pandemic countries

def get_explosive_countries(df, confirmed_cases, n_explosive_week):

    explosive_country = []

    confirmed_cases_per_1000_at_explosion = []

    

    # identify unique set of countries

    for cty in df['Country/Region'].unique():

        

        # filter observations one country at a time

        country = df[df['Country/Region']==cty]

        #print(cty)

        

        #By plotting the confirmed cases over time,

        #the confirmed cases takes exponential shape after critical mass of n confirmed cases

        

        country = country[country['Confirmed'] > confirmed_cases]



        if len(country): 

        

            # print("... confirmed cases more than " + str(confirmed_cases))  

            # print("... first day is " + str(country.iloc[0,0]))  

            

            country.reset_index(drop=True, inplace=True)

            spread_rate=country['Confirmed'].pct_change(7).values

            explosive_spread_counter=0

            tmp_list=[]



            #Check if there is an explosive growth over one-week period

            for i in range(7,len(spread_rate),7):

                if spread_rate[i] > 1.0: #100% growth over one-week

                    explosive_spread_counter += 1

                    tmp_list.append(country.iloc[i,38]) #confirmed cases per 1000 population

                    # print(tmp_list)

                    

            #Term a country pandemic if doubling effect continued for more than a week        

            if explosive_spread_counter > n_explosive_week: #100% growth over one-week for at least one week

                explosive_country.append(cty)

                confirmed_cases_per_1000_at_explosion.extend(tmp_list)

                

        else: 

            

            pass

            # print("... confirmed cases less than" + str(confirmed_cases))    

    

    

    # print(confirmed_cases_per_1000_at_explosion)

                     

    median_rate = np.quantile(confirmed_cases_per_1000_at_explosion,0.5)

    

    return explosive_country, median_rate
# identify current epicenters: at least 1000 confirmed cases and at least two weekly 100% surge in confirmed cases

explosive_country, median_rate = get_explosive_countries(full_grouped[full_grouped['Confirmed_per_1000'].notnull()],1000,2)



# display the list of explosive countries

print("List of Explosive Countries: ", explosive_country)



# display the median confirmed cases per 1000 population at explosion

print("Median Confirmed Cases per 1000 Population at Explosion: ", str(median_rate))
# create a variable 'explosive_country' indicating whether a country is in the list of explosive country

# turn datatype boolean to integer

worldometer_data['explosive_country'] = worldometer_data['Country/Region'].isin(explosive_country).astype('int')



# filter observations that are not in the list of explosive countries

worldometer_data[worldometer_data['explosive_country'] == 0]
# dependent/target/outcome variable

y = worldometer_data['explosive_country']



# independent/predictor/explanatory variable

X = worldometer_data[['avghumidity', 'avgtemp', 'urbanpop', 'gdp2019', 'healthperpop', 'TotalTests']]



# logit regression

# turn independent variables into floating type (best practice)

# "missing='drop'" drops rows with missing values from the regression

logit_model=sm.Logit(y,X.astype(float), missing='drop' )



# fit logit model into the data

result=logit_model.fit()



# summarize the logit model

print(result.summary2())
# calculate the predicted probability with the parameter estimates and values of Xs (i.e., the independent variables)

y_hats2 = result.predict(X)



# assign the values of predicted probabilities to worldometer_data

worldometer_data['explosive_country_probability'] = y_hats2



# filter countries that are not in the earlier list of countries with explosive number of confirmed cases

next_explosive_countries = worldometer_data[~worldometer_data['Country/Region'].isin(explosive_country)]



# sort dataframe next_explosive_countries by the predicted probabilities in a descending order

# and then display the 20 countries with the highest probabilities of experiencing explosive number of confirmed cases 

# note our covid19 dataset (i.e., confirmed cases) was last updated on 27 July 2020

next_explosive_countries.sort_values(by='explosive_country_probability', ascending=False)[['Country/Region', 'explosive_country_probability']].head(20)
# find the symbol (i.e., google the instrument + "yahoo finance")

# e.g., market/sector index ETF for your chosen country and various asset classes (e.g., Comex Gold's symbol is "GC=F")

symbols_list = ["SE", "SPY"]

start = dt.datetime(2019,8,30)

end = dt.datetime(2020,8,30)

data = yf.download(symbols_list, start=start, end=end)

# data.head()



df = data['Adj Close']

df =df.pct_change()[1:]



plt.figure(figsize=(20,10))

df['SE'].plot()

df['SPY'].plot()

plt.ylabel("Daily returns of SE and SPY")

plt.show()
X = df["SPY"]

y = df["SE"]



# Note the difference in argument order

X = sm.add_constant(X)

model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()

predictions = model.predict(X.astype(float)) # make the predictions by the model



# Print out the statistics

print(model.summary())
# import and merge these two data

sgairline = pd.read_csv('../input/sgairline/SIA.csv')

STI = pd.read_csv('../input/stiidx/STI.csv')



sgairline['Date'] = pd.to_datetime(sgairline['Date'])

sgairline.rename(columns={'Adj Close':'sia'}, inplace=True)

sgairline['sia_return'] = sgairline['sia'].pct_change()



STI['Date'] = pd.to_datetime(STI['Date'])

STI.rename(columns={' Close':'sti'}, inplace=True) # remember there is a space before "Close"

STI['sti_return'] = STI['sti'].pct_change()



sia_sti = STI.merge(sgairline, on = "Date", how = "left")

sia_sti.dropna(inplace = True)

sia_sti['Date'] = pd.to_datetime(sia_sti['Date'])

sia_sti = sia_sti[sia_sti['Date'] >= "2019-08-30"]

sia_sti = sia_sti[["Date","sia_return","sti_return"]]

sia_sti.set_index('Date', inplace = True)



plt.figure(figsize=(20,10))

sia_sti['sia_return'].plot()

sia_sti['sti_return'].plot()

plt.ylabel("Daily returns of SIA and STI")

plt.show()

X = sia_sti["sti_return"]

y = sia_sti["sia_return"]



# Note the difference in argument order

X = sm.add_constant(X)

model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()

predictions = model.predict(X.astype(float)) # make the predictions by the model



# Print out the statistics

print(model.summary())
# import and merge these two data

dbs = pd.read_csv('../input/dbsstock/D05.SI.csv')



dbs['Date'] = pd.to_datetime(dbs['Date'])

dbs.rename(columns={'Adj Close':'dbs'}, inplace=True)

dbs['dbs_return'] = dbs['dbs'].pct_change()



dbs_sti = STI.merge(dbs, on = "Date", how = "left")

dbs_sti.dropna(inplace = True)

dbs_sti['Date'] = pd.to_datetime(dbs_sti['Date'])

dbs_sti = dbs_sti[dbs_sti['Date'] >= "2019-08-30"]

dbs_sti = dbs_sti[["Date","dbs_return","sti_return"]]

dbs_sti.set_index('Date', inplace = True)



plt.figure(figsize=(20,10))

dbs_sti['dbs_return'].plot()

dbs_sti['sti_return'].plot()

plt.ylabel("Daily returns of DBS and STI")

plt.show()
X = dbs_sti["sti_return"]

y = dbs_sti["dbs_return"]



# Note the difference in argument order

X = sm.add_constant(X)

model = sm.OLS(y.astype(float), X.astype(float), missing='drop').fit()

predictions = model.predict(X.astype(float)) # make the predictions by the model



# Print out the statistics

print(model.summary())