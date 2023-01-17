from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
pd.options.display.max_columns=50

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Loading the datasets
shootings = pd.read_csv('/kaggle/input/police-deadly-force-usage-us/fatal-police-shootings-data.csv')
census = pd.read_csv('/kaggle/input/us-census-demographic-data/acs2017_census_tract_data.csv')
crime_rate = pd.read_csv('/kaggle/input/crime-rate-by-state/crime_rate.csv')
education = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding='unicode_escape')


shootings.head()
census.head()
## Changing the state names to their Abbreviations

census.State.replace({'California' : 'CA', 'Texas' : 'TX', 'Florida' : 'FL', 'New York' : 'NY', 'Pennsylvania' : 'PA',
       'Illinois' : 'IL', 'Ohio' : 'OH', 'Georgia' : 'GA', 'North Carolina' : 'NC', 'Michigan' : 'MI',
       'New Jersey' : 'NJ', 'Virginia' : 'VA', 'Washington' : 'WA', 'Arizona' : 'AZ', 'Massachusetts' : 'MA',
       'Tennessee' : 'TN', 'Indiana' : 'IN', 'Missouri' : 'MO', 'Maryland' : 'MD', 'Wisconsin' : 'WI',
       'Colorado' : 'CO', 'Minnesota' : 'MN', 'South Carolina' : 'SC', 'Alabama' : 'AL', 'Louisiana' : 'LA',
       'Kentucky' : 'KY', 'Oregon' : 'OR', 'Oklahoma' : 'OK', 'Connecticut' : 'CT', 'Iowa' : 'IA', 'Utah' : 'UT',
       'Nevada' : 'NV', 'Arkansas' : 'AR', 'Mississippi' : 'MS', 'Kansas' : 'KS', 'New Mexico' : 'NM',
       'Nebraska' : 'NE', 'West Virginia' : 'WV', 'Idaho' : 'ID', 'Hawaii' : 'HI', 'New Hampshire' : 'NH',
       'Maine' : 'ME', 'Montana' : 'MT', 'Rhode Island' : 'RI', 'Delaware' : 'DE', 'South Dakota' : 'SD',
       'North Dakota' : 'ND', 'Alaska' : 'AK', 'District of Columbia' : 'DC', 'Vermont' : 'VT',
       'Wyoming' : 'WY'}, inplace = True)
crime_rate.replace({'California' : 'CA', 'Texas' : 'TX', 'Florida' : 'FL', 'New York' : 'NY', 'Pennsylvania' : 'PA',
       'Illinois' : 'IL', 'Ohio' : 'OH', 'Georgia' : 'GA', 'North Carolina' : 'NC', 'Michigan' : 'MI',
       'New Jersey' : 'NJ', 'Virginia' : 'VA', 'Washington' : 'WA', 'Arizona' : 'AZ', 'Massachusetts' : 'MA',
       'Tennessee' : 'TN', 'Indiana' : 'IN', 'Missouri' : 'MO', 'Maryland' : 'MD', 'Wisconsin' : 'WI',
       'Colorado' : 'CO', 'Minnesota' : 'MN', 'South Carolina' : 'SC', 'Alabama' : 'AL', 'Louisiana' : 'LA',
       'Kentucky' : 'KY', 'Oregon' : 'OR', 'Oklahoma' : 'OK', 'Connecticut' : 'CT', 'Iowa' : 'IA', 'Utah' : 'UT',
       'Nevada' : 'NV', 'Arkansas' : 'AR', 'Mississippi' : 'MS', 'Kansas' : 'KS', 'New Mexico' : 'NM',
       'Nebraska' : 'NE', 'West Virginia' : 'WV', 'Idaho' : 'ID', 'Hawaii' : 'HI', 'New Hampshire' : 'NH',
       'Maine' : 'ME', 'Montana' : 'MT', 'Rhode Island' : 'RI', 'Delaware' : 'DE', 'South Dakota' : 'SD',
       'North Dakota' : 'ND', 'Alaska' : 'AK', 'District of Columbia' : 'DC', 'Vermont' : 'VT',
       'Wyoming' : 'WY'}, inplace = True)
education.isnull().sum()
### Drop null values from the education dataset

education = education.drop(education.loc[education['percent_completed_hs']=='-'].index)
education.rename(columns={'Geographic Area':'state'}, inplace=True)
education.percent_completed_hs = education.percent_completed_hs.astype(float)
census.columns
## Dropping puerto rico since it only appears in the census dataset

census.drop(census[census.State == 'Puerto Rico'].index, inplace=True)
census.rename(columns = {'State':'state'}, inplace=True)
### Creating a dataset for Total population by state and man/women ratio
pop = pd.DataFrame(census.groupby(by='state')[('TotalPop', 'Men', 'Women')].sum()).reset_index()
pop['Men_ratio'] = pop.Men/pop.TotalPop
pop['Women_ratio'] = pop.Women/pop.TotalPop
pop.head()
## creating a dataframe for ratios of race as a percent of total population by each state
race_ratios = pd.DataFrame({'state':[x for x in census.state.unique()]})
def get_share(race):
    countries = [x for x in race_ratios.state]
    share = []
    for country in countries:
            share.append(((census[race].loc[census['state']==country]*census.TotalPop.loc[census['state']==country])/
                    (census.TotalPop.loc[census['state']==country]).sum()).sum())
    race_ratios[race] = share
        
get_share('White')
get_share('Black')
get_share('Hispanic')
get_share('Native')
get_share('Asian')
get_share('Pacific')
race_ratios.head()
## creating a dataframe for rates of socio-economic factors for each states
# using the median for number of voting age citizens, income, and income per capita

socio_eco_factors = pd.DataFrame(census.groupby(by='state')[('VotingAgeCitizen','Income', 'IncomePerCap')].median()).reset_index()

def get_rated(data):
    countries = [x for x in socio_eco_factors.state]
    columns = ['Poverty',
       'ChildPoverty', 'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'Employed', 'PrivateWork', 'PublicWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment']
    for column in (columns):
        rate = []
        for country in (countries):
            rate.append(((census[column].loc[census['state']==country]*census.TotalPop.loc[census['state']==country])/
                (census.TotalPop.loc[census['state']==country]).sum()).sum())
        socio_eco_factors[column] = rate
        
get_rated(socio_eco_factors)
socio_eco_factors.head()
## merging the main dataframe with the ratios by race and the socio economic factors

df=shootings.merge(pop,on='state').merge(race_ratios,on='state').merge(socio_eco_factors, on='state')
df.head()
## creating a feature for the proportions of victims by race

df['percent_killed_race'] = df.race.apply(lambda x: df[df['race']==x].shape[0] / df.shape[0] * 100)
df.race.replace({'A':'Asian', 'H':'Hispanic','W':'White','N':'Native','B':'Black'}, inplace=True)
df[df['race'].isnull()]
df.isnull().sum()
## Function to fill the null age values based on the city and state

def set_med(data, col, col2, col3):
    index_nan = list(data[col][data[col].isnull()].index)
    for i in index_nan:
        state_med = data[col][((data[col2] == data.loc[i][col2]))].median()
        med_fill = data[col][((data[col2] == data.loc[i][col2]) & (data[col3] == data.loc[i][col3]))].median()
        if not np.isnan(med_fill):                  
            data[col].loc[i] = med_fill
        else: data[col].loc[i] = state_med

set_med(df, 'age', 'state', 'city')
df.race.fillna(value='Unknown', inplace= True)
df.flee.fillna(value='Unknown', inplace= True)
df.armed.fillna(value='Unknown', inplace= True)
df.gender.fillna(value='M', inplace= True)
df.isnull().sum()
races = pd.DataFrame({'race' : ['Hispanic', 'White',
       'Black', 'Native', 'Asian']})
population = census['TotalPop'].sum()
df.head()
races['Share Of Population'] = races['race'].apply(lambda x: census.apply(lambda y: y[x]*y['TotalPop']/population, axis=1).sum())
races['Percent Killed By The Police'] = races['race'].apply(lambda x: len(df[df['race']==x])/len(df))*100
races
races.head()
races_plot=races.melt(id_vars='race')
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots(1,1,figsize = (8,5))
sns.barplot('value', 'race',hue='variable', data=races_plot, ax=ax )
for i in ax.patches:
    width = i.get_width()
    plt.text(4+i.get_width(), i.get_y()+0.55*i.get_height(),
             '{:1.2f}%'.format(width),
             ha='center', va='center')
ax.tick_params(axis='both', labelsize=12)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title('Percent Killed vs Percent of Population')
ax.set_xlabel('')
ax.set_ylabel('Race')
ax.set_xticks([])
plt.tight_layout()
## creating a year column in the df dataset

df.date = pd.to_datetime(df.date)
df['year'] = df['date'].apply(lambda x: x.year)
## plotting the number of victims by creating a pivot table based on year and race

g = df[(df['race']!='Unknown') & (df['race']!='O')].pivot_table('id','year','race',aggfunc='count').plot(marker='o')
plt.xlabel('Year')
plt.ylabel('Number Of Victims')
t = plt.title('Total Number Of Victims By Race Over Time')
f = plt.legend()

## grouping the data by month and converting the month to ordinal values so we can plot them afterwards

by_month = df.groupby(pd.Grouper(key='date' ,freq='M')).count().reset_index()[['date', 'id']]
by_month['date_ordinal'] = by_month['date'].apply(lambda x: x.toordinal())

## plotting
years = df.year.unique()
fig, ax = plt.subplots(1,1,figsize = (10,5))
sns.regplot(by_month['date_ordinal'], by_month['id'],ci=95, ax=ax)
labels = [by_month['date_ordinal'].min() + (x * 365) for x in range(6)]
ax.set_xticks(labels)
labels = ax.set_xticklabels(years)
ax.set_xlabel('Year')
l = ax.set_ylabel('Number Of Victims')
t = plt.title('Number Of Victims over the years')


df.columns
sns.set_style('whitegrid')
plt.subplots(3,2, figsize=(13,10))
#fig, (ax1,ax2,ax3,ax4,ax5,ax6,ax7) = plt.subplots(3,3, figsize=(10,8))
cols = ['manner_of_death', 'flee', 'body_camera', 'gender', 'signs_of_mental_illness','threat_level']
for i in range(len(cols)):
    plt.subplot(2,3,i+1)
    sns.barplot(x=df[cols[i]], y=df[cols[i]],orient='v', estimator=lambda x:len(x)/len(df)*100, palette='bright').set(ylabel='Percent')
    if len(df[cols[i]].unique()) >= 3:
        plt.xticks(rotation=75)
plt.tight_layout()
plt.style.use('ggplot')
g = sns.distplot(df['age'])
plt.style.use('fivethirtyeight')
sns.set_palette('BrBG')
g = sns.FacetGrid(df[df['race']!='Unknown'],col ='race',col_wrap=3, height=2.4, aspect=2)
g = g.map(sns.distplot, 'age')
title = plt.suptitle('Distribution of Age by race',x=0.5, y=1.03)
plt.tight_layout()
### reducing the armed column values

df.armed.replace({'undetermined':'Unknown'}, inplace=True)
arm_values = ['gun', 'unarmed', 'toy weapon', 'knife', 'Unknown']
df.armed = df.armed.apply(lambda x: x.replace(x,'other') if x not in arm_values else x)
df.armed.unique()
### creating a column for "is armed"

arm_values = ['unarmed', 'toy weapon', 'Unknown']

df['is_armed'] = df['armed'].apply(lambda x: 'Armed' if x not in arm_values else 'Unarmed')

## creating seperate dataframes for armed and unarmed:


unarmed_data = df[df['is_armed'] == 'Unarmed']
armed_data = df[df['is_armed'] == 'Armed']
### plotting multiple variables comparing armed data and unarmed:

fig, axs = plt.subplots(4,2,figsize=(12, 12))
fig.suptitle('Armed vs Unarmed', x=0.54, y=1.03, fontsize=22)
sns.barplot(x='race', y='race', orient='v',ax=axs[0,0], data=armed_data,
                estimator=lambda x: len(x) / len(armed_data) * 100).set(ylabel='Percent')
sns.barplot(x='race', y='race', orient='v',ax=axs[0,1], data=unarmed_data,
                estimator=lambda x: len(x) / len(unarmed_data) * 100).set(ylabel='Percent')
## gender
sns.barplot(x='gender', y='gender', orient='v',ax=axs[1,0], data=armed_data,
                estimator=lambda x: len(x) / len(armed_data) * 100).set(ylabel='Percent')
sns.barplot(x='gender', y='gender', orient='v',ax=axs[1,1], data=unarmed_data,
                estimator=lambda x: len(x) / len(unarmed_data) * 100).set(ylabel='Percent')

## age
sns.barplot(x='age', y='age', orient='v',ax=axs[2,0], data=armed_data,
                estimator=lambda x: len(x) / len(armed_data) * 100).set(xticks=(range(0, 90, 10)),
                                                                        xticklabels=(range(0, 90, 10)), ylabel='Percent')
sns.barplot(x='age', y='age', orient='v',ax=axs[2,1], data=unarmed_data,
                estimator=lambda x: len(x) / len(unarmed_data) * 100).set(xticks=(range(0, 90, 10)),
                                                                        xticklabels=(range(0, 90, 10)), ylabel='Percent')


## weapon
sns.barplot(x='armed', y='armed', orient='v',ax=axs[3,0], data=armed_data,
                estimator=lambda x: len(x) / len(armed_data) * 100).set(ylabel='Percent', xlabel='weapon')
sns.barplot(x='armed', y='armed', orient='v',ax=axs[3,1], data=unarmed_data,
                estimator=lambda x: len(x) / len(unarmed_data) * 100).set(ylabel='Percent',xlabel='weapon')

plt.tight_layout()
## creating A dataframe by state:

data_by_state = df.groupby('state')['id'].count().reset_index()
data_by_state = data_by_state.merge(pop,on='state').merge(race_ratios,on='state').merge(socio_eco_factors, on='state')
data_by_state.rename(columns={'id': 'Number Of Victims', 'IncomePerCap': 'Income Per Capita'}, inplace = True)
data_by_state['victims per 100K citizens'] = data_by_state['Number Of Victims']/(data_by_state['TotalPop']/100000)
data_by_state = data_by_state.merge(crime_rate, on='state')
data_by_state.crime_rate = data_by_state.crime_rate.apply(lambda x: x.replace(',',''))
data_by_state.crime_rate = data_by_state.crime_rate.astype(float)
data_by_state.head()
countries = [x for x in data_by_state.state]
rates = []
for country in countries:
        rates.append(((education['percent_completed_hs'].loc[education['state']==country]).median()))
data_by_state['HS_over_25'] = rates
data_by_state.head()
### Using plotly interactive figures to show the number of victims for each state

fig = go.Figure(data=go.Choropleth(
    locations=data_by_state['state'],
    z = data_by_state['victims per 100K citizens'],
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "Deaths"
))

fig.update_layout(
    title_text = 'Victims per 100K citizens by state',
    geo_scope='usa'
)

fig.show()
top_cors = abs(data_by_state.corr()).nlargest(15, 'victims per 100K citizens').index
plt.figure(figsize = (12,8))
cormap = np.corrcoef(data_by_state[top_cors].values.T).round(2)
g = sns.heatmap(cormap, cbar=True, annot=True, cmap='BrBG',yticklabels = top_cors.values, xticklabels=top_cors.values)
plt.tight_layout()
plt.figure(figsize=(12,4))
plt.style.use('seaborn')
g = sns.regplot(x='Income Per Capita', y='victims per 100K citizens', data=data_by_state)
for i in range(0,data_by_state.shape[0]):
     g.text(data_by_state['Income Per Capita'][i]*1.005, data_by_state['victims per 100K citizens'][i],
            data_by_state.state[i], horizontalalignment='left', size=11, color='black', weight='semibold')
title = plt.title('Correlation Between Income Per Capita And Victims')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_by_state['Income Per Capita'],
                                y=data_by_state['victims per 100K citizens'],
                                mode='markers+text',
                                marker_color=data_by_state['victims per 100K citizens'],
                                text=data_by_state['state'],
                                textposition='top center'))
              

fig.update_layout(xaxis_title='Income Per Capita', yaxis_title = 'victims per 100K citizens',template ='ggplot2', title='Correlation Between Income Per Capita And Victims')
fig.show()
plt.figure(figsize=(12,4))
plt.style.use('seaborn')
g = sns.regplot(x='Poverty', y='victims per 100K citizens', data=data_by_state)
plt.xlabel('Poverty Rate')
for i in range(0,data_by_state.Poverty.shape[0]):
     g.text(data_by_state['Poverty'][i]*1.005, data_by_state['victims per 100K citizens'][i],
            data_by_state.state[i], horizontalalignment='left', size=11, color='black', weight='semibold')
title = plt.title('Correlation Between Poverty Rate And Victims')
plt.figure(figsize=(12,4))
plt.style.use('seaborn')
g = sns.regplot(x='crime_rate', y='victims per 100K citizens', data=data_by_state)
plt.xlabel('Crime Rate')
for i in range(0,data_by_state.Poverty.shape[0]):
     g.text(data_by_state['crime_rate'][i]*1.005, data_by_state['victims per 100K citizens'][i],
            data_by_state.state[i], horizontalalignment='left', size=11, color='black', weight='semibold')
title = plt.title('Correlation Between Crime Rate And Victims')
fig = go.Figure()
fig.add_trace(go.Scatter(x=data_by_state['crime_rate'],
                                y=data_by_state['victims per 100K citizens'],
                                mode='markers+text',
                                marker_color=data_by_state['victims per 100K citizens'],
                                text=data_by_state['state'],
                                textposition='top center'))
              

fig.update_layout(xaxis_title='Crime Rate', yaxis_title = 'Victims per 100K citizens',template ='ggplot2', title='Correlation Between Crime Rate And Victims')
fig.show()
### importing the packages

import xgboost as xgb
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold, GridSearchCV , RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC, LinearRegression
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from scipy.stats import norm
from scipy import stats
from sklearn.metrics import mean_squared_error 
### performing a log transformation on the target variable

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
sns.distplot(data_by_state['victims per 100K citizens'], fit=norm)

plt.subplot(1,2,2)
res = stats.probplot(data_by_state['victims per 100K citizens'], plot=plt)
plt.tight_layout()

y = np.log1p(data_by_state['victims per 100K citizens'].values)
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
g = sns.distplot(y, fit=norm)

plt.subplot(1,2,2)
res = stats.probplot(y, plot=plt)

plt.tight_layout()
data_by_state.columns
### selecting the features for the first x

cols_to_use = ['state', 'TotalPop', 'Men_ratio',
       'White', 'Black', 'Hispanic', 'Native', 'Asian',
       'Pacific', 'Income Per Capita', 'Poverty',
       'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'PrivateWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment',
       'crime_rate', 'HS_over_25']

x = data_by_state[cols_to_use]
x = pd.get_dummies(x)
scaler = StandardScaler()
x = scaler.fit_transform(x)
## splitting the data to train and test

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.7)
# Defining a function which examines each model based on the score, then show each one's score and STD, as well as graphic comparison
# evaluate each model in turn
def get_scores(score1, score2,x_train,x_test):
    models = []
    models.append(('Lasso', Lasso()))
    models.append(('RandomForest', RandomForestRegressor()))
    models.append(('XGB', xgb.XGBRegressor(objective='reg:squarederror')))
    models.append(('LR', LinearRegression()))
    models.append(('SVR', SVR()))
    models.append(('Enet',ElasticNet()))
    models.append(('LightGBM',lgb.LGBMRegressor()))
    models.append(('Bayes',BayesianRidge()))
    models.append(('GB',GradientBoostingRegressor()))

    cv_scores = []
    test_scores = []
    names = []
    stds = []
    differences = []
    res = pd.DataFrame()
    for index, model in enumerate(models):
        kfold = KFold(n_splits=7)
        cv_results = abs(cross_val_score(model[1], x_train, y_train, cv=kfold, scoring=score1))
        cv_scores.append(cv_results)
        names.append(model[0])
        model[1].fit(x_train,y_train)
        predictions = model[1].predict(x_test)
        test_score = score2(predictions, y_test)
        test_scores.append(test_score)
        stds.append(cv_results.std())
        differences.append((cv_results.mean() - test_score))
        res.loc[index,'Model'] = model[0]
        res.loc[index,score1+('(CV)')] = cv_results.mean()
        res.loc[index,score1+('(Test_Data)')] = test_score
        res.loc[index,'Std'] = cv_results.std()
        res.loc[index,'difference'] = cv_results.mean() - test_score
    # boxplot algorithm comparison
    fig = plt.figure(figsize = (12,5))
    fig.suptitle('Model Comparison')
    ax = fig.add_subplot(121)
    plt.boxplot(cv_scores)
    ax.set_xticklabels(names, rotation=70)
    axs = fig.add_subplot(122)
    sns.barplot(names,test_scores)
    axs.set_xticklabels(names, rotation=70)
    plt.tight_layout(pad=5)
    return res
    plt.show()

    

get_scores('neg_mean_squared_error', mean_squared_error, x_train, x_test)
## selecting the features for the second x

cols_to_use2 = ['TotalPop', 'Men_ratio',
       'White', 'Black', 'Hispanic', 'Native', 'Asian',
       'Pacific', 'Income Per Capita', 'Poverty',
       'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'PrivateWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment',
       'crime_rate', 'HS_over_25']
x2 = data_by_state[cols_to_use2]
scaler = StandardScaler()
x2 = scaler.fit_transform(x2)
x2_train, x2_test, y_train, y_test = train_test_split(x2,y,train_size=0.7)
get_scores('neg_mean_squared_error', mean_squared_error, x2_train, x2_test)
### third x

cols_to_use3 = ['Men_ratio',
       'White', 'Black', 'Hispanic', 'Native', 'Asian',
       'Pacific', 'Income Per Capita', 'Poverty',
       'Professional', 'Service', 'Office', 'Construction',
       'Production', 'Drive', 'Carpool', 'Transit', 'Walk', 'OtherTransp',
       'WorkAtHome', 'MeanCommute', 'PrivateWork',
       'SelfEmployed', 'FamilyWork', 'Unemployment',
       'crime_rate', 'HS_over_25']
x3 = data_by_state[cols_to_use3]
scaler = StandardScaler()
x3 = scaler.fit_transform(x3)
x3_train, x3_test, y_train, y_test = train_test_split(x3,y,train_size=0.7)
get_scores('neg_mean_squared_error', mean_squared_error,x3_train,x3_test)
params_GB = {'n_estimators':[100,200,500,800,1200],
    'max_depth':[10, 20, 30, 40, 50, 60, 70, None],
    'min_samples_split':[1,2,3,5],
    'min_samples_leaf':[1,2,3,5],
    'max_features': ['auto', 'sqrt'],
    'alpha':[0.03,0.06,0.09,0.5,0.9,2],
     'learning_rate':[0.1,0.2,0.5,0.8]}
model_GB=GradientBoostingRegressor()
randomgrid_GB = RandomizedSearchCV(estimator=model_GB, param_distributions = params_GB, 
                               cv=5, n_iter=25, scoring = 'neg_mean_squared_error',
                               n_jobs = 4, verbose = 3, random_state = 42,
                               return_train_score = True)
randomgrid_GB.fit(x3_train,y_train)
best_GB = randomgrid_GB.best_estimator_
GB_preds = best_GB.predict(x3_test)
mean_squared_error(GB_preds, y_test)
featuers_coefficients = best_GB.feature_importances_.round(6).tolist()
feature_names = cols_to_use3
feats = pd.DataFrame(pd.Series(featuers_coefficients, feature_names).sort_values(ascending=False),columns=['Coefficient'])
feats