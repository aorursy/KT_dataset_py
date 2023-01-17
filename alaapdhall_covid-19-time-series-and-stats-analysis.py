import numpy as np 

import matplotlib.pyplot as plt 

import pandas as pd 

import random

import math

import time

from sklearn.model_selection import RandomizedSearchCV, train_test_split

from sklearn.svm import SVR

from sklearn.metrics import mean_squared_error, mean_absolute_error

import datetime



import torch
#Use this to fetch latest data



confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

csse_daily_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-16-2020.csv')
confirmed_df

# confirmed_df has time series data of cases over time
recoveries_df

# same as confirmed_df, recoveries_df also has time series data of recoveries over time
all(confirmed_df.keys()[4:]==deaths_df.keys()[4:])
all(deaths_df.keys()[4:]==recoveries_df.keys()[4:])
csse_daily_df
csse_daily_df['Country_Region'].value_counts()

# this is the most recent data of Covid19 cases!
%pip install geopandas
# let's look at the csse daily data

import seaborn as sns

from shapely.geometry import Point

import geopandas as gpd

from geopandas import GeoDataFrame



geometry = [Point(xy) for xy in zip(csse_daily_df['Long_'], csse_daily_df['Lat'])]

gdf = GeoDataFrame(csse_daily_df[['Lat','Long_']], geometry=geometry)   



#this is a simple map that goes with geopandas

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

gdf.plot(ax=world.plot(figsize=(15, 10)), marker='o', color='red', markersize=15);
# Top 7 countries

csse_daily_df['Country_Region'].value_counts()[:7]

# US has over 3000 entries in data
top_c = csse_daily_df.groupby(['Country_Region']).Confirmed.sum().sort_values(ascending=False)

top_c
# Top 50 Country wise Confirmed cases acc to given data



fig, ax = plt.subplots(figsize=(10, 13))

sns.barplot(y=top_c.index[:50], x=top_c.values[:50])
csse_daily_df_us = csse_daily_df[csse_daily_df['Country_Region']=='US'] # get US only data



fig, ax = plt.subplots(figsize=(17, 7))

sns.distplot(csse_daily_df_us['Confirmed'].dropna())
# Drop Unnecessary Columns

csse_daily_df_us.drop(['Country_Region','Province_State','Long_','Last_Update','Combined_Key'], axis=1, inplace=True)



csse_daily_df_us.Recovered.value_counts()
# since recovered has only 1 statis value, we will simply drop it too



csse_daily_df_us.drop(['Recovered'], axis=1, inplace=True)
csse_daily_df_us.info()
print("{} \nNan values found".format(csse_daily_df_us.isna().sum()))

csse_daily_df_us.dropna(inplace=True) # drop na
sns.pairplot(csse_daily_df_us[['Confirmed','Deaths','Active','Incidence_Rate','Case-Fatality_Ratio']])
fig, ax = plt.subplots(figsize=(12, 8))

corr = csse_daily_df_us.corr()

sns.heatmap(corr, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
# Label Encode Admin2 columns



from sklearn.preprocessing import LabelEncoder



admin2 = LabelEncoder()



csse_daily_df_us['Admin2'] = admin2.fit_transform(csse_daily_df_us['Admin2'])

csse_daily_df_us
from scipy.stats import pearsonr



should_drop = []

for idx,i in enumerate(['FIPS', 'Admin2', 'Lat', 'Active','Incidence_Rate', 'Case-Fatality_Ratio']):

    for j in ['Confirmed','Deaths']: # go through columns and find corr

        corr, _ = pearsonr(csse_daily_df_us[i],csse_daily_df_us[j]) # use scipy

        print(i,"has corr value =",corr,"with",j)

        if corr > 0.85: 

            should_drop.append(i)

#             csse_daily_df_us.drop(j,axis=1,inplace = True) # drop it

            

    

should_drop
# LEt's keep Confirmed a target variable

X_conf = csse_daily_df_us.drop(['Confirmed'],axis=1)

y_conf = csse_daily_df_us['Confirmed']



from sklearn.feature_selection import SelectKBest, chi2



best_feature = SelectKBest(score_func= chi2, k = 'all')

best_feature = best_feature.fit(X_conf,y_conf)



col_scores = pd.DataFrame(best_feature .scores_)

col_names = pd.DataFrame(X_conf.columns)



feature_score = pd.concat([col_names, col_scores], axis=1)

feature_score.columns = ['attribute', 'score']

feature_score
# get a list of dates 

dates = confirmed_df.keys()[4:]



confirmed = confirmed_df.loc[:, dates]

deaths = deaths_df.loc[:, dates]

recoveries = recoveries_df.loc[:, dates]



confirmed
total_cases = confirmed.sum(axis=0).values

total_deaths = deaths.sum(axis=0).values

total_recoveries = recoveries.sum(axis=0).values



# let's find how many are active still, those who didn't die or recovered but were daignozed

total_active = total_cases - total_deaths - total_recoveries
print("total cases accumulated = {} \noverall total_cases = {}".format(max(total_cases),max(confirmed['7/13/20'])))
# get the unique countries

countries = confirmed_df['Country/Region'].unique()

len(countries)
def p(x):

    return x.loc[:,dates].sum(axis=1).values[0]



worst_countries = confirmed_df.groupby('Country/Region').apply(p).sort_values(ascending = False)[:50]

worst_countries[:10]
fig, ax = plt.subplots(figsize=(10, 10))

sns.barplot(y=worst_countries.index[:25], x=worst_countries.values[:25])
# This method is imp because we have data in accumulative format

def daily_values(data):

    d = [] 

    d.append(data[0])

    for i in range(1,len(data)):

        d.append(data[i]-data[i-1]) # get unique date for the day, since it is accumulative

    return d 



def weekly_average(data):

    weekly_average = []

    for i in range(len(data)):

        if i + 7 < len(data):

            weekly_average.append(np.mean(data[i:i+7]))

        else:

            weekly_average.append(np.mean(data[i:len(data)]))

    return weekly_average





# mortality rate

mortality_rate = np.array(daily_values(total_deaths))/np.array(daily_values(total_cases))



#recovery rate

recovery_rate = np.array(daily_values(total_recoveries))/np.array(daily_values(total_cases))
#days array

days = np.array(range(len(dates))).reshape(-1, 1)



# confirmed cases

global_daily_values = daily_values(total_cases)

global_daily_increase_avg = weekly_average(global_daily_values)
fig, ax = plt.subplots(figsize=(12, 7))



sns.barplot(x = list(range(len(global_daily_values))),y = global_daily_values)

plt.plot(days, global_daily_increase_avg, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.legend(['Weekly Average'], prop={'size': 15})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

# Daily Deaths

global_daily_deaths = daily_values(total_deaths)

global_daily_death_avg = weekly_average(global_daily_deaths)



fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(x = list(range(len(global_daily_deaths))),y=global_daily_deaths)

plt.plot(days, global_daily_death_avg, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Deaths Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Deaths', size=15)

plt.legend(['Weekly Average'], prop={'size': 15})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
# Daily Recoveries

global_daily_recoveries = daily_values(total_recoveries)

global_daily_recoveries_avg = weekly_average(global_daily_recoveries)





fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(x = list(range(len(global_daily_recoveries))),y=global_daily_recoveries)

plt.plot(days, global_daily_recoveries_avg, linestyle='dashed', color='orange')

plt.title('# of Coronavirus Cases Recoveries Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Recoveries', size=15)

plt.legend(['Weekly Average'], prop={'size': 15})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
#Active Cases

global_daily_active = daily_values(total_active)

global_active_avg = weekly_average(global_daily_active)





fig, ax = plt.subplots(figsize=(12, 7))

sns.barplot(x = list(range(len(global_daily_active))),y=global_daily_active)

plt.plot(days, global_active_avg, linestyle='dashed', color='orange')

plt.title('# of Active Coronavirus Cases', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Active Cases', size=15)

plt.legend(['Weekly Average'], prop={'size': 15})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
plt.figure(figsize=(12, 7))

sns.pointplot(x = days[:,0] , y = np.log(global_daily_values))

plt.title('Log of # of Coronavirus Cases Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()
plt.figure(figsize=(12, 7))

sns.pointplot(x = days[:,0] , y = np.log(global_daily_deaths))

plt.title('Log of # of Coronavirus Deaths Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()
plt.figure(figsize=(12, 7))

sns.pointplot(x = days[:,0] , y = np.log(global_daily_recoveries))

plt.title('Log of # of Coronavirus Recoveries Per Day', size=20)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()
plt.figure(figsize=(12, 7))



sns.pointplot(days[:,0], mortality_rate, color='Green')

plt.axhline(y = np.mean(mortality_rate),linestyle='-', color='red')

plt.title('Mortality Rate of Coronavirus = {0:2f}'.format(np.mean(mortality_rate)), size=25)

plt.legend(['mortality rate', 'y='+str(np.mean(mortality_rate))], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('Case Mortality Rate', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()
plt.figure(figsize=(12, 7))



sns.pointplot(days[:,0], recovery_rate, color='red')

plt.axhline(y = np.mean(recovery_rate),linestyle='-', color='black')

plt.title('Recovery Rate of Coronavirus = {0:2f}'.format(np.mean(recovery_rate)), size=25)

plt.legend(['recovery rate', 'y='+str(np.mean(recovery_rate))], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('Case Recovery Rate', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.grid()
#Draw a joint reg plot of mortality rate and recovery rate



sns.jointplot(mortality_rate, recovery_rate , kind="reg", size=7)

plt.show()
!pip install autoviz 



from autoviz.AutoViz_Class import AutoViz_Class

from IPython.display import display # display from IPython.display



AV = AutoViz_Class()
# Let's now visualize the plots generated by AutoViz.

report_2 = AV.AutoViz('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/08-16-2020.csv')
con_wise_confirm_cases = {}

con_wise_recovered_cases = {}

con_wise_death_cases = {}



for c in worst_countries.index:

    con_wise_confirm_cases[c] = confirmed_df[confirmed_df['Country/Region']==c].loc[:,dates].sum(axis = 0).values

    con_wise_death_cases[c] = deaths_df[deaths_df['Country/Region']==c].loc[:,dates].sum(axis = 0).values

    con_wise_recovered_cases[c] = recoveries_df[recoveries_df['Country/Region']==c].loc[:,dates].sum(axis = 0).values

    

# Let's look at confirmed cases in India

con_wise_confirm_cases['India']
# list of all the countries!!

con_wise_confirm_cases.keys()
india_daily = daily_values(con_wise_confirm_cases['India'])

india_avg = weekly_average(india_daily)



india_daith_daily = daily_values(con_wise_death_cases['India'])

india_death_avg = weekly_average(india_daith_daily)



india_recovery_daily = daily_values(con_wise_recovered_cases['India'])

india_recovert_avg = weekly_average(india_recovery_daily)



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], india_daily)

plt.title('India Confirmed Cases', size=20)

plt.plot(days, india_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], india_daith_daily)

plt.title('India Death Cases', size=20)

plt.plot(days, india_death_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], india_recovery_daily)

plt.title('India Recovery Cases', size=20)

plt.plot(days, india_recovert_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
con_daily = daily_values(con_wise_confirm_cases['Italy'])

con_avg = weekly_average(con_daily)



con_daith_daily = daily_values(con_wise_death_cases['Italy'])

con_death_avg = weekly_average(con_daith_daily)



con_recovery_daily = daily_values(con_wise_recovered_cases['Italy'])

con_recovert_avg = weekly_average(con_recovery_daily)



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_daily)

plt.title('Italy Confirmed Cases', size=20)

plt.plot(days, con_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_daith_daily)

plt.title('Italy Death Cases', size=20)

plt.plot(days, con_death_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_recovery_daily)

plt.title('Italy Recovery Cases', size=20)

plt.plot(days, con_recovert_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
con_daily = daily_values(con_wise_confirm_cases['US'])

con_avg = weekly_average(con_daily)



con_daith_daily = daily_values(con_wise_death_cases['US'])

con_death_avg = weekly_average(con_daith_daily)



con_recovery_daily = daily_values(con_wise_recovered_cases['US'])

con_recovert_avg = weekly_average(con_recovery_daily)



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_daily)

plt.title('US Confirmed Cases', size=20)

plt.plot(days, con_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_daith_daily)

plt.title('US Death Cases', size=20)

plt.plot(days, con_death_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()



plt.figure(figsize=(13, 7))

sns.barplot(days[:,0], con_recovery_daily)

plt.title('US Recovery Cases', size=20)

plt.plot(days, con_recovert_avg, color='orange', linestyle='dashed')

plt.legend(['Weekly Avg'], prop={'size': 15})

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
## Cannot use Area Plot with -ve values, so let's go with Line Plot
plt.figure(figsize=(13, 9))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['India']))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['Italy']))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['US']))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['Russia']))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['Spain']))

sns.lineplot(days[:,0], daily_values(con_wise_confirm_cases['Brazil']))



plt.title('# of Confirmed Coronavirus Cases', size=25)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.legend(['India', 'Italy', 'US', 'Russia', 'Spain','Brazil'], prop={'size': 12})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()

plt.figure(figsize=(13, 7))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['India']))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['Italy']))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['US']))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['Russia']))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['Spain']))

sns.lineplot(days[:,0], daily_values(con_wise_recovered_cases['Brazil']))



plt.title('# of Recovered Coronavirus Cases', size=25)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.legend(['India', 'Italy', 'US', 'Russia', 'Spain','Brazil'], prop={'size': 12})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
plt.figure(figsize=(13, 7))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['India']))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['Italy']))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['US']))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['Russia']))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['Spain']))

sns.lineplot(days[:,0], daily_values(con_wise_death_cases['Brazil']))



plt.title('# of Coronavirus Deaths', size=25)

plt.xlabel('Days Since 27/09/2020', size=15)

plt.ylabel('# of Cases', size=15)

plt.legend(['India', 'Italy', 'US', 'Russia', 'Spain','Brazil'], prop={'size': 12})

plt.xticks(size=15)

plt.yticks(size=15)

plt.show()
latest_data_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports_us/09-26-2020.csv')

latest_data_df.head() # Latest data has data of only 1 country, USA
plt.figure(figsize=(10,13))

sns.barplot(x = latest_data_df['Confirmed'],y=latest_data_df['Province_State'])

plt.title('USA State wise cases', size=20)

plt.xlabel('# of case', size=15)

plt.ylabel('States', size=15)

plt.show()
x = latest_data_df['Confirmed'].sort_values(ascending=False)[:10]

y = []

for i in x.index:

    y.append(latest_data_df['Province_State'][i])

    

x = list(x)

temp = sum(latest_data_df['Confirmed'].sort_values(ascending=False)[10:])

x.append(temp)

y.append('Others')
colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0,0,0,0,0,0.1]



# visual

plt.figure(figsize = (10,10))

plt.pie(x, labels=y, explode = explode,colors=colors, autopct='%1.1f%%')

plt.title('% of cases based on US states',color = 'blue',fontsize = 15)
total_cases_X = np.array(daily_values(total_cases)).reshape(-1, 1)

total_deaths_X = np.array(daily_values(total_deaths)).reshape(-1, 1)

total_recoveries_X = np.array(daily_values(total_recoveries)).reshape(-1, 1)
X_train_confirmed, X_test_confirmed, y_train_confirmed, y_test_confirmed = train_test_split(days, total_cases_X, test_size=0.11, shuffle=False)

print("size of X_train = {} \nsize of X_test = {}".format(len(X_train_confirmed),len(X_test_confirmed)))
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV



import xgboost as xgb



xgb_model = xgb.XGBRegressor(objective="reg:linear")

xgb_model.fit(X_train_confirmed, y_train_confirmed)
xg_pred = xgb_model.predict(X_test_confirmed)

print(mean_absolute_error(y_test_confirmed, xg_pred))
#Try Linear Regression



import numpy as np

from sklearn.linear_model import LinearRegression

Lreg = LinearRegression(normalize=True).fit(X_train_confirmed, y_train_confirmed)



# check against testing data

m_pred = Lreg.predict(X_test_confirmed)

plt.plot(y_test_confirmed)

plt.plot(m_pred)

plt.legend(['Test Data', 'Linear Predictions'])

print('MAE:', mean_absolute_error(xg_pred, y_test_confirmed))

print('MSE:',mean_squared_error(xg_pred, y_test_confirmed))
#Try Ridge Regression



from sklearn.linear_model import Ridge



reg = Ridge(alpha=1.0).fit(X_train_confirmed, y_train_confirmed)



# check against testing data

m_pred = reg.predict(X_test_confirmed)

plt.plot(y_test_confirmed)

plt.plot(m_pred)

plt.legend(['Test Data', 'Ridge Predictions'])

print('MAE:', mean_absolute_error(m_pred, y_test_confirmed))

print('MSE:',mean_squared_error(m_pred, y_test_confirmed))
# Let's create a formatted date column to do prediction

start = '1/22/2020'

start_date = datetime.datetime.strptime(start, '%m/%d/%Y')

future_forcast_dates = []

for i in range(len(days)):

    future_forcast_dates.append((start_date + datetime.timedelta(days=i)).strftime('%Y-%m-%d'))

future_forcast_dates
time_df = pd.DataFrame({'date':future_forcast_dates,'Confirmed':daily_values(total_cases)})

time_df = time_df.sort_values('date')

time_df = time_df.groupby('date')['Confirmed'].sum().reset_index()

time_df = time_df.set_index('date')

time_df.index = pd.to_datetime(time_df.index)

time_df
temp = time_df['Confirmed'].resample('D').mean()

temp
temp.plot(figsize=(15, 6))

plt.show()
import warnings

warnings.filterwarnings("ignore")

import statsmodels.api as sm

from pylab import rcParams

import itertools



rcParams['figure.figsize'] = 16, 12



# LEt's use statsmodel api to get some visualization and pre-built models



decomposition = sm.tsa.seasonal_decompose(temp, model='additive')

fig = decomposition.plot()

plt.show()
import sklearn.metrics as metrics



# Let's define a function to get all the metrics



def regression_results(y_true, y_pred):# Regression metrics

    

    explained_variance = metrics.explained_variance_score(y_true, y_pred)

    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred) 

    mse = metrics.mean_squared_error(y_true, y_pred) 

    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)

    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)

    r2 = metrics.r2_score(y_true, y_pred)

    

    print('explained_variance: ', round(explained_variance,4))    

    print('mean_squared_log_error: ', round(mean_squared_log_error,4))

    print('r2: ', round(r2,4))

    print('MAE: ', round(mean_absolute_error,4))

    print('MSE: ', round(mse,4))

    print('RMSE: ', round(np.sqrt(mse),4))
# we will ad a yesterday column basedon which we will predict Confirmed cases

time_df.loc[:,'Yesterday'] = time_df.loc[:,'Confirmed'].shift()# inserting another column with day before yesterday's values.

time_df = time_df.dropna()

time_df
X_train_confirmed = time_df[:'2020-06'].drop(['Confirmed'], axis = 1)

y_train_confirmed = time_df.loc[:'2020-06', 'Confirmed']



X_test_confirmed = time_df['2020-07':].drop(['Confirmed'], axis = 1)

y_test_confirmed = time_df.loc['2020-07':, 'Confirmed']



X_train_confirmed , y_train_confirmed
from sklearn.model_selection import TimeSeriesSplit

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor

rcParams['figure.figsize'] = 17, 9

    

models = []

models.append(('LR', LinearRegression()))

models.append(('KNN', KNeighborsRegressor())) 

models.append(('RF', RandomForestRegressor(n_estimators = 10))) # Ensemble method - collection of many decision trees

models.append(('SVR', SVR(gamma='auto'))) # kernel = linear# Evaluate each model in turn

results = []

names = []



for name, model in models:

    # TimeSeries Cross validation

    tscv = TimeSeriesSplit(n_splits=22)

    

    cv_results = cross_val_score(model, X_train_confirmed, y_train_confirmed, cv=tscv, scoring='r2')

    results.append(cv_results)

    names.append(name)

    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

    

# Compare Algorithms

plt.boxplot(results, labels=names)

plt.title('Algorithm Comparison')

plt.show()
from sklearn.model_selection import GridSearchCV



model = RandomForestRegressor()



from sklearn.metrics import make_scorer



def rmse(actual, predict):

    predict = np.array(predict)

    actual = np.array(actual)

    distance = predict - actual

    square_distance = distance ** 2

    mean_square_distance = square_distance.mean()

    score = np.sqrt(mean_square_distance)

    return score



rmse_score = make_scorer(rmse, greater_is_better = False)



param_search = { 

    'n_estimators': [25, 50, 100],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [i for i in range(5,15)]

}

tscv = TimeSeriesSplit(n_splits=5)

gsearch = GridSearchCV(estimator=model, cv=tscv, param_grid = param_search, scoring = rmse_score)

gsearch.fit(X_train_confirmed, y_train_confirmed)

best_score = gsearch.best_score_

best_model = gsearch.best_estimator_
best_model
y_true = y_test_confirmed.values

y_pred = best_model.predict(X_test_confirmed)



regression_results(y_true, y_pred)
plt.figure(figsize=(10,7))

plt.plot(y_true)

plt.plot(y_pred)

plt.legend(['Test Data', 'Random Forest'])

print('MAE:', mean_absolute_error(y_pred, y_test_confirmed))

print('MSE:',mean_squared_error(y_pred, y_test_confirmed))