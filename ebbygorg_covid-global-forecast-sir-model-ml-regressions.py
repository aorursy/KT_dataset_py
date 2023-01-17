import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

display(train.head(5))

display(train.describe())

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train[train['Province_State'].isna()==False]['Country_Region'].unique())
#confirmed_country = train.groupby(['Country/Region', 'Province/State']).agg({'ConfirmedCases':['sum']})

#fatalities_country = train.groupby(['Country/Region', 'Province/State']).agg({'Fatalities':['sum']})

confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Global confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
#confirmed_country_noChina = train[train['Country_Region']!='China'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_noChina = train[train['Country_Region']!='China'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_noChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
#confirmed_country_China = train[train['Country_Region']=='China'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_China = train[train['Country_Region']=='China'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_China = train[train['Country_Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_China = confirmed_total_date_China.join(fatalities_total_date_China)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_China.plot(ax=ax1)

ax1.set_title("China confirmed cases", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_China.plot(ax=ax2, color='orange')

ax2.set_title("China deceased cases", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
#confirmed_country_Italy = train[train['Country_Region']=='Italy'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Italy = train[train['Country_Region']=='Italy'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[train['Country_Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



#confirmed_country_Spain = train[train['Country_Region']=='Spain'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Spain = train[train['Country_Region']=='Spain'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Spain = train[train['Country_Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)



#confirmed_country_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_UK = train[train['Country_Region']=='United Kingdom'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_UK = confirmed_total_date_UK.join(fatalities_total_date_UK)



#confirmed_country_Australia = train[train['Country_Region']=='Australia'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Australia = train[train['Country_Region']=='Australia'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Australia = train[train['Country_Region']=='Australia'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Australia = confirmed_total_date_Australia.join(fatalities_total_date_Australia)



#confirmed_country_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Singapore = train[train['Country_Region']=='Singapore'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Singapore = confirmed_total_date_Singapore.join(fatalities_total_date_Singapore)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Italy.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 2)

total_date_Spain.plot(ax=plt.gca(), title='Spain')



plt.subplot(2, 2, 3)

total_date_UK.plot(ax=plt.gca(), title='United Kingdom')

plt.ylabel("Confirmed infection cases", size=13)



plt.subplot(2, 2, 4)

total_date_Singapore.plot(ax=plt.gca(), title='Singapore')
pop_italy = 60486683.

pop_spain = 46749696.

pop_UK = 67784927.

pop_singapore = 5837230.



total_date_Italy.ConfirmedCases = total_date_Italy.ConfirmedCases/pop_italy*100.

total_date_Italy.Fatalities = total_date_Italy.ConfirmedCases/pop_italy*100.

total_date_Spain.ConfirmedCases = total_date_Spain.ConfirmedCases/pop_spain*100.

total_date_Spain.Fatalities = total_date_Spain.ConfirmedCases/pop_spain*100.

total_date_UK.ConfirmedCases = total_date_UK.ConfirmedCases/pop_UK*100.

total_date_UK.Fatalities = total_date_UK.ConfirmedCases/pop_UK*100.

total_date_Singapore.ConfirmedCases = total_date_Singapore.ConfirmedCases/pop_singapore*100.

total_date_Singapore.Fatalities = total_date_Singapore.ConfirmedCases/pop_singapore*100.



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_date_Italy.ConfirmedCases.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.06)



plt.subplot(2, 2, 2)

total_date_Spain.ConfirmedCases.plot(ax=plt.gca(), title='Spain')

plt.ylim(0, 0.06)



plt.subplot(2, 2, 3)

total_date_UK.ConfirmedCases.plot(ax=plt.gca(), title='United Kingdom')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.005)



plt.subplot(2, 2, 4)

total_date_Singapore.ConfirmedCases.plot(ax=plt.gca(), title='Singapore')

plt.ylim(0, 0.005)
#confirmed_country_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Italy = train[(train['Country_Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Italy = confirmed_total_date_Italy.join(fatalities_total_date_Italy)



#confirmed_country_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Spain = train[(train['Country_Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Spain = confirmed_total_date_Spain.join(fatalities_total_date_Spain)



#confirmed_country_UK = train[(train['Country_Region']=='United Kingdom') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_UK = train[(train['Country_Region']=='United Kingdom') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_UK = train[(train['Country_Region']=='United Kingdom') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_UK = train[(train['Country_Region']=='United Kingdom') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_UK = confirmed_total_date_UK.join(fatalities_total_date_UK)



#confirmed_country_Australia = train[(train['Country_Region']=='Australia') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Australia = train[(train['Country_Region']=='Australia') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Australia = train[(train['Country_Region']=='Australia') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Australia = train[(train['Country_Region']=='Australia') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Australia = confirmed_total_date_Australia.join(fatalities_total_date_Australia)



#confirmed_country_Singapore = train[(train['Country_Region']=='Singapore') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

#fatalities_country_Singapore = train[(train['Country_Region']=='Singapore') & (train['ConfirmedCases']!=0)].groupby(['Country_Region', 'Province_State']).agg({'Fatalities':['sum']})

confirmed_total_date_Singapore = train[(train['Country_Region']=='Singapore') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_Singapore = train[(train['Country_Region']=='Singapore') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_Singapore = confirmed_total_date_Singapore.join(fatalities_total_date_Singapore)



italy = [i for i in total_date_Italy.ConfirmedCases['sum'].values]

italy_30 = italy[0:50] 

spain = [i for i in total_date_Spain.ConfirmedCases['sum'].values]

spain_30 = spain[0:50] 

UK = [i for i in total_date_UK.ConfirmedCases['sum'].values]

UK_30 = UK[0:50] 

singapore = [i for i in total_date_Singapore.ConfirmedCases['sum'].values]

singapore_30 = singapore[0:50] 





# Plots

plt.figure(figsize=(12,6))

plt.plot(italy_30)

plt.plot(spain_30)

plt.plot(UK_30)

plt.plot(singapore_30)

plt.legend(["Italy", "Spain", "UK", "Singapore"], loc='upper left')

plt.title("COVID-19 infections from the first confirmed case", size=15)

plt.xlabel("Days", size=13)

plt.ylabel("Infected cases", size=13)

plt.ylim(0, 60000)

plt.show()
# Susceptible equation

def fa(N, a, b, beta):

    fa = -beta*a*b

    return fa



# Infected equation

def fb(N, a, b, beta, gamma):

    fb = beta*a*b - gamma*b

    return fb



# Recovered/deceased equation

def fc(N, b, gamma):

    fc = gamma*b

    return fc
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible a, infected b and recovered r)

def rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs):

    a1 = fa(N, a, b, beta)*hs

    b1 = fb(N, a, b, beta, gamma)*hs

    c1 = fc(N, b, gamma)*hs

    ak = a + a1*0.5

    bk = b + b1*0.5

    ck = c + c1*0.5

    a2 = fa(N, ak, bk, beta)*hs

    b2 = fb(N, ak, bk, beta, gamma)*hs

    c2 = fc(N, bk, gamma)*hs

    ak = a + a2*0.5

    bk = b + b2*0.5

    ck = c + c2*0.5

    a3 = fa(N, ak, bk, beta)*hs

    b3 = fb(N, ak, bk, beta, gamma)*hs

    c3 = fc(N, bk, gamma)*hs

    ak = a + a3

    bk = b + b3

    ck = c + c3

    a4 = fa(N, ak, bk, beta)*hs

    b4 = fb(N, ak, bk, beta, gamma)*hs

    c4 = fc(N, bk, gamma)*hs

    a = a + (a1 + 2*(a2 + a3) + a4)/6

    b = b + (b1 + 2*(b2 + b3) + b4)/6

    c = c + (c1 + 2*(c2 + c3) + c4)/6

    return a, b, c
def SIR(N, b0, beta, gamma, hs):

    

    """

    N = total number of population

    beta = transition rate S->I

    gamma = transition rate I->R

    k =  denotes the constant degree distribution of the network (average value for networks in which 

    the probability of finding a node with a different connectivity decays exponentially fast

    hs = jump step of the numerical integration

    """

    

    # Initial condition

    a = float(N-1)/N -b0

    b = float(1)/N +b0

    c = 0.



    sus, inf, rec= [],[],[]

    for i in range(10000): # Run for a certain number of time-steps

        sus.append(a)

        inf.append(b)

        rec.append(c)

        a,b,c = rK4(N, a, b, c, fa, fb, fc, beta, gamma, hs)



    return sus, inf, rec
# Parameters of the model

N = 7800*(10**6)

b0 = 0

beta = 0.7

gamma = 0.2

hs = 0.1



sus, inf, rec = SIR(N, b0, beta, gamma, hs)



f = plt.figure(figsize=(8,5)) 

plt.plot(sus, 'b.', label='susceptible');

plt.plot(inf, 'r.', label='infected');

plt.plot(rec, 'c.', label='recovered/deceased');

plt.title("SIR model")

plt.xlabel("time", fontsize=10);

plt.ylabel("Fraction of population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.savefig('SIR_example.png')

plt.show()
population = float(46750238)

country_df = pd.DataFrame()

country_df['ConfirmedCases'] = train.loc[train['Country_Region']=='Spain'].ConfirmedCases.diff().fillna(0)

country_df = country_df[10:]

country_df['day_count'] = list(range(1,len(country_df)+1))



ydata = [i for i in country_df.ConfirmedCases]

xdata = country_df.day_count

ydata = np.array(ydata, dtype=float)

xdata = np.array(xdata, dtype=float)



N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0



def sir_model(y, x, beta, gamma):

    sus = -beta * y[0] * y[1] / N

    rec = gamma * y[1]

    inf = -(sus + rec)

    return sus, inf, rec



def fit_odeint(x, beta, gamma):

    return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]



popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)

fitted = fit_odeint(xdata, *popt)



plt.plot(xdata, ydata, 'o')

plt.plot(xdata, fitted)

plt.title("Fit of SIR model for Spain infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
# Merge train and test, exclude overlap

dates_overlap = ['2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', 

                 '2020-03-26', '2020-03-27', '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']

train2 = train.loc[~train['Date'].isin(dates_overlap)]

all_data = pd.concat([train2, test], axis = 0, sort=False)



# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11

all_data.loc[all_data['Date'] >= '2020-03-19', 'ConfirmedCases'] = np.nan

all_data.loc[all_data['Date'] >= '2020-03-19', 'Fatalities'] = np.nan

all_data['Date'] = pd.to_datetime(all_data['Date'])



# Create date columns

le = preprocessing.LabelEncoder()

all_data['Day_num'] = le.fit_transform(all_data.Date)

all_data['Day'] = all_data['Date'].dt.day

all_data['Month'] = all_data['Date'].dt.month

all_data['Year'] = all_data['Date'].dt.year



# Fill null values given that we merged train-test datasets

all_data['Province_State'].fillna("None", inplace=True)

all_data['ConfirmedCases'].fillna(0, inplace=True)

all_data['Fatalities'].fillna(0, inplace=True)

all_data['Id'].fillna(-1, inplace=True)

all_data['ForecastId'].fillna(-1, inplace=True)



display(all_data)

display(all_data.loc[all_data['Date'] == '2020-03-19'])
missings_count = {col:all_data[col].isnull().sum() for col in all_data.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
def calculate_trend(df, lag_list, column):

    for lag in lag_list:

        trend_column_lag = "Trend_" + column + "_" + str(lag)

        df[trend_column_lag] = (df[column]-df[column].shift(lag, fill_value=-999))/df[column].shift(lag, fill_value=0)

    return df





def calculate_lag(df, lag_list, column):

    for lag in lag_list:

        column_lag = column + "_" + str(lag)

        df[column_lag] = df[column].shift(lag, fill_value=0)

    return df





ts = time.time()

all_data = calculate_lag(all_data, range(1,7), 'ConfirmedCases')

all_data = calculate_lag(all_data, range(1,7), 'Fatalities')

all_data = calculate_trend(all_data, range(1,7), 'ConfirmedCases')

all_data = calculate_trend(all_data, range(1,7), 'Fatalities')

all_data.replace([np.inf, -np.inf], 0, inplace=True)

all_data.fillna(0, inplace=True)

print("Time spent: ", time.time()-ts)
all_data[all_data['Country_Region']=='Spain'].iloc[40:50][['Id', 'Province_State', 'Country_Region', 'Date',

       'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num', 'ConfirmedCases_1',

       'ConfirmedCases_2', 'ConfirmedCases_3', 'Fatalities_1', 'Fatalities_2',

       'Fatalities_3']]
# Load countries data file

world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")



# Select desired columns and rename some of them

world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

world_population.columns = ['Country (or dependency)', 'Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']



# Replace United States by US

world_population.loc[world_population['Country (or dependency)']=='United States', 'Country (or dependency)'] = 'US'



# Remove the % character from Urban Pop values

world_population['Urban Pop'] = world_population['Urban Pop'].str.rstrip('%')



# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int

world_population.loc[world_population['Urban Pop']=='N.A.', 'Urban Pop'] = int(world_population.loc[world_population['Urban Pop']!='N.A.', 'Urban Pop'].mode()[0])

world_population['Urban Pop'] = world_population['Urban Pop'].astype('int16')

world_population.loc[world_population['Med Age']=='N.A.', 'Med Age'] = int(world_population.loc[world_population['Med Age']!='N.A.', 'Med Age'].mode()[0])

world_population['Med Age'] = world_population['Med Age'].astype('int16')



print("Cleaned country details dataset")

display(world_population)



# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities

print("Joined dataset")

all_data = all_data.merge(world_population, left_on='Country_Region', right_on='Country (or dependency)', how='left')

all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)

display(all_data)



print("Encoded dataset")

# Label encode countries and provinces. Save dictionary for exploration purposes

all_data.drop('Country (or dependency)', inplace=True, axis=1)

all_data['Country_Region'] = le.fit_transform(all_data['Country_Region'])

number_c = all_data['Country_Region']

countries = le.inverse_transform(all_data['Country_Region'])

country_dict = dict(zip(countries, number_c)) 

all_data['Province_State'] = le.fit_transform(all_data['Province_State'])

number_p = all_data['Province_State']

province = le.inverse_transform(all_data['Province_State'])

province_dict = dict(zip(province, number_p)) 

display(all_data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



# Day_num = 38 is March 1st

y1 = all_data[(all_data['Country_Region']==country_dict['Spain']) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']]

x1 = range(0, len(y1))

ax1.plot(x1, y1, 'bo--')

ax1.set_title("Spain ConfirmedCases between days 39 and 49")

ax1.set_xlabel("Days")

ax1.set_ylabel("ConfirmedCases")



y2 = all_data[(all_data['Country_Region']==country_dict['Spain']) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']].apply(lambda x: np.log(x))

x2 = range(0, len(y2))

ax2.plot(x2, y2, 'bo--')

ax2.set_title("Spain Log ConfirmedCases between days 39 and 49")

ax2.set_xlabel("Days")

ax2.set_ylabel("Log ConfirmedCases")
# Filter selected features

data = all_data.copy()

features = ['Id', 'ForecastId', 'Country_Region', 'Province_State', 'ConfirmedCases', 'Fatalities', 

       'Day_num', 'Day', 'Month', 'Year']

data = data[features]



# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))



# Replace infinites

data.replace([np.inf, -np.inf], 0, inplace=True)





# Split data into train/test

def split_data(data):

    

    # Train set

    x_train = data[data.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    y_train_1 = data[data.ForecastId == -1]['ConfirmedCases']

    y_train_2 = data[data.ForecastId == -1]['Fatalities']



    # Test set

    x_test = data[data.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)



    # Clean Id columns and keep ForecastId as index

    x_train.drop('Id', inplace=True, errors='ignore', axis=1)

    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    x_test.drop('Id', inplace=True, errors='ignore', axis=1)

    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    

    return x_train, y_train_1, y_train_2, x_test





# Linear regression model

def lin_reg(X_train, Y_train, X_test):

    # Create linear regression object

    regr = linear_model.LinearRegression()



    # Train the model using the training sets

    regr.fit(X_train, Y_train)



    # Make predictions using the testing set

    y_pred = regr.predict(X_test)

    

    return regr, y_pred





# Submission function

def get_submission(df, target1, target2):

    

    prediction_1 = df[target1]

    prediction_2 = df[target2]



    # Submit predictions

    prediction_1 = [int(item) for item in list(map(round, prediction_1))]

    prediction_2 = [int(item) for item in list(map(round, prediction_2))]

    

    submission = pd.DataFrame({

        "ForecastId": df['ForecastId'].astype('int32'), 

        "ConfirmedCases": prediction_1, 

        "Fatalities": prediction_2

    })

    submission.to_csv('submission.csv', index=False)
# Select train (real) data from March 1 to March 22nd

dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', 

                 '2020-03-10', '2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',

                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', 

                 '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31']
all_data.loc[all_data['Country_Region']==country_dict['Spain']][40:65]
def plot_linreg_basic_country(data, country_name, dates_list, day_start, shift):

    

    data_country = data[data['Country_Region']==country_dict[country_name]]

    data_country = data_country.loc[data_country['Day_num']>=day_start]

    X_train, Y_train_1, Y_train_2, X_test = split_data(data_country)

    model, pred = lin_reg(X_train, Y_train_1, X_test)



    # Create a df with both real cases and predictions (predictions starting on March 12th)

    X_train_check = X_train.copy()

    X_train_check['Target'] = Y_train_1



    X_test_check = X_test.copy()

    X_test_check['Target'] = pred



    X_final_check = pd.concat([X_train_check, X_test_check])



    # Select predictions from March 1st to March 25th

    predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(dates_list_num, np.expm1(predicted_data))

    ax1.plot(dates_list_num, real_data)

    ax1.axvline(17-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (from March " + str(1+shift) + " to March 25th)")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(17-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (from March " + str(1+shift) + " to March 30th)")

    ax2.set_ylabel("Log Confirmed Cases")



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))

    

    

    

# Filter Spain, run the Linear Regression workflow

country_name = "Spain"

march_day = 0

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Spain, run the Linear Regression workflow

country_name = "Spain"

march_day = 15

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Italy, run the Linear Regression workflow

country_name = "Italy"

march_day = 0

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Italy, run the Linear Regression workflow

country_name = "Italy"

march_day = 15

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Germany, run the Linear Regression workflow

country_name = "Germany"

march_day = 0

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Germany, run the Linear Regression workflow

country_name = "Germany"

march_day = 15

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Albania, run the Linear Regression workflow

country_name = "Albania"

march_day = 0

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Albania, run the Linear Regression workflow

country_name = "Albania"

march_day = 15

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day)
# Filter Andorra, run the Linear Regression workflow

country_name = "Andorra"

shift = 0

day_start = 39+shift

dates_list2 = dates_list[shift:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, shift)
# Filter Andorra, run the Linear Regression workflow

country_name = "Andorra"

shift = 7

day_start = 39+shift

dates_list2 = dates_list[shift:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, shift)
ts = time.time()



def linreg_basic_all_countries(data, day_start):

    

    data2 = data.loc[data.Day_num >= day_start]



    # Set the dataframe where we will update the predictions

    data_pred = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]

    data_pred = data_pred.loc[data_pred['Day_num']>=day_start]

    data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)

    data_pred['Predicted_Fatalities'] = [0]*len(data_pred)



    print("Currently running Logistic Regression for all countries")



    # Main loop for countries

    for c in data2['Country_Region'].unique():



        # List of provinces

        provinces_list = data2[data2['Country_Region']==c]['Province_State'].unique()



        # If the country has several Province/State informed

        if len(provinces_list)>1:

            for p in provinces_list:

                data_cp = data2[(data2['Country_Region']==c) & (data2['Province_State']==p)]

                X_train, Y_train_1, Y_train_2, X_test = split_data(data_cp)

                model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

                model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

                data_pred.loc[((data_pred['Country_Region']==c) & (data2['Province_State']==p)), 'Predicted_ConfirmedCases'] = pred_1

                data_pred.loc[((data_pred['Country_Region']==c) & (data2['Province_State']==p)), 'Predicted_Fatalities'] = pred_2



        # No Province/State informed

        else:

            data_c = data2[(data2['Country_Region']==c)]

            X_train, Y_train_1, Y_train_2, X_test = split_data(data_c)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred.loc[(data_pred['Country_Region']==c), 'Predicted_ConfirmedCases'] = pred_1

            data_pred.loc[(data_pred['Country_Region']==c), 'Predicted_Fatalities'] = pred_2



    # Aplly exponential transf. and clean potential infinites due to final numerical precision

    data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))

    data_pred.replace([np.inf, -np.inf], 0, inplace=True) 

    

    return data_pred





day_start = 39

data_pred = linreg_basic_all_countries(data, day_start)

#get_submission(data_pred, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')



print("Process finished in ", round(time.time() - ts, 2), " seconds")
ts = time.time()



# Set the dataframe where we will update the predictions

data2 = data.loc[data.Day_num >= day_start]

data_pred3 = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]

data_pred3['Predicted_ConfirmedCases'] = [0]*len(data_pred3)

data_pred3['Predicted_Fatalities'] = [0]*len(data_pred3)

how_many_days = test.Date.nunique()

    

print("Currently running Logistic Regression for all countries")



# Main loop for countries

for c in data['Country_Region'].unique():

    

    # List of provinces

    provinces_list = data2[data2['Country_Region']==c]['Province_State'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        

        for p in provinces_list:

            # Only fit starting from the first confirmed case in the country

            train_countries_no0 = data.loc[(data['Country_Region']==c) & (data['Province_State']==p) & (data.ConfirmedCases!=0) & (data.ForecastId==-1)]

            test_countries_no0 = data.loc[(data['Country_Region']==c) & (data['Province_State']==p) &  (data.ForecastId!=-1)]

            data2 = pd.concat([train_countries_no0, test_countries_no0])



            # If there are no previous cases, predict 0

            if len(train_countries_no0) == 0:

                data_pred3.loc[((data_pred2['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

                data_pred3.loc[((data_pred2['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_Fatalities'] = [0]*how_many_days

                

            # Else run LinReg

            else: 

                data_cp = data2[(data2['Country_Region']==c) & (data2['Province_State']==p)]

                X_train, Y_train_1, Y_train_2, X_test = split_data(data_cp)

                model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

                model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

                data_pred3.loc[((data_pred3['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_ConfirmedCases'] = pred_1

                data_pred3.loc[((data_pred3['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_Fatalities'] = pred_2



    # No Province/State informed

    else:

        # Only fit starting from the first confirmed case in the country

        train_countries_no0 = data.loc[(data['Country_Region']==c) & (data.ConfirmedCases!=0) & (data.ForecastId==-1)]

        test_countries_no0 = data.loc[(data['Country_Region']==c) &  (data.ForecastId!=-1)]

        data2 = pd.concat([train_countries_no0, test_countries_no0])



        # If there are no previous cases, predict 0

        if len(train_countries_no0) == 0:

            data_pred3.loc[((data_pred3['Country_Region']==c)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

            data_pred3.loc[((data_pred3['Country_Region']==c)), 'Predicted_Fatalities'] = [0]*how_many_days

        

        # Else, run LinReg

        else:

            data_c = data2[(data2['Country_Region']==c)]

            X_train, Y_train_1, Y_train_2, X_test = split_data(data_c)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred3.loc[(data_pred3['Country_Region']==c), 'Predicted_ConfirmedCases'] = pred_1

            data_pred3.loc[(data_pred3['Country_Region']==c), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))

data_pred3.replace([np.inf, -np.inf], 0, inplace=True) 



#get_submission(data_pred3, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')



print("Process finished in ", round(time.time() - ts, 2), " seconds")
# New split function, for one forecast day

def split_data_one_day(data, d):

    

    #Train

    x_train = data[data.Day_num<d]

    y_train_1 = x_train.ConfirmedCases

    y_train_2 = x_train.Fatalities

    x_train.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    

    #Test

    x_test = data[data.Day_num==d]

    x_test.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    

    # Clean Id columns and keep ForecastId as index

    x_train.drop('Id', inplace=True, errors='ignore', axis=1)

    x_train.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    x_test.drop('Id', inplace=True, errors='ignore', axis=1)

    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    

    return x_train, y_train_1, y_train_2, x_test





def plot_real_vs_prediction_country(data, train, country_name, day_start, dates_list, march_day):



    # Select predictions from March 1st to March 25th

    predicted_data = data.loc[(data['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].ConfirmedCases

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(dates_list_num, np.expm1(predicted_data))

    ax1.plot(dates_list_num, real_data)

    ax1.axvline(17-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March 1st)")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(17-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (starting on March 1st)")

    ax2.set_ylabel("Log Confirmed Cases")



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))

    

    

def plot_real_vs_prediction_country_fatalities(data, train, country_name, day_start, dates_list, march_day):



    # Select predictions from March 1st to March 25th

    predicted_data = data.loc[(data['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Fatalities

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['Fatalities']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(dates_list_num, np.expm1(predicted_data))

    ax1.plot(dates_list_num, real_data)

    ax1.axvline(17-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March 1st)")

    ax1.set_ylabel("Fatalities Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(17-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (starting on March 1st)")

    ax2.set_ylabel("Log Fatalities Cases")



    plt.suptitle(("Fatalities predictions based on Log-Lineal Regression for "+country_name))
# Function to compute the Linear Regression predictions with lags, for a certain Country/Region

def lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict):

    

    ts = time.time()

    

    # Filter country and features from all_data (dataset without data leaking)

    data = all_data.copy()

    features = ['Id', 'Province_State', 'Country_Region',

           'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num']

    data = data[features]



    # Select country an data start (all days)

    data = data[data['Country_Region']==country_dict[country_name]]

    data = data.loc[data['Day_num']>=day_start]



    # Lags

    data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

    data = calculate_lag(data, range(1,8), 'Fatalities')



    filter_col_confirmed = [col for col in data if col.startswith('Confirmed')]

    filter_col_fatalities= [col for col in data if col.startswith('Fataliti')]

    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)

    

    # Apply log transformation

    data[filter_col] = data[filter_col].apply(lambda x: np.log1p(x))

    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.fillna(0, inplace=True)





    # Start/end of forecast

    start_fcst = all_data[all_data['Id']==-1].Day_num.min()

    end_fcst = all_data[all_data['Id']==-1].Day_num.max()



    for d in list(range(start_fcst, end_fcst+1)):

        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d)

        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) 

                 & (data['Day_num']==d), 'ConfirmedCases'] = pred_1[0]

        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) 

                 & (data['Day_num']==d), 'Fatalities'] = pred_2[0]



        # Recompute lags 

        data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

        data = calculate_lag(data, range(1,8), 'Fatalities')

        data.replace([np.inf, -np.inf], 0, inplace=True)

        data.fillna(0, inplace=True)



    #print("Process for ", country_name, "finished in ", round(time.time() - ts, 2), " seconds")

    

    return data





# Function to compute the Linear Regression predictions with lags, for a certain Country/Region and State/province

def lin_reg_with_lags_country_province(all_data, country_name, province_name, day_start, lag_size, country_dict):

    

    ts = time.time()

    

    # Filter country and features from all_data (dataset without data leaking)

    data = all_data.copy()

    features = ['Id', 'Province_State', 'Country_Region',

           'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num']

    data = data[features]



    # Select country an data start (all days)

    data = data[(data['Country_Region']==country_dict[country_name]) & (data['Province_State']==province_dict[province_name])]

    data = data.loc[data['Day_num']>=day_start]



    # Lags

    data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

    data = calculate_lag(data, range(1,lag_size), 'Fatalities')



    # Apply log transformation

    filter_col_confirmed = [col for col in data if col.startswith('Confirmed')]

    filter_col_fatalities= [col for col in data if col.startswith('Fataliti')]

    filter_col = np.append(filter_col_confirmed, filter_col_fatalities)

    data[filter_col] = data[filter_col].apply(lambda x: np.log1p(x))

    data.replace([np.inf, -np.inf], 0, inplace=True)

    data.fillna(0, inplace=True)



    # Start/end of forecast

    start_fcst = all_data[all_data['Id']==-1].Day_num.min()

    end_fcst = all_data[all_data['Id']==-1].Day_num.max()



    for d in list(range(start_fcst, end_fcst+1)):

        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d)

        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) & (data['Province_State']==province_dict[province_name]) 

                 & (data['Day_num']==d), 'ConfirmedCases'] = pred_1[0]

        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

        data.loc[(data['Country_Region']==country_dict[country_name]) & (data['Province_State']==province_dict[province_name])

                 & (data['Day_num']==d), 'Fatalities'] = pred_2[0]



        # Recompute lags 

        data = calculate_lag(data, range(1,lag_size), 'ConfirmedCases')

        data = calculate_lag(data, range(1,lag_size), 'Fatalities')

        data.replace([np.inf, -np.inf], 0, inplace=True)

        data.fillna(0, inplace=True)



    #print("Process for ", country_name, "/", province_name, "finished in ", round(time.time() - ts, 2), " seconds")

    

    return data







# Run the model for Spain

country_name = 'Spain'

march_day = 0

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 30



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
ts = time.time()



# Inputs

country_name = "Italy"

march_day = 0

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 30



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
# Inputs

country_name = "Germany"

march_day = 0

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 30



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
# Inputs

country_name = "Albania"

march_day = 10

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 7



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
# Inputs

country_name = "Andorra"

march_day = 5

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 1



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
# Inputs

day_start = 39 

lag_size = 30



results_df = pd.DataFrame()



tp = time.time()



# Main loop for countries

for country_name in train['Country_Region'].unique():



    # List of provinces

    provinces_list = all_data[all_data['Country_Region']==country_name]['Province_State'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        for province_name in provinces_list:

            pred_province = lin_reg_with_lags_country_province(all_data, country_name, province_name, day_start, lag_size, country_dict)

            results_df = pd.concat([results_df, pred_province])



    else:

        pred_country = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

        results_df = pd.concat([results_df, pred_country])

        

results_df_submit = results_df.copy()

results_df_submit['ConfirmedCases'] = results_df_submit['ConfirmedCases'].apply(lambda x: np.expm1(x))

results_df_submit['Fatalities'] = results_df_submit['Fatalities'].apply(lambda x: np.expm1(x))

        

get_submission(results_df_submit.loc[results_df_submit['ForecastId']!=-1], 'ConfirmedCases', 'Fatalities')

print("Complete process finished in ", time.time()-tp)
"""

results_df_2 = results_df.copy()



day_start = 50

data_pred2 = linreg_basic_all_countries(data, day_start)



day_num_test = 57



# Main loop for countries

for country_name in train['Country_Region'].unique():



    # List of provinces

    provinces_list = all_data[all_data['Country_Region']==country_name]['Province_State'].unique()

        

    # Countries with several Province_State informed

    if len(provinces_list)>1:

        for province_name in provinces_list:

            tmp_index = results_df_2.index[(results_df_2['Country_Region']==country_dict[country_name]) & 

                           (results_df_2['Province_State']==province_dict[province_name]) & 

                           (results_df_2['Day_num']<day_num_test) & 

                           (results_df_2['ConfirmedCases']!=0)]



            # When there is not enough data

            if len(tmp_index) < 10:

                

                # ConfirmedCases

                results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                                  (results_df_2['Province_State']==province_dict[province_name]) &

                                  (results_df_2['Day_num']>=day_num_test)), 'ConfirmedCases'] = data_pred2.loc[((data_pred2['Country_Region']==country_dict[country_name]) & 

                                  (data_pred2['Province_State']==province_dict[province_name]) & 

                                  (data_pred2['Day_num']>=day_num_test)), 'Predicted_ConfirmedCases'].apply(lambda x: np.log1p(x))

                

                #Fatalities

                results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                                  (results_df_2['Province_State']==province_dict[province_name]) &

                                  (results_df_2['Day_num']>=day_num_test)), 'Fatalities'] = data_pred2.loc[((data_pred2['Country_Region']==country_dict[country_name]) & 

                                  (data_pred2['Province_State']==province_dict[province_name]) & 

                                  (data_pred2['Day_num']>=day_num_test)), 'Predicted_Fatalities'].apply(lambda x: np.log1p(x))

                

    # Countries without Province_State

    else:

        tmp_index = results_df_2.index[(results_df_2['Country_Region']==country_dict[country_name]) & 

                           (results_df_2['Day_num']<day_num_test) & 

                           (results_df_2['ConfirmedCases']!=0)]



        # When there is not enough data

        if len(tmp_index) < 10:

            

            #Confirmed Cases

            results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                            (results_df_2['Day_num']>=day_num_test)), 'ConfirmedCases'] = data_pred2.loc[((data_pred2['Country_Region']==country_dict[country_name]) & 

                            (data_pred2['Day_num']>=day_num_test)), 'Predicted_ConfirmedCases'].apply(lambda x: np.log1p(x))

            

            results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                            (results_df_2['Day_num']>=day_num_test)), 'Fatalities'] = data_pred2.loc[((data_pred2['Country_Region']==country_dict[country_name]) & 

                            (data_pred2['Day_num']>=day_num_test)), 'Predicted_Fatalities'].apply(lambda x: np.log1p(x))

            

results_df_2 = results_df_2.loc[results_df_2['Day_num']>=day_num_test]

results_df_2[['ConfirmedCases', 'Fatalities']] = results_df_2[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.expm1(x))

get_submission(results_df_2, 'ConfirmedCases', 'Fatalities')

"""
# ALTERNATIVE FUNCTIONS FOR FURTHER ANALYSIS



# LGB Regressor 

"""

model=lgb.LGBMRegressor(

        n_estimators=1000,

        learning_rate=0.3,

        #min_child_weight=300,

        #num_leaves=32,

        colsample_bytree=0.8,

        subsample=0.8,

        max_depth=16,

        #reg_alpha=0.04,

        #reg_lambda=0.073,

        #min_split_gain=0.0222415,

        verbose=1,

        seed=21)



model.fit(X_train, Y_train_1, eval_metric="rmse", eval_set=[(X_train, Y_train_1), (X_valid, Y_valid_1)], verbose=1, early_stopping_rounds = 20)



Y_test = model.predict(X_test)

"""





# Show Spain enriched data

"""

data[(data['Lat']==40.0) & (data['Long']==-4.0) & (data['Day']==49)]

train[(train['Country/Region']=='Spain') & (train['Date']=='2020-03-12')]

data[(data['Lat']==40.0) & (data['Long']==-4.0) & (data['Day']==49)]['ConfirmedCases'].apply(lambda x: np.exp(x))

"""





# XGB Regressor

"""

reg = xgb.XGBRegressor()

reg.fit(X_train, Y_train_1,

        eval_set=[(X_train, Y_train_1), (X_valid, Y_valid_1)],

        early_stopping_rounds=50,

       verbose=True)



_ = plot_importance(reg, height=0.9, max_num_features=20)

Y_pred = reg.predict(X_valid)

print(Y_pred)

"""





# XGB GridSearch

"""

ts = time.time()



xgb1 = xgb.XGBRegressor()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'learning_rate': [.07, .01, .005], #so called `eta` value

              'max_depth': [4, 5, 6, 7],

              'min_child_weight': [4, 5, 6, 7],

              'silent': [0],

              'subsample': [0.7],

              'colsample_bytree': [0.7],

              'n_estimators': [500]}



xgb_grid = GridSearchCV(xgb1,

                        parameters,

                        cv = 3,

                        n_jobs = 5,

                        verbose=True)



xgb_grid.fit(X_train, Y_train_1)



print(xgb_grid.best_score_)

print(xgb_grid.best_params_)



Y_pred = xgb_grid.predict(X_valid)

print(Y_pred, Y_valid_1)



print("Time spent: ", ts-time.time())

"""





# Some plots

"""

plt.plot(data[(data['Lat']==40.0) & (data['Long']==-4.0) & (data['Day']<=49)]['ConfirmedCases'])



y = data[(data['Lat']==40.0) & (data['Long']==-4.0) & (data['Day']> 35) & (data['Day']<=49)][['ConfirmedCases']].apply(lambda x: np.log(x))

x = range(0, len(y))

plt.plot(x, y, 'go--')

"""





# Linear Regression

"""

# Create linear regression object

regr = linear_model.LinearRegression()



# Train the model using the training sets

regr.fit(X_train, Y_train_1)



# Make predictions using the testing set

y_pred = regr.predict(X_test)



# Compute mean squared error on real values (inverse transform of log)

#rms = mean_squared_error(Y_test_1.apply(lambda x: np.exp(x)).astype('int16'), np.exp(y_pred).astype('int16'))

#print("Mean squared error: ", rms)

"""



# LGB 2

"""

def lgb_reg(X_train, Y_train_1, X_test):

    # Create linear regression object

    regr = lgb.LGBMRegressor(n_estimators=1000)



    # Train the model using the training sets

    regr.fit(X_train, Y_train_1, eval_metric="rmse")



    # Make predictions using the testing set

    y_pred = regr.predict(X_test)

    

    return regr, y_pred

"""





# Predictions

"""

#Predictions

X_test_compare = X_test.copy()

X_test_compare['PredictedCases'] = y_pred_xgb

X_test_compare.loc[(X_test_compare['Day']>=50) & (X_test_compare['Day']<73)]



# Train test

compare = train.loc[train['Date'].isin(dates_overlap)]

compare = compare.merge(test[['Country/Region', 'Date', 'ForecastId']], on=['Country/Region', 'Date'])

#compare['PredictedCases'] = X_test_compare.loc[(X_test_compare['Day']>=50) & (X_test_compare['Day']<73), 'PredictedCases']

compare



# Total comparison

total_compare = X_test_compare.join(compare, on='ForecastId', rsuffix='X_compare')

total_compare[['ForecastId', 'Country/Region', 'ConfirmedCases', 'PredictedCases']]

"""





# More plots for LinReg (all countries)

"""

# Select predictions from March 1st to March 22nd



predicted_data = data_pred.loc[(data_pred['Country/Region']==1)].Predicted_ConfirmedCases[:11]



dates_list = ['2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',

                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22']

real_data = train.loc[(train['Country/Region']=='Albania') & (train['Date'].isin(dates_list))]['ConfirmedCases']

dates_list_num = list(range(0,len(dates_list)))





fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



ax1.plot(dates_list_num, predicted_data)

ax1.plot(dates_list_num, real_data)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 22nd)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, np.log(predicted_data))

ax2.plot(dates_list_num, np.log(real_data))

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 22nd)")

ax2.set_ylabel("Log Confirmed Cases")

"""



# ARIMA

"""

# definir conjunto de datos

x = df_mon_to_fri



# instanciar modelo

sarima_model = SARIMAX(x, order=(2,0,1), seasonal_order=(2, 1, 0, 24))



# ajustar modelo

results = sarima_model.fit()



# mirar el AIC

results.aic

"""