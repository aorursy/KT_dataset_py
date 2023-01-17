import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

from scipy import integrate, optimize



import lightgbm as lgb

import xgboost as xgb

from xgboost import plot_importance, plot_tree

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
test = pd.read_csv("../input/covid19-global-forecasting-week-1/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-1/train.csv")

display(train.describe())

print("Number of Country: ", train['Country/Region'].nunique())

print("Dates from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province informed: ", train[train['Province/State'].isna()==False]['Country/Region'].unique())
print(test.shape)

print(train.shape)
country_stats = train.groupby(["Country/Region", "Date"])[["ConfirmedCases", "Fatalities"]].sum().reset_index()

print("# of Entries:", country_stats.shape[0])

print("# of Non-Zero Entries:", country_stats[country_stats.ConfirmedCases > 0].shape[0])

print("# of Countries:", country_stats["Country/Region"].nunique())

print("# of Countries with confirmed cases:", country_stats[country_stats.ConfirmedCases > 0]["Country/Region"].nunique())

country_stats.sample(1)
confirmed_total_date = train.groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date = train.groupby(['Date']).agg({'Fatalities':['sum']})

total_date = confirmed_total_date.join(fatalities_total_date)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date.plot(ax=ax1)

ax1.set_title("Globally confirmed cases", size=15)

ax1.set_ylabel("Number of cases", size=15)

ax1.set_xlabel("Date", size=15)

fatalities_total_date.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases", size=15)

ax2.set_ylabel("Number of cases", size=15)

ax2.set_xlabel("Date", size=15)
confirmed_total_date_woChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_woChina = train[train['Country/Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_woChina = confirmed_total_date_woChina.join(fatalities_total_date_woChina)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))

total_date_woChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=15)

ax1.set_ylabel("Number of cases", size=15)

ax1.set_xlabel("Date", size=15)

fatalities_total_date_woChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=15)

ax2.set_ylabel("Number of cases", size=15)

ax2.set_xlabel("Date", size=15)
confirmed_total_date_China = train[train['Country/Region']=='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_China = train[train['Country/Region']=='China'].groupby(['Date']).agg({'Fatalities':['sum']})

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
ppl_italy = 60486683.

ppl_spain = 46749696.



confirmed_Italy = train[train['Country/Region']=='Italy'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Italy = train[train['Country/Region']=='Italy'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Italy = confirmed_Italy.join(fatalities_Italy)

confirmed_Spain = train[train['Country/Region']=='Spain'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Spain = train[train['Country/Region']=='Spain'].groupby(['Date']).agg({'Fatalities':['sum']})

total_Spain = confirmed_Spain.join(fatalities_Spain)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_Italy.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(2, 2, 2)

total_Spain.plot(ax=plt.gca(), title='Spain')



total_Italy.ConfirmedCases = total_Italy.ConfirmedCases/ppl_italy*100.

total_Italy.Fatalities = total_Italy.ConfirmedCases/ppl_italy*100.

total_Spain.ConfirmedCases = total_Spain.ConfirmedCases/ppl_spain*100.

total_Spain.Fatalities = total_Spain.ConfirmedCases/ppl_spain*100.



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_Italy.ConfirmedCases.plot(ax=plt.gca(), title='Italy')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.06)



plt.subplot(2, 2, 2)

total_Spain.ConfirmedCases.plot(ax=plt.gca(), title='Spain')

plt.ylim(0, 0.06)
ppl_India = 1376400726.

ppl_US = 330489477.



confirmed_India = train[train['Country/Region']=='India'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_India = train[train['Country/Region']=='India'].groupby(['Date']).agg({'Fatalities':['sum']})

total_India = confirmed_India.join(fatalities_India)

confirmed_US = train[train['Country/Region']=='US'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_US = train[train['Country/Region']=='US'].groupby(['Date']).agg({'Fatalities':['sum']})

total_US = confirmed_US.join(fatalities_US)



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_India.plot(ax=plt.gca(), title='India')

plt.ylabel("Confirmed infection cases", size=15)

plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_US.plot(ax=plt.gca(), title='USA')

plt.ylabel("Confirmed infection cases", size=15)



total_India.ConfirmedCases = total_India.ConfirmedCases/ppl_India*100.

total_India.Fatalities = total_India.ConfirmedCases/ppl_India*100.

total_US.ConfirmedCases = total_US.ConfirmedCases/ppl_US*100.

total_US.Fatalities = total_US.ConfirmedCases/ppl_US*100.



plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_India.ConfirmedCases.plot(ax=plt.gca(), title='India')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.06)

plt.figure(figsize=(15,10))

plt.subplot(2, 2, 1)

total_US.ConfirmedCases.plot(ax=plt.gca(), title='USA')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.06)
confirmed_Italy = train[(train['Country/Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Italy = train[(train['Country/Region']=='Italy') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_Italy = confirmed_Italy.join(fatalities_Italy)



confirmed_Spain = train[(train['Country/Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_Spain = train[(train['Country/Region']=='Spain') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_Spain = confirmed_Spain.join(fatalities_Spain)



confirmed_India = train[(train['Country/Region']=='India') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_India = train[(train['Country/Region']=='India') & train['ConfirmedCases']!=0].groupby(['Date']).agg({'Fatalities':['sum']})

total_India = confirmed_India.join(fatalities_India)



confirmed_US = train[(train['Country/Region']=='US') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_US = train[(train['Country/Region']=='US') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_US = confirmed_US.join(fatalities_US)



confirmed_China = train[(train['Country/Region']=='China') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_China = train[(train['Country/Region']=='China') & (train['ConfirmedCases']!=0)].groupby(['Date']).agg({'Fatalities':['sum']})

total_China = confirmed_China.join(fatalities_China)





italy = [i for i in total_Italy.ConfirmedCases['sum'].values]

italy_30 = italy[0:50] 

spain = [i for i in total_Spain.ConfirmedCases['sum'].values]

spain_30 = spain[0:50] 

US = [i for i in total_US.ConfirmedCases['sum'].values]

US_30 = US[0:50] 

India = [i for i in total_India.ConfirmedCases['sum'].values]

India_30 = India[0:50] 

China = [i for i in total_China.ConfirmedCases['sum'].values]

China_30 = China[0:50] 



plt.figure(figsize=(12,6))

plt.plot(italy_30)

plt.plot(spain_30)

plt.plot(US_30)

plt.plot(India_30)

plt.plot(China_30)

plt.legend(["Italy", "Spain", "US", "India", "China"], loc='upper left')

plt.title("COVID-19 infections from the first confirmed case", size=20)

plt.xlabel("Days", size=20)

plt.ylabel("Infected cases", size=20)

plt.ylim(0, 100000)

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

plt.ylabel("Population", fontsize=10);

plt.legend(loc='best')

plt.xlim(0,1000)

plt.savefig('SIR')

plt.show()
ppl_China= 1437857876;

population = float(ppl_China)

country_df = total_China[9:]

country_df['day_count'] = list(range(1,len(country_df)+1))



ydata = [i for i in country_df.ConfirmedCases['sum'].values]

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

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
population = float(ppl_italy)

country_df = total_Italy[9:]

country_df['day_count'] = list(range(1,len(country_df)+1))



ydata = [i for i in country_df.ConfirmedCases['sum'].values]

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

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
population = float(ppl_spain)

country_df = total_Spain[9:]

country_df['day_count'] = list(range(1,len(country_df)+1))



ydata = [i for i in country_df.ConfirmedCases['sum'].values]

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

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
population = float(ppl_India)

country_df = total_India[9:]

country_df['day_count'] = list(range(1,len(country_df)+1))



ydata = [i for i in country_df.ConfirmedCases['sum'].values]

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

plt.title("Fit of SIR model to global infected cases")

plt.ylabel("Population infected")

plt.xlabel("Days")

plt.show()

print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])
# Merge train and test, exclude overlap

dates_overlap = ['2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18','2020-03-19','2020-03-20','2020-03-21','2020-03-22']

train2 = train.loc[~train['Date'].isin(dates_overlap)]

all_data = pd.concat([train2, test], axis = 0, sort=False)



# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11

all_data.loc[all_data['Date'] >= '2020-03-12', 'ConfirmedCases'] = np.nan

all_data.loc[all_data['Date'] >= '2020-03-12', 'Fatalities'] = np.nan

all_data['Date'] = pd.to_datetime(all_data['Date'])



# Create date columns

le = preprocessing.LabelEncoder()

all_data['Day_num'] = le.fit_transform(all_data.Date)

all_data['Day'] = all_data['Date'].dt.day

all_data['Month'] = all_data['Date'].dt.month

all_data['Year'] = all_data['Date'].dt.year



# Fill null values given that we merged train-test datasets

all_data['Province/State'].fillna("None", inplace=True)

all_data['ConfirmedCases'].fillna(0, inplace=True)

all_data['Fatalities'].fillna(0, inplace=True)

all_data['Id'].fillna(-1, inplace=True)

all_data['ForecastId'].fillna(-1, inplace=True)



# Aruba has no Lat nor Long. Inform it manually

all_data.loc[all_data['Lat'].isna()==True, 'Lat'] = 12.510052

all_data.loc[all_data['Long'].isna()==True, 'Long'] = -70.009354



display(all_data)

display(all_data.loc[all_data['Date'] == '2020-03-12'])
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

all_data = calculate_trend(all_data, [1], 'ConfirmedCases')

all_data = calculate_trend(all_data, [1], 'Fatalities')

all_data.replace([np.inf, -np.inf], 0, inplace=True)

all_data.fillna(0, inplace=True)

print("Time spent: ", time.time()-ts)
all_data[all_data['Country/Region']=='Spain'].iloc[40:50][['Id', 'Province/State', 'Country/Region', 'Lat', 'Long', 'Date',

       'ConfirmedCases', 'Fatalities', 'ForecastId', 'Day_num', 'ConfirmedCases_1', 'ConfirmedCases_2', 'ConfirmedCases_3', 'Fatalities_1', 'Fatalities_2',

       'Fatalities_3']]

# Load countries data file

world_population = pd.read_csv("../input/population-by-country-2020/population_by_country_2020.csv")



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

all_data = all_data.merge(world_population, left_on='Country/Region', right_on='Country (or dependency)', how='left')

all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']] = all_data[['Population (2020)', 'Density', 'Land Area', 'Med Age', 'Urban Pop']].fillna(0)

display(all_data)



print("Encoded dataset")

# Label encode countries and provinces. Save dictionary for exploration purposes

all_data.drop('Country (or dependency)', inplace=True, axis=1)

all_data['Country/Region'] = le.fit_transform(all_data['Country/Region'])

number_c = all_data['Country/Region']

countries = le.inverse_transform(all_data['Country/Region'])

country_dict = dict(zip(countries, number_c)) 

all_data['Province/State'] = le.fit_transform(all_data['Province/State'])

number_p = all_data['Province/State']

province = le.inverse_transform(all_data['Province/State'])

province_dict = dict(zip(province, number_p)) 

display(all_data)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



# Day_num = 38 is March 1st

y1 = all_data[(all_data['Lat']==40.0) & (all_data['Long']==-4.0) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']]

x1 = range(0, len(y1))

ax1.plot(x1, y1, 'bo--')

ax1.set_title("Spain ConfirmedCases between days 39 and 49 (last 10 days)")

ax1.set_xlabel("Days")

ax1.set_ylabel("ConfirmedCases")



y2 = all_data[(all_data['Lat']==40.0) & (all_data['Long']==-4.0) & (all_data['Day_num']>39) & (all_data['Day_num']<=49)][['ConfirmedCases']].apply(lambda x: np.log(x))

x2 = range(0, len(y2))

ax2.plot(x2, y2, 'bo--')

ax2.set_title("Spain Log ConfirmedCases between days 39 and 49 (last 10 days)")

ax2.set_xlabel("Days")

ax2.set_ylabel("Log ConfirmedCases")
# Filter selected features

data = all_data.copy()

features = ['Id', 'ForecastId', 'Country/Region', 'Province/State', 'ConfirmedCases', 'Fatalities', 

       'Day_num', 'Day', 'Month', 'Year', 'Long', 'Lat']

data = data[features]



# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log(x))



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

    index = x_test['ForecastId'].astype('int32')

    x_test.drop('ForecastId', inplace=True, errors='ignore', axis=1)

    

    return x_train, y_train_1, y_train_2, x_test, index





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

def get_submission(index, df):

    

    prediction_1 = data_pred['Predicted_ConfirmedCases']

    prediction_2 = data_pred['Predicted_Fatalities']



    # Submit predictions

    prediction_1 = [int(item) for item in list(map(round, prediction_1))]

    prediction_2 = [int(item) for item in list(map(round, prediction_2))]

    

    submission = pd.DataFrame({

        "ForecastId": df['ForecastId'].astype('int32'), 

        "ConfirmedCases": prediction_1, 

        "Fatalities": prediction_2

    })

    submission.to_csv('submission.csv', index=False)
ts = time.time()



day_start = 39

data2 = data.loc[data.Day_num >= day_start]



# Set the dataframe where we will update the predictions

data_pred = data[data.ForecastId != -1][['Country/Region', 'Province/State', 'Day_num', 'ForecastId']]

data_pred = data_pred.loc[data_pred['Day_num']>=day_start]

data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)

data_pred['Predicted_Fatalities'] = [0]*len(data_pred)

    

print("Currently running Logistic Regression for all countries")



# Main loop for countries

for c in data2['Country/Region'].unique():

    

    # List of provinces

    provinces_list = data2[data2['Country/Region']==c]['Province/State'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        for p in provinces_list:

            data_cp = data2[(data2['Country/Region']==c) & (data2['Province/State']==p)]

            X_train, Y_train_1, Y_train_2, X_test, index = split_data(data_cp)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred.loc[((data_pred['Country/Region']==c) & (data2['Province/State']==p)), 'Predicted_ConfirmedCases'] = pred_1

            data_pred.loc[((data_pred['Country/Region']==c) & (data2['Province/State']==p)), 'Predicted_Fatalities'] = pred_2



    # No Province/State informed

    else:

        data_c = data2[(data2['Country/Region']==c)]

        X_train, Y_train_1, Y_train_2, X_test, index = split_data(data_c)

        model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

        model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

        data_pred.loc[(data_pred['Country/Region']==c), 'Predicted_ConfirmedCases'] = pred_1

        data_pred.loc[(data_pred['Country/Region']==c), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.exp(x))

data_pred.replace([np.inf, -np.inf], 0, inplace=True) 



get_submission(index, data_pred)



print("Process finished in ", round(time.time() - ts, 2), " seconds")
ts = time.time()



# Set the dataframe where we will update the predictions

data_pred = data[data.ForecastId != -1][['Country/Region', 'Province/State', 'Day_num', 'ForecastId']]

data_pred['Predicted_ConfirmedCases'] = [0]*len(data_pred)

data_pred['Predicted_Fatalities'] = [0]*len(data_pred)

how_many_days = test.Date.nunique()

    

print("Currently running Logistic Regression for all countries")



# Main loop for countries

for c in data['Country/Region'].unique():

    

    # List of provinces

    provinces_list = data2[data2['Country/Region']==c]['Province/State'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        

        for p in provinces_list:

            # Only fit starting from the first confirmed case in the country

            train_countries_no0 = data.loc[(data['Country/Region']==c) & (data['Province/State']==p) & (data.ConfirmedCases!=0) & (data.ForecastId==-1)]

            test_countries_no0 = data.loc[(data['Country/Region']==c) & (data['Province/State']==p) &  (data.ForecastId!=-1)]

            data2 = pd.concat([train_countries_no0, test_countries_no0])



            # If there are no previous cases, predict 0

            if len(train_countries_no0) == 0:

                data_pred.loc[((data_pred['Country/Region']==c) & (data_pred['Province/State']==p)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

                data_pred.loc[((data_pred['Country/Region']==c) & (data_pred['Province/State']==p)), 'Predicted_Fatalities'] = [0]*how_many_days

                

            # Else run LinReg

            else: 

                data_cp = data2[(data2['Country/Region']==c) & (data2['Province/State']==p)]

                X_train, Y_train_1, Y_train_2, X_test, index = split_data(data_cp)

                model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

                model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

                data_pred.loc[((data_pred['Country/Region']==c) & (data2['Province/State']==p)), 'Predicted_ConfirmedCases'] = pred_1

                data_pred.loc[((data_pred['Country/Region']==c) & (data2['Province/State']==p)), 'Predicted_Fatalities'] = pred_2



    # No Province/State informed

    else:

        # Only fit starting from the first confirmed case in the country

        train_countries_no0 = data.loc[(data['Country/Region']==c) & (data.ConfirmedCases!=0) & (data.ForecastId==-1)]

        test_countries_no0 = data.loc[(data['Country/Region']==c) &  (data.ForecastId!=-1)]

        data2 = pd.concat([train_countries_no0, test_countries_no0])



        # If there are no previous cases, predict 0

        if len(train_countries_no0) == 0:

            data_pred.loc[((data_pred['Country/Region']==c)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

            data_pred.loc[((data_pred['Country/Region']==c)), 'Predicted_Fatalities'] = [0]*how_many_days

        

        # Else, run LinReg

        else:

            data_c = data2[(data2['Country/Region']==c)]

            X_train, Y_train_1, Y_train_2, X_test, index = split_data(data_c)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred.loc[(data_pred['Country/Region']==c), 'Predicted_ConfirmedCases'] = pred_1

            data_pred.loc[(data_pred['Country/Region']==c), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.exp(x))

data_pred.replace([np.inf, -np.inf], 0, inplace=True) 



#get_submission(index, data_pred)



print("Process finished in ", round(time.time() - ts, 2), " seconds")