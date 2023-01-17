# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
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
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-4/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-4/train.csv")

train.Province_State.fillna("None", inplace=True)

display(train.head(5))

display(train.describe())

print("Number of Country_Region: ", train['Country_Region'].nunique())

print("Dates go from day", max(train['Date']), "to day", min(train['Date']), ", a total of", train['Date'].nunique(), "days")

print("Countries with Province/State informed: ", train.loc[train['Province_State']!='None']['Country_Region'].unique())

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

confirmed_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'ConfirmedCases':['sum']})

fatalities_total_date_noChina = train[train['Country_Region']!='China'].groupby(['Date']).agg({'Fatalities':['sum']})

total_date_noChina = confirmed_total_date_noChina.join(fatalities_total_date_noChina)



fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))

total_date_noChina.plot(ax=ax1)

ax1.set_title("Global confirmed cases excluding China", size=13)

ax1.set_ylabel("Number of cases", size=13)

ax1.set_xlabel("Date", size=13)

fatalities_total_date_noChina.plot(ax=ax2, color='orange')

ax2.set_title("Global deceased cases excluding China", size=13)

ax2.set_ylabel("Number of cases", size=13)

ax2.set_xlabel("Date", size=13)
confirmed_country_Italy = train[train['Country_Region']=='Italy'].groupby(['Country_Region', 'Province_State']).agg({'ConfirmedCases':['sum']})

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



plt.figure(figsize=(17,10))

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

plt.ylim(0, 0.5)



plt.subplot(2, 2, 2)

total_date_Spain.ConfirmedCases.plot(ax=plt.gca(), title='Spain')

plt.ylim(0, 0.5)



plt.subplot(2, 2, 3)

total_date_UK.ConfirmedCases.plot(ax=plt.gca(), title='United Kingdom')

plt.ylabel("Fraction of population infected")

plt.ylim(0, 0.1)



plt.subplot(2, 2, 4)

total_date_Singapore.ConfirmedCases.plot(ax=plt.gca(), title='Singapore')

plt.ylim(0, 0.05)
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

dates_overlap = ['2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05', '2020-04-06', '2020-04-07', '2020-04-08',

                 '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13', '2020-04-14']

train2 = train.loc[~train['Date'].isin(dates_overlap)]

all_data = pd.concat([train2, test], axis = 0, sort=False)



# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11

all_data.loc[all_data['Date'] >= '2020-04-01', 'ConfirmedCases'] = 0

all_data.loc[all_data['Date'] >= '2020-04-01', 'Fatalities'] = 0

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

display(all_data.loc[all_data['Date'] == '2020-04-01'])
missings_count = {col:all_data[col].isnull().sum() for col in all_data.columns}

missings = pd.DataFrame.from_dict(missings_count, orient='index')

print(missings.nlargest(30, 0))
def calculate_lag(df, lag_list, column):

    for lag in lag_list:

        column_lag = column + "_" + str(lag)

        df[column_lag] = df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0)

    return df



def calculate_trend(df, lag_list, column):

    for lag in lag_list:

        trend_column_lag = "Trend_" + column + "_" + str(lag)

        df[trend_column_lag] = (df.groupby(['Country_Region', 'Province_State'])[column].shift(0, fill_value=0) - 

                                df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0))/df.groupby(['Country_Region', 'Province_State'])[column].shift(lag, fill_value=0.001)

    return df





ts = time.time()

all_data = calculate_lag(all_data.reset_index(), range(1,7), 'ConfirmedCases')

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

       'Day_num']

data = data[features]



# Apply log transformation to all ConfirmedCases and Fatalities columns, except for trends

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')

data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))



# Replace infinites

data.replace([np.inf, -np.inf], 0, inplace=True)





# Split data into train/test

def split_data(df, train_lim, test_lim):

    

    df.loc[df['Day_num']<=train_lim , 'ForecastId'] = -1

    df = df[df['Day_num']<=test_lim]

    

    # Train set

    x_train = df[df.ForecastId == -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)

    y_train_1 = df[df.ForecastId == -1]['ConfirmedCases']

    y_train_2 = df[df.ForecastId == -1]['Fatalities']



    # Test set

    x_test = df[df.ForecastId != -1].drop(['ConfirmedCases', 'Fatalities'], axis=1)



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
# Select train (real) data from March 1 to last day updated

dates_list = ['2020-03-01', '2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05', '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09', 

                 '2020-03-10', '2020-03-11','2020-03-12','2020-03-13','2020-03-14','2020-03-15','2020-03-16','2020-03-17','2020-03-18',

                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27', 

                 '2020-03-28', '2020-03-29', '2020-03-30', '2020-03-31', '2020-04-01', '2020-04-02', '2020-04-03', '2020-04-04', '2020-04-05', 

                 '2020-04-06', '2020-04-07', '2020-04-08', '2020-04-09', '2020-04-10', '2020-04-11', '2020-04-12', '2020-04-13', '2020-04-14']
all_data.loc[all_data['Country_Region']==country_dict['Spain']][50:70]

def plot_linreg_basic_country(data, country_name, dates_list, day_start, shift, train_lim, test_lim):

    

    data_country = data[data['Country_Region']==country_dict[country_name]]

    data_country = data_country.loc[data_country['Day_num']>=day_start]

    X_train, Y_train_1, Y_train_2, X_test = split_data(data_country, train_lim, test_lim)

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

    ax1.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (from March " + str(1+shift) + " to March 25th)")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(30-shift, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (from March " + str(1+shift) + " to March 30th)")

    ax2.set_ylabel("Log Confirmed Cases")



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))

    

    

    

# Filter Spain, run the Linear Regression workflow

country_name = "Spain"

march_day = 10

day_start = 39+march_day

dates_list2 = dates_list[march_day:]

train_lim, test_lim = 69, 113

plot_linreg_basic_country(data, country_name, dates_list2, day_start, march_day, train_lim, test_lim)
ts = time.time()



# Set the dataframe where we will update the predictions

day_start = 65

data2 = data.loc[data.Day_num >= day_start]

data_pred3 = data2[data2.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]

data_pred3['Predicted_ConfirmedCases'] = [0]*len(data_pred3)

data_pred3['Predicted_Fatalities'] = [0]*len(data_pred3)

how_many_days = test.Date.nunique()

    

print("Currently running Linear Regression for all countries")



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

                X_train, Y_train_1, Y_train_2, X_test = split_data(data_cp, train_lim, test_lim)

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

            X_train, Y_train_1, Y_train_2, X_test = split_data(data_c, train_lim, test_lim)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred3.loc[(data_pred3['Country_Region']==c), 'Predicted_ConfirmedCases'] = pred_1

            data_pred3.loc[(data_pred3['Country_Region']==c), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

#data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))

#data_pred3.replace([np.inf, -np.inf], 0, inplace=True) 



#get_submission(data_pred3, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')



print("Process finished in ", round(time.time() - ts, 2), " seconds")
# Filter Andorra, run the Linear Regression workflow

country_name = "Andorra"

shift = 21

day_start = 39+shift

dates_list2 = dates_list[shift:]

plot_linreg_basic_country(data, country_name, dates_list2, day_start, shift, train_lim, test_lim)
# New split function, for one forecast day

def split_data_one_day(df, d, train_lim, test_lim):

    

    df.loc[df['Day_num']<=train_lim , 'ForecastId'] = -1

    df = df[df['Day_num']<=test_lim]

    

    #Train

    x_train = df[df.Day_num<d]

    y_train_1 = x_train.ConfirmedCases

    y_train_2 = x_train.Fatalities

    x_train.drop(['ConfirmedCases', 'Fatalities'], axis=1, inplace=True)

    

    #Test

    x_test = df[df.Day_num==d]

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

    ax1.axvline(30-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March " + str(march_day) + "))")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(30-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (starting on March " + str(march_day) + ")")

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

    ax1.axvline(30-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March " + str(march_day) + ")")

    ax1.set_ylabel("Fatalities Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log1p(real_data))

    ax2.axvline(30-march_day, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (starting on March " + str(march_day) + ")")

    ax2.set_ylabel("Log Fatalities Cases")



    plt.suptitle(("Fatalities predictions based on Log-Lineal Regression for "+country_name))
# Function to compute the Linear Regression predictions with lags, for a certain Country/Region

def lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict, train_lim, test_lim):

    

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

        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d, train_lim, test_lim)

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

def lin_reg_with_lags_country_province(all_data, country_name, province_name, day_start, lag_size, country_dict, train_lim, test_lim):

    

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

        X_train, Y_train_1, Y_train_2, X_test = split_data_one_day(data, d, train_lim, test_lim)

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

march_day = 10

day_start = 39 + march_day

dates_list2 = dates_list[march_day:]

lag_size = 30



data_c = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict, train_lim, test_lim)

plot_real_vs_prediction_country(data_c, train, country_name, day_start, dates_list2, march_day)

plot_real_vs_prediction_country_fatalities(data_c, train, country_name, day_start, dates_list2, march_day)
def logistic_function(x, a, b, c, d):

    return a / (1. + np.exp(-c * (x - d))) + b



def fit_logistic(all_data, country_name, province_name, train_lim, target):

    data_cp = all_data.loc[(all_data['Country_Region']==country_dict[country_name]) & (all_data['Province_State']==province_dict[province_name])]

    y = data_cp.loc[(data_cp['Day_num'])<=train_lim, target].astype(np.int32)

    x = list(range(0, len(y)))



    # Initial guess

    p0 = [0,1,1,0]



    (a_, b_, c_, d_), cov = optimize.curve_fit(logistic_function, x, y, bounds=(0, [500000., 10., 1000., 1000., ]), p0=p0, maxfev=10**9)

    y_fit = logistic_function(x, a_, b_, c_, d_)

    

    return x, y, y_fit, (a_, b_, c_, d_), cov



def plot_logistic(x, y, y_fit, country_name, province_name, target):

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax.plot(x, y, 'o')

    ax.plot(x, y_fit, '-')

    ax.set_xlabel("Day count (starting on January 22nd)")

    ax.set_ylabel(target)

    ax.set_title("Fit to logistic regression for "+ country_name+"/"+province_name)

    



def plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, target, x, a_, b_, c_, d_):

    forecast = logistic_function(list(range(len(x)+60)), a_, b_, c_, d_)

    df_train = train.loc[(train['Country_Region']==country_name) & (train['Province_State']==province_name), target]

    df_fcst = forecast[:len(df_train)]

    dates = list(range(0,len(df_train)))

    

    # Plot results

    fig, (ax1) = plt.subplots(1, 1, figsize=(6,4))

    ax1.plot(dates, df_fcst)

    ax1.plot(dates, df_train)

    ax1.axvline(len(df_train)-valid_num-1, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.set_title("Actual ConfirmedCases vs predictions based on Logistic curve for "+country_name + "/"+province_name)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count starting on January 22nd")

    ax1.set_ylabel("ConfirmedCases")





# Fit country to logistic curve

country_name = 'Spain'

province_name = 'None'

train_lim = 69

valid_lim = 84 # needs to be changed as more days of training data are included

test_lim = 112

valid_num=valid_lim-train_lim 



x, y, y_fit, (a_, b_, c_, d_), cov = fit_logistic(all_data, country_name, province_name, train_lim, 'ConfirmedCases')

plot_logistic(x, y, y_fit, country_name, province_name, 'ConfirmedCases')

plot_logistic_country(all_data, train, dates_overlap, country_name, province_name, valid_num, 'ConfirmedCases', x, a_, b_, c_, d_)

submission_example = pd.read_csv("../input/covid19-global-forecasting-week-4/submission.csv")

sub= pd.read_csv("submission.csv")
