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

        df[trend_column_lag] = (df[column]-df[column].shift(periods=lag).fillna(-999))/df[column].shift(periods=lag).fillna(0)

    return df





def calculate_lag(df, lag_list, column):

    for lag in lag_list:

        column_lag = column + "_" + str(lag)

        df[column_lag] = df[column].shift(periods=lag).fillna(0)

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
def put_SIR(data):

    df = pd.DataFrame()

    

    df['ConfirmedCases'] = data.ConfirmedCases.diff().fillna(0)

    df = df[5:]

    df['day_count'] = list(range(1,len(df)+1))

    df['ForecastId'] = data['ForecastId']

    

    dft = df.loc[df.ForecastId == -1]

    

    ydata = [i for i in dft.ConfirmedCases]

    xdata = dft.day_count

    ydata = np.array(ydata, dtype=float)

    xdata = np.array(xdata, dtype=float)

    

    N = data['Population (2020)'].iloc[0]

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

    

    #print(xdata)

    #print(ydata)

    

    try:

        popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata,  bounds=([1.2, 1], [10, 8]))

    except ValueError:

        popt = np.zeros(2)

        popt[0] = 6

        popt[1] = 4

        print('Ups')

        

    xdata = df.day_count

    

    fitted = fit_odeint(xdata, *popt)

    

    fitted = fitted[len(dft)]

    

    m = mortality_ratio(data)

    

    mort = fitted * m

    #print("Optimal parameters: beta =", popt[0], " and gamma = ", popt[1])

    #print("Mort = ", m)

    

    return fitted, mort
m_global = all_data['Fatalities'].fillna(0).sum() / (all_data['ConfirmedCases'].fillna(0).sum() )

print(m_global)
def mortality_ratio(data):

    df = pd.DataFrame()

    data2 = data.loc[data.ForecastId != -1]

    df['ConfirmedCases'] = data.ConfirmedCases.diff().fillna(0)

    df['Fatalities'] = data['Fatalities']

    

    df['day_count'] = list(range(1,len(df)+1))

    df['ForecastId'] = data['ForecastId']

    

    dft = df.loc[df.ForecastId != -1]



    m = dft['Fatalities'].fillna(0).sum() / (dft['ConfirmedCases'].fillna(0).sum() + 1)

        

    return m_global
march_day = 0

day_start = 39+march_day
ig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



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

#data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].astype('float64')

#data[['ConfirmedCases', 'Fatalities']] = data[['ConfirmedCases', 'Fatalities']].apply(lambda x: np.log1p(x))



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
data = all_data

# Set the dataframe where we will update the predictions

data2 = all_data.loc[data.Day_num >= day_start]

data_pred3 = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]

data_pred3['Predicted_ConfirmedCases'] = [0]*len(data_pred3)

data_pred3['Predicted_Fatalities'] = [0]*len(data_pred3)

how_many_days = test.Date.nunique()

    

print("Currently running SIR for all countries")



# Main loop for countries

for c in data['Country_Region'].unique():

    

    # List of provinces

    provinces_list = data2[data2['Country_Region']==c]['Province_State'].unique()

        

    # If the country has several Province/State informed

    if len(provinces_list)>1:

        

        for p in provinces_list:

            # Only fit starting from the first confirmed case in the country

            train_countries_no0 = data.loc[(data['Country_Region']==c) & (data['Province_State']==p)  & (data.ForecastId==-1)]

            test_countries_no0 = data.loc[(data['Country_Region']==c) & (data['Province_State']==p) &  (data.ForecastId!=-1)]

            data2 = pd.concat([train_countries_no0, test_countries_no0])



            data_cp = data2[(data2['Country_Region']==c) & (data2['Province_State']==p)]

            

            pred_1, pred_2 = put_SIR(data_cp)

            

            data_pred3.loc[((data_pred3['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_ConfirmedCases'] = pred_1

            data_pred3.loc[((data_pred3['Country_Region']==c) & (data_pred3['Province_State']==p)), 'Predicted_Fatalities'] = pred_2



    # No Province/State informed

    else:

        # Only fit starting from the first confirmed case in the country

        train_countries_no0 = data.loc[(data['Country_Region']==c) & (data.ForecastId==-1)]

        test_countries_no0 = data.loc[(data['Country_Region']==c) &  (data.ForecastId!=-1)]

        data2 = pd.concat([train_countries_no0, test_countries_no0])

        

        pred_1, pred_2 = put_SIR(data2)

            

        data_pred3.loc[((data_pred3['Country_Region']==c)), 'Predicted_ConfirmedCases'] = pred_1

        data_pred3.loc[((data_pred3['Country_Region']==c) ), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

#data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred3[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.expm1(x))

data_pred3.replace([np.inf, -np.inf], 0, inplace=True) 

data_pred3.fillna(0, inplace=True)



get_submission(data_pred3, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')



print("Process finished in ", round(time.time() - ts, 2), " seconds")