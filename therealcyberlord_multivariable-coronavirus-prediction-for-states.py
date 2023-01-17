import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression, BayesianRidge, LinearRegression
import datetime
import matplotlib.pyplot as plt 
import warnings
import tensorflow as tf 
from scipy.stats import exponweib
%matplotlib inline
data = pd.read_csv('https://covidtracking.com/api/v1/states/daily.csv')
data.head()
data.info()
data.isnull().sum()
data.sort_values('date', inplace=True)
# drop columns with 50% null values 
def drop_na_50(df):
    columns = df.columns
    for col in columns:
       if data[col].isnull().sum() > 0.5 * len(df):
         df.drop(col, 1, inplace=True)
    return df 
         
new_data = drop_na_50(data)
new_data.isnull().sum()
new_data.fillna(0, inplace=True)
new_data[new_data.state=='CT'].hist(bins=10, figsize=(20, 15))
plt.show()
# drop variables with few/no variations
new_data.drop(['commercialScore', 'fips', 'hospitalizedIncrease', 'negativeRegularScore', 'negativeScore', 'positiveScore', 'score'], axis=1, inplace=True)
new_data[new_data.state=='CT'].hist(bins=10, figsize=(20, 15))
plt.show()
new_data.isnull().sum()
# dictonary for mapping state to abbreviation 
us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}

us_state_abbrev['Connecticut']
# this data frame is going to provide static variables
additional_data = pd.read_csv('../input/covid19-state-data/COVID19_state.csv')
additional_data.head()
additional_data.isnull().sum()
# additional_data.hist(bins=10, figsize=(20, 15))
# plt.show()
# make the data consistent with the other data frame 
for i in additional_data.State.unique():
    additional_data.replace(i, us_state_abbrev[i], inplace=True)
additional_data[additional_data.State=='CA']['Pop Density']
# incoporate population density into prediction 
pop_dense = [] 
for i in new_data.state:
    pop_dense.append(additional_data[additional_data.State==i]['Pop Density'].sum())
new_data['pop_dense'] = pop_dense
new_data['pop_dense'].head()
# scales the population density based on all the states 
min_max_scaler = MinMaxScaler()
new_data['pop_dense'] = min_max_scaler.fit_transform(np.array(new_data['pop_dense']).reshape(-1, 1))
new_data['pop_dense'].head()
#this dataset is going to provide mobility information within the US

mobility_data = pd.read_csv('https://covid19-static.cdn-apple.com/covid19-mobility-data/2014HotfixDev17/v3/en-us/applemobilitytrends-2020-08-16.csv')
mobility_data.head()
mobility_data.isnull().sum()
# make the data consistent with the other data frame 
for i in mobility_data['sub-region'].unique():
    if i in us_state_abbrev.keys():
        mobility_data.replace(i, us_state_abbrev[i], inplace=True)
mobility_data['transportation_type'].unique()
# get individual states and dates
unique_states = new_data.state.unique()
unique_states.sort()
unique_dates = new_data.date.unique()

# making sure that the dates match between mobility and testing/cases
mobility_latest_date = datetime.datetime.strptime(mobility_data.columns[-1], '%Y-%m-%d').strftime('%Y%m%d')
mobility_latest_index = np.where(unique_dates == int(mobility_latest_date))[0][0]

# start from a later date 3/1/2020
unique_dates = unique_dates[39:mobility_latest_index+1]
# gets the mobility information of a particular day
def get_mobility_by_state(transport_type, state, day):
    return mobility_data[mobility_data['sub-region']==state][mobility_data['transportation_type']==transport_type].sum()[day]
get_mobility_by_state('walking', 'FL', '2020-03-01')
# change the date format to match the mobility data 
revised_unique_dates = [] 
for i in range(len(unique_dates)):
    revised_unique_dates.append(datetime.datetime.strptime(str(unique_dates[i]), '%Y%m%d').strftime('%Y-%m-%d'))
revised_unique_dates
print(get_mobility_by_state('transit', 'FL', revised_unique_dates[9]))
def convert_date_to_int(d):
    return [i for i in range(len(revised_unique_dates))]

days_since_3_1 = convert_date_to_int(revised_unique_dates)
days_ahead = 10
future_dates = [i for i in range(len(revised_unique_dates)+days_ahead)]
def svm_reg(X_train, X_test, y_train, y_test, future_forecast, state):
        
    svm_confirmed = SVR(shrinking=True, kernel='poly',gamma=0.01, epsilon=1, degree=4, C=0.1)
    svm_confirmed.fit(X_train, y_train)
    test_svm_pred = svm_confirmed.predict(X_test)
    svm_pred = svm_confirmed.predict(future_forecast)
    
    plt.plot(y_test)
    plt.plot(test_svm_pred)
    plt.title('Testing Set Evaluation for {}'.format(state))
    plt.xlabel('Days since 3/1/2020')
    plt.ylabel('# of positive coronavirus cases')
    plt.legend(['Actual', 'Predicted'])
    plt.show()
    
    print('MAE:', mean_absolute_error(test_svm_pred, y_test))
    print('MSE:',mean_squared_error(test_svm_pred, y_test))

    # plot the graph to see compare predictions and actual coronavirus cases
    plt.plot(positive)
    plt.plot(svm_pred)
    plt.title('Coronavirus Cases in {}'.format(state))
    plt.legend(['Actual cases', 'Predicted cases using support vector regression'])
    plt.xlabel('Days since 3/1/2020')
    plt.ylabel('# of positive coronavirus cases')
    plt.show()
    print('Completed:', state)

def bayesian_ridge(X_train, X_test, y_train, y_test, future_forecast, state):
        
    # convert data to be compatible with polynomial regression
    bayesian_poly = PolynomialFeatures(degree=3)
    bayesian_poly_X_train = bayesian_poly.fit_transform(X_train)
    bayesian_poly_X_test = bayesian_poly.fit_transform(X_test)
    bayesian_poly_future_forecast = bayesian_poly.fit_transform(future_forecast)
    
    # polynomial regression model
    # bayesian ridge polynomial regression
    tol = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    alpha_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    alpha_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_1 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    lambda_2 = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
    normalize = [True, False]
    fit_intercept = [True,  False]
    lambda_init = [1e-2, 1e-1, 1, 1e1]

    bayesian_grid = {'tol': tol, 'alpha_1': alpha_1, 'alpha_2' : alpha_2, 'lambda_1': lambda_1, 'lambda_2' : lambda_2, 
                    'normalize' : normalize, 'fit_intercept': fit_intercept, 'lambda_init' : lambda_init}

    bayesian = BayesianRidge()
    bayesian_search = RandomizedSearchCV(bayesian, bayesian_grid, scoring='neg_root_mean_squared_error', cv=3, return_train_score=True, n_jobs=-1, n_iter=200, verbose=1)
    bayesian_search.fit(bayesian_poly_X_train, y_train)
    
    # get the best estimator 
    best_params = bayesian_search.best_params_
    bayesian_confirmed = BayesianRidge(**best_params)
    bayesian_confirmed.fit(bayesian_poly_X_train, y_train)
    
    test_bayesian_pred = bayesian_confirmed.predict(bayesian_poly_X_test)
    bayesian_pred = bayesian_confirmed.predict(bayesian_poly_future_forecast)
    
    plt.plot(y_test)
    plt.plot(test_bayesian_pred)
    plt.title('Testing Set Evaluation for {}'.format(state))
    plt.xlabel('Days since 3/1/2020')
    plt.ylabel('# of positive coronavirus cases')
    plt.legend(['Actual', 'Predicted'])
    plt.show()
    
    print('MAE:', mean_absolute_error(test_bayesian_pred, y_test))
    print('MSE:',mean_squared_error(test_bayesian_pred, y_test))
    print('Weight:', bayesian_confirmed.coef_)

    # plot the graph to see compare predictions and actual coronavirus cases
    plt.plot(positive)
    plt.plot(bayesian_pred)
    plt.title('Coronavirus Cases in {}'.format(state))
    plt.legend(['Actual cases', 'Predicted cases using bayesian ridge'])
    plt.xlabel('Days since 3/1/2020')
    plt.ylabel('# of positive coronavirus cases')
    plt.show()
    print('Completed:', state)

# helper function for finding daily change 
def daily_change(y2, y1):
    return (y2-y1)
# get moving average for positive case 

def moving_positive_cases(data, window_size):
    moving_positive = []
    for i in range(len(data)):
        if i + window_size < len(data):
            moving_positive.append(np.mean(data[i:i+window_size]))
        else:
            moving_positive.append(np.mean(data[i:len(data)]))
    return moving_positive
def future_testing_extrapolation(X, y, future_forecast, state):
    poly = PolynomialFeatures(degree=3)
    poly_X = poly.fit_transform(X)
    poly_future_forecast = poly.fit_transform(future_forecast)
    
    poly_confirmed = LinearRegression(fit_intercept=True, normalize=True)
    poly_confirmed.fit(poly_X, y)
    
    poly_pred = poly_confirmed.predict(poly_future_forecast)
    
    plt.plot(y)
    plt.plot(poly_pred)
    plt.title('Coronavirus testing in {}'.format(state))
    plt.legend(['Actual testing', 'Predicted testing using polynomial regression'])
    plt.xlabel('Days since 3/1/2020')
    plt.ylabel('# of testing')
    plt.show()
    
    future_increases = [] 
    
    # calulate future rates of change 
    for i in range(days_ahead):
        c = len(X) - 1
        future_increases.append(daily_change(poly_pred[c+i+1], poly_pred[c+i]))
    return future_increases
days_since_3_1 = np.array(days_since_3_1).reshape(-1, 1)
future_dates = np.array(future_dates).reshape(-1, 1)
def window_average(window_size, data, method):
    avg_data = [] 
    date_length = len(data)
    
    for i in range(len(data)):
        remainder = i % window_size 
        if method == 'median':
            if i - remainder + window_size - 1 < date_length:
                avg_data.append(np.median(data[i-remainder:i-remainder+window_size-1]))
            else:
                delta = date_length % window_size 
                avg_data.append(np.median(data[date_length-delta-1:date_length-1]))
        elif method == 'mean':
             if i - remainder + window_size - 1 < date_length:
                avg_data.append(np.mean(data[i-remainder:i-remainder+window_size-1]))
             else:
                delta = date_length % window_size 
                avg_data.append(np.mean(data[date_length-delta-1:date_length-1]))
        else:
            warnings.warn('Methods can only be mean or median')
            
    return avg_data
# returns true if it is a weekend, and false if it is a weekday 
def weekday_or_weekend(date):
    date_obj = datetime.datetime.strptime(str(date), '%Y%m%d')
    day_of_the_week =  date_obj.weekday()
    if (day_of_the_week+1) % 6 == 0 or (day_of_the_week+1) % 7 == 0:
        return True 
    else:
        return False 
len(future_dates[-10:])
# implementing this in the prediction in the future 
def mobility_scenario(mobility, mode):
    local_min = np.min(mobility)
    local_max = np.max(mobility)
    
    local_min_index = np.where(mobility==local_min)[0]
    local_max_index = np.where(mobility==local_max)[0]
    
    slope = (local_max - local_min) / (local_max_index - local_min_index)
    
#     plt.axhline(y=local_min, color='red')
#     plt.axhline(y=local_max, color='purple')
    
    # if mobility increases 
    
    if mode == 'decrease':
        # extrapolating mobility 
        m = mobility[-1] + slope 
        if m < local_min:
            future_mobility = local_min
        else:
            future_mobility = m
    
    # if mobility decreases 
    if mode == 'increase':
         # extrapolating mobility 
        m = mobility[-1] + np.abs(slope) 
        if m > local_max:
            future_mobility = local_max
        else:
            future_mobility = m
            
    return future_mobility
states = ['FL', 'CA', 'GA', 'TX']

for state in states:
    positive = []
    pop_density = [] 
    testing = [] 
    
    # mobility data
    walking_weekday = [] 
    walking_weekend = [] 
    walking = []
    walking_weekday_window = 7
    walking_weekend_window = 7
    
    # adjust window size for mobility
    
    date_length = len(revised_unique_dates)
    
    # get cases in sequential order for each state
    for i in range(date_length):
        positive.append(new_data[new_data.date==unique_dates[i]][new_data.state==state].positive.sum())
        pop_density.append(new_data[new_data.state==state]['pop_dense'].max())
        testing.append(new_data[new_data.date==unique_dates[i]][new_data.state==state].totalTestResults.sum())
        
        # determines if it is a weekend or weekday 
        if weekday_or_weekend(unique_dates[i]): 
            walking_weekend.append(get_mobility_by_state('walking', state, revised_unique_dates[i]))
        else:
            walking_weekday.append(get_mobility_by_state('walking', state, revised_unique_dates[i]))
        
#         remainder = i % window_size 
#         if i - remainder + window_size < date_length:
#             walking.append(get_mobility_by_state('walking', state, revised_unique_dates[i-remainder], revised_unique_dates[i-remainder+window_size-1], 'median'))
#         else:
#             # if extrapolating use the mobility average from the last few days based on the window size
#             delta = date_length % window_size 
#             walking.append(get_mobility_by_state('walking', state, revised_unique_dates[date_length-delta-1], revised_unique_dates[date_length-1], 'median'))


    # remove any decreases in cum testing and positive cases
    for i in range(len(testing)):
        if i != 0:
            if testing[i] < testing[i-1]:
                testing[i] = testing[i-1]
            if positive[i] < positive[i-1]:
                positive[i] = positive[i-1]
    
    # remove 0 in mobility from both weekday and weekend data (there are few null values from Apple's mobility data)
    for i in range(len(walking_weekend)):       
        if walking_weekend[i] == 0 and i != 0:
            walking_weekend[i] = walking_weekend[i-1]
            
    for i in range(len(walking_weekday)):
        if walking_weekday[i] == 0 and i != 0:
            walking_weekday[i] = walking_weekday[i-1]
            
    
    # taking window average for mobility 
    walking_weekday_avg = window_average(7, walking_weekday, 'mean')
    walking_weekend_avg = window_average(7, walking_weekend, 'mean')

    
    # making sure the shape of the mobility arrays match 
    r_walking_weekday_avg = [] 
    r_walking_weekend_avg = [] 
    
    k = 0 
    j = 0 
    for i in range(date_length):
        if i % walking_weekday_window == 0 and i != 0:
            if k + walking_weekday_window < len(walking_weekday_avg):
                k += walking_weekday_window
            else:
                k = len(walking_weekday_avg) - 1 
                
            if j + walking_weekend_window < len(walking_weekend_avg):
                j += walking_weekend_window
            else:
                j = len(walking_weekend_avg) - 1
        
        r_walking_weekday_avg.append(walking_weekday_avg[k])
        r_walking_weekend_avg.append(walking_weekend_avg[j])
        

    # take moving average for positive cases
    positive = moving_positive_cases(positive, 3)

    # future testing extrapolations from poylnomial prediction 
    future_testing = future_testing_extrapolation(days_since_3_1, testing, future_dates, state)
    for i in future_testing:
        testing.append(testing[-1] + i)
    
    testing = np.array(testing).reshape(-1, 1)
    positive = np.array(positive).reshape(-1, 1)
    r_walking_weekday_avg = np.array(r_walking_weekday_avg).reshape(-1, 1)
    r_walking_weekend_avg = np.array(r_walking_weekend_avg).reshape(-1, 1)
    
    min_max_scaler = MinMaxScaler()
    testing = min_max_scaler.fit_transform(testing)
    r_walking_weekday_avg = min_max_scaler.fit_transform(r_walking_weekday_avg)
    r_walking_weekend_avg = min_max_scaler.fit_transform(r_walking_weekend_avg)
    
    # combining the two features
    X = [] 
    future_forecast = []
    
    for i in range(len(days_since_3_1)):
        X.append([days_since_3_1[i][0], pop_density[0], testing[i][0], r_walking_weekday_avg[i][0], r_walking_weekend_avg[i][0]])
    
    X = np.array(X, object).reshape(-1, 5)
    
    for i in range(len(future_dates)):
        if i < date_length:
            future_forecast.append([future_dates[i][0], pop_density[0], testing[i][0], r_walking_weekday_avg[i][0], r_walking_weekend_avg[i][0]])
        else:
            future_forecast.append([future_dates[i][0], pop_density[0], testing[i][0], r_walking_weekday_avg[-1][0], r_walking_weekend_avg[-1][0]])
            
    future_forecast = np.array(future_forecast, object).reshape(-1, 5)
    
    # splitting into training and testing sets 
    X_train, X_test, y_train, y_test = train_test_split(X, positive, shuffle=False, test_size=0.05)
    bayesian_ridge(X_train, X_test, y_train, y_test, future_forecast, state)