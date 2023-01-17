import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import time

from datetime import datetime

import warnings

warnings.filterwarnings('ignore')



# ML libraries

import lightgbm as lgb

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

from sklearn import linear_model

from sklearn.metrics import mean_squared_error
submission_example = pd.read_csv("../input/covid19-global-forecasting-week-2/submission.csv")

test = pd.read_csv("../input/covid19-global-forecasting-week-2/test.csv")

train = pd.read_csv("../input/covid19-global-forecasting-week-2/train.csv")

for data in [train, test]:

    data.rename(

        columns={

            "Id": "id",

            "ForecastId": "forecast_id",

            "Date": "date",

            "Country_Region": "country",

            "Province_State": "city",

            "ConfirmedCases": "positives",

            "Fatalities": "deaths"},

        inplace=True

    )
display(train.head(5))

display(test.head(5))

display(train.describe())

print("Number of countries: ", train['country'].nunique())

print("Dates go from day", min(train['date']), "to day", max(train['date']), ", a total of", train['date'].nunique(), "days")

print("Countries with area information: ", train[~train['city'].isna()]['country'].unique())
# Total by date

total_by_date = train.groupby("date").sum().drop("id", axis=1)

total_by_date.plot()



# Total by date and country

total_by_date_country = train.groupby(["date", "country"]).sum().drop("id", axis=1)

#total_by_date_country.plot()





# Maximun by country

max_by_country = total_by_date_country.groupby("country").max().sort_values(by="positives", ascending=False)

n_countries = 10

top_countries = max_by_country.index[:n_countries].to_list()

max_by_country.head(n_countries).plot.bar()
# Merge train and test, exclude overlap

dates_overlap = ['2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27']

train2 = train.loc[~train['date'].isin(dates_overlap)]

all_data = pd.concat([train2, test], axis = 0, sort=False)



# Double check that there are no informed ConfirmedCases and Fatalities after 2020-03-11

all_data.loc[all_data['date'] >= '2020-03-19', 'positives'] = np.nan

all_data.loc[all_data['date'] >= '2020-03-19', 'deaths'] = np.nan

all_data['date'] = pd.to_datetime(all_data['date'])



# Create date columns

positives_by_date_country = all_data.groupby(["date", "country"])["positives"].sum().reset_index()

positives_by_date_country["counter"] = positives_by_date_country["positives"] > 0

positives_by_date_country["counter"] = positives_by_date_country.groupby(["country"])["counter"].cumsum().astype(int)

all_data = all_data.merge(positives_by_date_country)

all_data['day'] = all_data['date'].dt.day

all_data['month'] = all_data['date'].dt.month

all_data['year'] = all_data['date'].dt.year



display(all_data[all_data["country"]=="Germany"].head(50))

#display(all_data.loc[all_data['date'] == '2020-03-19'])
def calculate_trend(df, lag_list, column):

    for lag in lag_list:

        trend_column_lag = column + "_trend_" + str(lag)

        df[trend_column_lag] = (df[column]-df[column].shift(lag, fill_value=-999))/df[column].shift(lag, fill_value=0)

    return df





def calculate_lag(df, lag_list, column):

    for lag in lag_list:

        column_lag = column + "_lagged_" + str(lag)

        df[column_lag] = df[column].shift(lag, fill_value=0)

    return df





all_data = calculate_lag(all_data, range(1,7), 'positives')

all_data = calculate_lag(all_data, range(1,7), 'deaths')

all_data = calculate_trend(all_data, range(1,7), 'positives')

all_data = calculate_trend(all_data, range(1,7), 'deaths')

all_data.replace([np.inf, -np.inf], 0, inplace=True)

all_data.fillna(0, inplace=True)
all_data[all_data['country']=='Spain'].iloc[40:50]
# Load countries data file

world_population = pd.read_csv("/kaggle/input/population-by-country-2020/population_by_country_2020.csv")



# Select desired columns and rename some of them

world_population = world_population[['Country (or dependency)', 'Population (2020)', 'Density (P/Km²)', 'Land Area (Km²)', 'Med. Age', 'Urban Pop %']]

world_population.columns = ['country', 'population', 'density', 'land_area', 'age', 'population_urban']



# Replace United States by US

world_population.loc[world_population['country']=='United States', 'country'] = 'US'

# Remove the % character from Urban Pop values

world_population['population_urban'] = world_population['population_urban'].str.rstrip('%')

# Replace Urban Pop and Med Age "N.A" by their respective modes, then transform to int

world_population.loc[world_population['population_urban']=='N.A.', 'population_urban'] = int(world_population.loc[world_population['population_urban']!='N.A.', 'population_urban'].mode()[0])

world_population['population_urban'] = world_population['population_urban'].astype('int16')

world_population.loc[world_population['age']=='N.A.', 'age'] = int(world_population.loc[world_population['age']!='N.A.', 'age'].mode()[0])

world_population['age'] = world_population['age'].astype('int16')



print("Cleaned country details dataset")

display(world_population)



# Now join the dataset to our previous DataFrame and clean missings (not match in left join)- label encode cities

print("Joined dataset")

all_data = all_data.merge(world_population, left_on='country', right_on='country', how='left')

all_data[['population', 'density', 'land_area', 'age', 'population_urban']] = all_data[['population', 'density', 'land_area', 'age', 'population_urban']].fillna(0)

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

                 '2020-03-19','2020-03-20','2020-03-21','2020-03-22','2020-03-23', '2020-03-24', '2020-03-25', '2020-03-26', '2020-03-27']
all_data.loc[all_data['Country_Region']==country_dict['Spain']][45:65]
# Filter Spain, run the Linear Regression workflow

country_name = "Spain"

day_start = 39

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



ax1.plot(dates_list_num, np.exp(predicted_data))

ax1.plot(dates_list_num, real_data)

ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 25th)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, predicted_data)

ax2.plot(dates_list_num, np.log(real_data))

ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 25th)")

ax2.set_ylabel("Log Confirmed Cases")



plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
# Filter Italy, run the Linear Regression workflow

country_name = "Italy"

day_start = 39

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



# Select predictions from March 1st to March 24th

predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

dates_list_num = list(range(0,len(dates_list)))



# Plot results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



ax1.plot(dates_list_num, np.exp(predicted_data))

ax1.plot(dates_list_num, real_data)

ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 22nd)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, predicted_data)

ax2.plot(dates_list_num, np.log(real_data))

ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 22nd)")

ax2.set_ylabel("Log Confirmed Cases")



plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
# Filter Germany, run the Linear Regression workflow

country_name = "Germany"

day_start = 39

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





# Select predictions from March 1st to March 24th

predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

dates_list_num = list(range(0,len(dates_list)))



# Plot results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



ax1.plot(dates_list_num, np.exp(predicted_data))

ax1.plot(dates_list_num, real_data)

ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 22nd)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, predicted_data)

ax2.plot(dates_list_num, np.log(real_data))

ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 22nd)")

ax2.set_ylabel("Log Confirmed Cases")



plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
# Filter Albania, run the Linear Regression workflow

country_name = "Albania"

day_start = 39

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



# Select predictions from March 1st to March 24th

predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

dates_list_num = list(range(0,len(dates_list)))



# Plot results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



ax1.plot(dates_list_num, np.exp(predicted_data))

ax1.plot(dates_list_num, real_data)

ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 22nd)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, predicted_data)

ax2.plot(dates_list_num, np.log(real_data))

ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 22nd)")

ax2.set_ylabel("Log Confirmed Cases")



plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
# Filter Andorra, run the Linear Regression workflow

country_name = "Andorra"

day_start = 39

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



# Select predictions from March 1st to March 24th

predicted_data = X_final_check.loc[(X_final_check['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Target

real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

dates_list_num = list(range(0,len(dates_list)))



# Plot results

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



ax1.plot(dates_list_num, np.exp(predicted_data))

ax1.plot(dates_list_num, real_data)

ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax1.set_xlabel("Day count (from March 1st to March 22nd)")

ax1.set_ylabel("Confirmed Cases")



ax2.plot(dates_list_num, predicted_data)

ax2.plot(dates_list_num, np.log(real_data))

ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

ax2.set_xlabel("Day count (from March 1st to March 22nd)")

ax2.set_ylabel("Log Confirmed Cases")



plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))
ts = time.time()



day_start = 39

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

data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.exp(x))

data_pred.replace([np.inf, -np.inf], 0, inplace=True) 



get_submission(data_pred, 'Predicted_ConfirmedCases', 'Predicted_Fatalities')



print("Process finished in ", round(time.time() - ts, 2), " seconds")
ts = time.time()



# Set the dataframe where we will update the predictions

data_pred2 = data[data.ForecastId != -1][['Country_Region', 'Province_State', 'Day_num', 'ForecastId']]

data_pred2['Predicted_ConfirmedCases'] = [0]*len(data_pred2)

data_pred2['Predicted_Fatalities'] = [0]*len(data_pred2)

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

                data_pred2.loc[((data_pred2['Country_Region']==c) & (data_pred2['Province_State']==p)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

                data_pred2.loc[((data_pred2['Country_Region']==c) & (data_pred2['Province_State']==p)), 'Predicted_Fatalities'] = [0]*how_many_days

                

            # Else run LinReg

            else: 

                data_cp = data2[(data2['Country_Region']==c) & (data2['Province_State']==p)]

                X_train, Y_train_1, Y_train_2, X_test = split_data(data_cp)

                model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

                model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

                data_pred2.loc[((data_pred2['Country_Region']==c) & (data_pred2['Province_State']==p)), 'Predicted_ConfirmedCases'] = pred_1

                data_pred2.loc[((data_pred2['Country_Region']==c) & (data_pred2['Province_State']==p)), 'Predicted_Fatalities'] = pred_2



    # No Province/State informed

    else:

        # Only fit starting from the first confirmed case in the country

        train_countries_no0 = data.loc[(data['Country_Region']==c) & (data.ConfirmedCases!=0) & (data.ForecastId==-1)]

        test_countries_no0 = data.loc[(data['Country_Region']==c) &  (data.ForecastId!=-1)]

        data2 = pd.concat([train_countries_no0, test_countries_no0])



        # If there are no previous cases, predict 0

        if len(train_countries_no0) == 0:

            data_pred2.loc[((data_pred2['Country_Region']==c)), 'Predicted_ConfirmedCases'] = [0]*how_many_days

            data_pred2.loc[((data_pred2['Country_Region']==c)), 'Predicted_Fatalities'] = [0]*how_many_days

        

        # Else, run LinReg

        else:

            data_c = data2[(data2['Country_Region']==c)]

            X_train, Y_train_1, Y_train_2, X_test = split_data(data_c)

            model_1, pred_1 = lin_reg(X_train, Y_train_1, X_test)

            model_2, pred_2 = lin_reg(X_train, Y_train_2, X_test)

            data_pred2.loc[(data_pred2['Country_Region']==c), 'Predicted_ConfirmedCases'] = pred_1

            data_pred2.loc[(data_pred2['Country_Region']==c), 'Predicted_Fatalities'] = pred_2



# Aplly exponential transf. and clean potential infinites due to final numerical precision

data_pred2[['Predicted_ConfirmedCases', 'Predicted_Fatalities']] = data_pred2[['Predicted_ConfirmedCases', 'Predicted_Fatalities']].apply(lambda x: np.exp(x))

data_pred2.replace([np.inf, -np.inf], 0, inplace=True) 



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





def plot_real_vs_prediction_country(data, train, country_name, day_start, dates_list):



    # Select predictions from March 1st to March 25th

    predicted_data = data.loc[(data['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].ConfirmedCases

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['ConfirmedCases']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(dates_list_num, np.exp(predicted_data))

    ax1.plot(dates_list_num, real_data)

    ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March 1st)")

    ax1.set_ylabel("Confirmed Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log(real_data))

    ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax2.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax2.set_xlabel("Day count (starting on March 1st)")

    ax2.set_ylabel("Log Confirmed Cases")



    plt.suptitle(("ConfirmedCases predictions based on Log-Lineal Regression for "+country_name))

    

    

def plot_real_vs_prediction_country_fatalities(data, train, country_name, day_start, dates_list):



    # Select predictions from March 1st to March 25th

    predicted_data = data.loc[(data['Day_num'].isin(list(range(day_start, day_start+len(dates_list)))))].Fatalities

    real_data = train.loc[(train['Country_Region']==country_name) & (train['Date'].isin(dates_list))]['Fatalities']

    dates_list_num = list(range(0,len(dates_list)))



    # Plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,6))



    ax1.plot(dates_list_num, np.exp(predicted_data))

    ax1.plot(dates_list_num, real_data)

    ax1.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

    ax1.legend(['Predicted cases', 'Actual cases', 'Train-test split'], loc='upper left')

    ax1.set_xlabel("Day count (starting on March 1st)")

    ax1.set_ylabel("Fatalities Cases")



    ax2.plot(dates_list_num, predicted_data)

    ax2.plot(dates_list_num, np.log(real_data))

    ax2.axvline(17, linewidth=2, ls = ':', color='grey', alpha=0.5)

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

    data[filter_col] = data[filter_col].apply(lambda x: np.log(x))

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

    data[filter_col] = data[filter_col].apply(lambda x: np.log(x))

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

day_start = 35 

lag_size = 30



data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)

plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
ts = time.time()



# Inputs

country_name = "Italy"

day_start = 35 

lag_size = 30



data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)

plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
# Inputs

country_name = "Germany"

day_start = 35 

lag_size = 30



data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)

plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
# Inputs

country_name = "Albania"

day_start = 35 

lag_size = 30



data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)

plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
# Inputs

country_name = "Andorra"

day_start = 35 

lag_size = 30



data = lin_reg_with_lags_country(all_data, country_name, day_start, lag_size, country_dict)

plot_real_vs_prediction_country(data, train, country_name, 39, dates_list)

plot_real_vs_prediction_country_fatalities(data, train, country_name, 39, dates_list)
# Inputs

day_start = 35 

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

        

#get_submission(results_df, 'ConfirmedCases', 'Fatalities')

print("Complete process finished in ", time.time()-tp)
results_df_2 = results_df.copy()



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

            if len(tmp_index) < 30:

                # ConfirmedCases

                results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                                  (results_df_2['Province_State']==province_dict[province_name]) &

                                  (results_df_2['Day_num']>=day_num_test)), 'ConfirmedCases'] = data_pred.loc[((data_pred['Country_Region']==country_dict[country_name]) & 

                                  (data_pred['Province_State']==province_dict[province_name]) & 

                                  (data_pred['Day_num']>=day_num_test)), 'Predicted_ConfirmedCases'].apply(lambda x: np.log(x))

                

                #Fatalities

                results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                                  (results_df_2['Province_State']==province_dict[province_name]) &

                                  (results_df_2['Day_num']>=day_num_test)), 'Fatalities'] = data_pred.loc[((data_pred['Country_Region']==country_dict[country_name]) & 

                                  (data_pred['Province_State']==province_dict[province_name]) & 

                                  (data_pred['Day_num']>=day_num_test)), 'Predicted_Fatalities'].apply(lambda x: np.log(x))



    # Countries without Province_State

    else:

        tmp_index = results_df_2.index[(results_df_2['Country_Region']==country_dict[country_name]) & 

                           (results_df_2['Day_num']<day_num_test) & 

                           (results_df_2['ConfirmedCases']!=0)]



        # When there is not enough data

        if len(tmp_index) < 30:

            

            #Confirmed Cases

            results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                            (results_df_2['Day_num']>=day_num_test)), 'ConfirmedCases'] = data_pred.loc[((data_pred['Country_Region']==country_dict[country_name]) & 

                            (data_pred['Day_num']>=day_num_test)), 'Predicted_ConfirmedCases'].apply(lambda x: np.log(x))

            

            results_df_2.loc[((results_df_2['Country_Region']==country_dict[country_name]) & 

                            (results_df_2['Day_num']>=day_num_test)), 'Fatalities'] = data_pred.loc[((data_pred['Country_Region']==country_dict[country_name]) & 

                            (data_pred['Day_num']>=day_num_test)), 'Predicted_Fatalities'].apply(lambda x: np.log(x))

            

results_df_2 = results_df_2.loc[results_df_2['Day_num']>=day_num_test]

# get_submission(results_df_2, 'ConfirmedCases', 'Fatalities')