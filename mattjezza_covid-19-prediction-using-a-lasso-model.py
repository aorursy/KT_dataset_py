# Import libraries

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt



# Matplotlib converters

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()



# For handling time/date series

from datetime import datetime, time, timedelta, date



# For modeling

from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import Ridge

from sklearn.linear_model import Lasso



# Evaluation - RMSLE

from sklearn.metrics import mean_squared_log_error
LAST_PUB_TRAIN_DATE = datetime(2020, 3, 31)

FIRST_PUB_PRED_DATE = LAST_PUB_TRAIN_DATE + timedelta(1)

FIRST_STORED_PUB_PRED_DATE = datetime(2020, 4, 2)

LAST_PUB_PRED_DATE = datetime(2020, 4, 15)



LAST_PRIV_TRAIN_DATE = datetime(2020, 4, 9)

FIRST_PRIV_PRED_DATE = LAST_PRIV_TRAIN_DATE + timedelta(1)

FIRST_STORED_PRIV_PRED_DATE = datetime(2020, 4, 16)

LAST_PRIV_PRED_DATE = datetime(2020, 5, 14)
df_train = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

df_train.head()
df_test = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')
test_province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

print("Number of unique province_country groups in test file: {}".format(

    len(test_province_country_groups.groups.keys())))



province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

print("Number of unique province_country groups in training file: {}".format(

    len(province_country_groups.groups.keys())))
df_train['Date'] = pd.to_datetime(df_train['Date'], format='%Y-%m-%d')

df_test['Date'] = pd.to_datetime(df_test['Date'], format='%Y-%m-%d')
df_train.tail()
df_train.info()
df_train.isnull().sum()
df_train.loc[df_train['Province_State'].isnull(), 

             'Province_State'] = 'None'

df_train.isnull().sum()
province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

print("Number of unique province_country groups in training data: {}".format(

    len(province_country_groups.groups.keys())))
df_test.loc[df_test['Province_State'].isnull(), 

             'Province_State'] = 'None'

df_test.isnull().sum()
province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

confirmed_cases_list = []

for p_c in province_country_groups.groups.keys():

    #print(p_c)

    confirmed_cases_list = province_country_groups.get_group(p_c)['ConfirmedCases'].tolist()

    corrected = False

    for i in range(len(confirmed_cases_list)-1):

    #for i in range(len(confirmed_cases_list) - 1):

        # Detect a data error

        if (confirmed_cases_list[i] > confirmed_cases_list[i+1]):

            # Correct a one-off low data point

            try:

                if (confirmed_cases_list[i] <= confirmed_cases_list[i+2]):

                    print('Correcting low data point. Replaced {0} with {1} for country/province {2}'.format(

                      confirmed_cases_list[i+1],

                      confirmed_cases_list[i],

                      p_c))

                    confirmed_cases_list[i+1] = confirmed_cases_list[i]

                # Correct a one-off high data point

                else:

                    if (confirmed_cases_list[i-1] <= confirmed_cases_list[i+1]):

                        print('Correcting high data point. Replaced {0} with {1} for country/province {2}'.format(

                          confirmed_cases_list[i],

                          confirmed_cases_list[i-1],

                          p_c))

                        confirmed_cases_list[i] = confirmed_cases_list[i-1]

                    else:

                        print('Not able to correct an erroneous point for for country/province {0} automatically'.format(p_c))

            # Where there is no data point at i+2, i.e. i is penultimate

            except IndexError:

                print('Correcting penultimate data point. Replaced {0} with {1} for country/province {2}'.format(

                      confirmed_cases_list[i+1],

                      confirmed_cases_list[i],

                      p_c))

                confirmed_cases_list[i+1] = confirmed_cases_list[i]

            corrected = True

    if corrected == True:

        print("Correcting for country/province {0}".format(p_c))

        df_train.loc[(df_train['Country_Region'] == p_c[1]) &

                     (df_train['Province_State'] == p_c[0]), 'ConfirmedCases'] = confirmed_cases_list
date_range = df_train['Date']

day_groups = df_train.groupby('Date')

latest = day_groups.get_group((max(date_range)))
worst_affected = latest.sort_values(by = 'ConfirmedCases', ascending = False).head(20)

worst_affected.drop(columns = ['Id', 'Date'], inplace=True)

worst_affected
worst_affected_locations = [worst_affected['Province_State'].iloc[i] if (worst_affected['Province_State'].iloc[i] != 'None') else

                    worst_affected['Country_Region'].iloc[i] for i in range(len(worst_affected))]

worst_affected_locations
plt.figure(figsize = (18, 9))

plt.bar(worst_affected_locations, worst_affected['ConfirmedCases'])

plt.title('Number of Confirmed Cases in the 20 Worst Affected Locations')

plt.ylabel('Number of confirmed cases')

plt.xticks(rotation='vertical')

plt.show()
plt.figure(figsize = (18, 9))

plt.bar(worst_affected_locations, worst_affected['Fatalities'])

plt.title('Number of Fatalities in the 20 Worst Affected Locations')

plt.ylabel('Number of fatalities')

plt.xticks(rotation='vertical')

plt.show()
province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])
uk_group = province_country_groups.get_group(('None', 'United Kingdom'))

ticks = list(uk_group['Date'])[0::1]

plt.figure(figsize=(18, 9))

plt.plot(uk_group['Date'], uk_group['ConfirmedCases'], label = 'Confirmed Cases')

plt.plot(uk_group['Date'], uk_group['Fatalities'], label = 'Fatalities')

plt.xticks(ticks, rotation='vertical')

plt.xlabel('Date')

plt.ylabel('Number of cases, number of fatalities')

plt.legend()

plt.title('Graph to show number of confirmed cases in the UK')

plt.show()
sk_group = province_country_groups.get_group(('None', 'Korea, South'))

ticks = list(sk_group['Date'])[0::1]

plt.figure(figsize=(18, 9))

plt.plot(uk_group['Date'], sk_group['ConfirmedCases'], label = 'Confirmed Cases')

plt.plot(uk_group['Date'], sk_group['Fatalities'], label = 'Fatalities')

plt.xticks(ticks, rotation='vertical')

plt.xlabel('Date')

plt.ylabel('Number of cases, number of fatalities')

plt.legend()

plt.title('Graph to show number of confirmed cases in South Korea')

plt.show()
df_train['SocDist'] = 0
socdist_dict = {('None', 'Italy'): datetime(2020, 3, 9),

                 ('None', 'Spain'): datetime(2020, 3, 14),

                 ('Hubei', 'China'): datetime(2020, 1, 23),

                 ('None', 'Germany'): datetime(2020, 3, 22),

                 ('New York', 'US'): datetime(2020, 3, 15),

                 ('None', 'France'): datetime(2020, 3, 17),

                 ('None', 'Iran'): datetime(2020, 12, 31),

                 ('None', 'United Kingdom'): datetime(2020, 3, 22),

                 ('New Jersey', 'US'): datetime(2020, 3, 15),

                 ('None', 'Switzerland'): datetime(2020, 3, 20),

                 ('None', 'Belgium'): datetime(2020, 3, 17),

                 ('None', 'Netherlands'): datetime(2020, 3, 23),

                 ('None', 'Turkey'): datetime(2020, 3, 18),

                 ('None', 'Korea, South'): datetime(2020, 2, 25),

                 ('None', 'Austria'): datetime(2020, 3, 16),

                 ('California', 'US'): datetime(2020, 3, 15),

                 ('Michigan', 'US'): datetime(2020, 3, 23),

                 ('None', 'Portugal'): datetime(2020, 3, 19),

                 ('Massachusetts', 'US'): datetime(2020, 12, 31),

                 ('Florida', 'US'): datetime(2020, 12, 31),

                 ('Louisiana', 'US'): datetime(2020, 3, 19),

                 ('Pennsylvania', 'US'): datetime(2020, 3, 19),

                 ('None', 'Brazil'): datetime(2020, 3, 19)

               }
province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

worst_affected_groups = worst_affected.groupby(['Province_State', 'Country_Region'])

for location in worst_affected_groups.groups.keys():

    socdist_list = []

    p_c = province_country_groups.get_group(location)

    for i in range(p_c.shape[0]):

        if (p_c.iloc[i]['Date'] - socdist_dict[location]).days > 0:

            socdist_list.append(1)

        else:

            socdist_list.append(0)

    

    df_train.loc[(df_train['Country_Region'] == location[1]) &

                 (df_train['Province_State'] == location[0]),

                 'SocDist'] = socdist_list
df_train.head()
df_train[df_train['SocDist']>0]
df_train['NewCases'] = 0

province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])



for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    

    # Confirmed cases

    df_train.loc[(df_train['Country_Region'] == p_c[1]) &

                 (df_train['Province_State'] == p_c[0]),

                 'NewCases'] = province_country_group['ConfirmedCases'].diff()



# Replace null values with 0.

df_train.loc[df_train['NewCases'].isnull(), 'NewCases'] = 0
df_train['NewFatalities'] = 0

province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])



for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    

    # Confirmed cases

    df_train.loc[(df_train['Country_Region'] == p_c[1]) &

                 (df_train['Province_State'] == p_c[0]),

                 'NewFatalities'] = province_country_group['Fatalities'].diff()



# Replace null values with 0.

df_train.loc[df_train['NewFatalities'].isnull(), 'NewFatalities'] = 0
df_train['ActiveCases'] = 0

province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])



for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    

    # Active cases (proxy value)

    df_train.loc[(df_train['Country_Region'] == p_c[1]) &

                 (df_train['Province_State'] == p_c[0]),

                 'ActiveCases'] = province_country_group['NewCases'].rolling(10, min_periods=1).sum()



# Replace null values with 0.

df_train.loc[df_train['ActiveCases'].isnull(), 'ActiveCases'] = 0
df_train.head()
all_training_dates = df_train['Date'].unique()

all_testing_dates = df_test['Date'].unique()

all_prediction_dates = np.setdiff1d(all_testing_dates, all_training_dates)

all_prediction_dates
template_data = {'Id': np.nan,

                 'Province_State': np.nan,

                 'Country_Region': np.nan,

                 'Date': all_prediction_dates,

                 'ConfirmedCases': np.nan,

                 'Fatalities': np.nan,

                 'SocDist': np.nan,

                 'NewCases': np.nan,

                 'NewFatalities': np.nan,

                 'ActiveCases': np.nan

                }

template_df = pd.DataFrame(data=template_data)

template_df.head()
df_train_ex = pd.DataFrame(columns = df_train.columns)

df_train_ex.head()
province_country_groups = df_train.groupby(['Province_State', 'Country_Region'])

for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    df_train_ex = df_train_ex.append(province_country_group)

    template_df['Province_State'] = p_c[0]

    template_df['Country_Region'] = p_c[1]

    df_train_ex = df_train_ex.append(template_df)

    

df_train_ex.sort_values(by=['Country_Region', 'Province_State'], inplace=True)

df_train_ex.reset_index(drop=True, inplace=True)
df_train_ex.head()
# Define the lag time window.

max_lag=15

min_lag=5
province_country_groups = df_train_ex.groupby(['Province_State', 'Country_Region'])



for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    for i in range (min_lag, max_lag):

        feat = 'ActiveCases-{}'.format(str(i))

        df_train_ex.loc[(df_train_ex['Country_Region'] == p_c[1]) &

                 (df_train_ex['Province_State'] == p_c[0]),

                 feat] = province_country_group['ActiveCases'].shift(i)



        feat = 'SocDist-{}'.format(str(i))

        df_train_ex.loc[(df_train_ex['Country_Region'] == p_c[1]) &

                 (df_train_ex['Province_State'] == p_c[0]),

                 feat] = province_country_group['SocDist'].shift(i)
df_train_ex.head(16)
province_country_groups = df_train_ex.groupby(['Province_State', 'Country_Region'])

to_drop = []

for p_c in list(province_country_groups.groups.keys()):

    province_country_group = province_country_groups.get_group(p_c)

    to_drop.extend(list(df_train_ex.loc[(df_train_ex['Country_Region'] == p_c[1]) &

         (df_train_ex['Province_State'] == p_c[0])].iloc[0:14].index))

    

df_train_ex.drop(index=to_drop, inplace=True)
df_train_ex_public = df_train_ex.copy()

df_train_ex_public.loc[df_train_ex_public['Date'] >= datetime(2020, 4, 1),

                       ['ConfirmedCases', 'Fatalities', 'NewCases', 'ActiveCases']] = np.nan
def train_model(X, y, min_rows, total_rows):

    """

    Function to train the prediction model.

    Returns trained models and some prediction/test pairs to check performance.

    

    Input parameters:

    X - training data

    y - output labels

    min_rows - minimum number of rows to train before attempting walk-forward

    total_rows - total number of data points to train



    Returns:

    model                - the trained prediction model

    poly_transform_model - the trained polynomial transform model

    predictions          - list of predicted values obtained via "walk forward"

    test_values          - list of test values

    """

    

    predictions = []

    test_values = []

    

    for i in range(min_rows, total_rows):

        #print("Training step {}".format(i-min_rows+1))

        X_train, X_test = X[0:i], X[i:i+1]

        y_train, y_test = y[0:i], y[i:i+1]

        poly_transform_model = PolynomialFeatures(degree=1).fit(X_train)

        X_train_t = poly_transform_model.transform(X_train)

        X_test_t = poly_transform_model.transform(X_test)

        model = Lasso(alpha=0.001, max_iter=200000).fit(X_train_t, y_train)

        y_pred = model.predict(X_test_t)

        predictions.append(int(y_pred))

        test_values.append(y_test)

    

    return model, poly_transform_model, predictions, test_values
# Set training data

training_features = []

for i in range(min_lag, max_lag):

    feat = 'ActiveCases-{}'.format(str(i))

    training_features.append(feat)

    feat = 'SocDist-{}'.format(str(i))

    training_features.append(feat)



X = df_train_ex_public[df_train_ex_public['Date'] <= LAST_PUB_TRAIN_DATE][training_features]

y = df_train_ex_public[df_train_ex_public['Date'] <= LAST_PUB_TRAIN_DATE]['NewCases'].to_list()



# Scaling

scaler = MinMaxScaler()

cases_scaler_model = scaler.fit(X)

X_scaled = cases_scaler_model.transform(X)



# Total number of rows

num_rows = X_scaled.shape[0]
public_cases_model, public_cases_poly_transform_model, predictions, test_values = train_model(X_scaled, y, int(num_rows*0.95), num_rows)
plt.figure(figsize=(18, 9))

plt.plot(list(np.arange(len(predictions)))[-1000:], predictions[-1000:], label = 'Predicted New Cases')

plt.plot(list(np.arange(len(predictions)))[-1000:], test_values[-1000:], label = 'Actual New Cases')

plt.xlabel('Index')

plt.ylabel('Predicted and Actual New Cases')

plt.legend()

plt.title('Quick Plot of Predicted and Actual New Cases')

plt.show()
# Set training data

training_features = []

for i in range(min_lag, max_lag):

    feat = 'ActiveCases-{}'.format(str(i))

    training_features.append(feat)

    feat = 'SocDist-{}'.format(str(i))

    training_features.append(feat)



X = df_train_ex_public[df_train_ex_public['Date'] <= LAST_PUB_TRAIN_DATE][training_features]

y = df_train_ex_public[df_train_ex_public['Date'] <= LAST_PUB_TRAIN_DATE]['NewFatalities'].to_list()



# Scaling

scaler = MinMaxScaler()

fatalities_scaler_model = scaler.fit(X)

X_scaled = fatalities_scaler_model.transform(X)



# Total number of rows

num_rows = X_scaled.shape[0]
public_fatalities_model, public_fatalities_poly_transform_model, predictions, test_values = train_model(X_scaled, y, int(num_rows*0.95), num_rows)
plt.figure(figsize=(18, 9))

plt.plot(list(np.arange(len(predictions)))[-1000:], predictions[-1000:], label = 'Predicted New Fatalities')

plt.plot(list(np.arange(len(predictions)))[-1000:], test_values[-1000:], label = 'Actual New Fatalities')

plt.xlabel('Index')

plt.ylabel('Predicted and Actual New Fatalities')

plt.legend()

plt.title('Quick Plot of Predicted and Actual New Fatalities')

plt.show()
def predict(df,

            latest, earliest, 

            cases_prediction_model,

            fatalities_prediction_model,

            cases_scaler_model = None,

            cases_poly_transform_model = None,

            fatalities_scaler_model = None,

            fatalities_poly_transform_model = None):

    """

    Function to make predictions of confirmed cases and fatalities from trained models.

    This function updates dataframe df as it goes along.

    

    Input parameters:

    df                              - dataframe to use for reading/writing

    latest                          - latest date to predict

    earliest                        - earliest date to predict

    cases_prediction_model          - trained model for predicting confirmed cases

    fatalities_prediction_model     - trained model for predicting fatalities

    cases_scaler_model              - trained scaler model to use (optional)

    cases_poly_transform_model      - trained polynomial model to use (optional)

    fatalities_scaler_model         - trained scaler model to use (optional)

    fatalities_poly_transform_model - trained polynomial model to use (optional)

    

    Returns:

    None

    """

    province_country_groups = df.groupby(['Province_State', 'Country_Region'])

    for p_c in list(province_country_groups.groups.keys()):

        province_country_group = province_country_groups.get_group(p_c)



        for i in range((latest-earliest).days + 1):

            prediction_date = earliest + timedelta(i)

            #print(prediction_date)

            X = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date)][training_features]



            if cases_scaler_model != None:

                X_cases_scaled = cases_scaler_model.transform(X)

            else:

                X_cases_scaled = X

            if cases_poly_transform_model != None:

                X_cases_scaled_poly = cases_poly_transform_model.transform(X_cases_scaled)

            else:

                X_cases_scaled_poly = X_cases_scaled



            y_pred = cases_prediction_model.predict(X_cases_scaled_poly)

            #print(y_pred)

            # The model sometimes produces negative values when the input features are low.

            # Correct these to 0 here.

            if y_pred < 0:

                y_pred = 0

            predicted_new_cases = int(y_pred)

            

            if fatalities_scaler_model != None:

                X_fatalities_scaled = fatalities_scaler_model.transform(X)

            else:

                X_fatalities_scaled = X

            if fatalities_poly_transform_model != None:

                X_fatalities_scaled_poly = fatalities_poly_transform_model.transform(X_fatalities_scaled)

            else:

                X_fatalities_scaled_poly = X_fatalities_scaled

            

            y_pred = fatalities_prediction_model.predict(X_fatalities_scaled_poly)

            #print(y_pred)

            # The model sometimes produces negative values when the input features are low.

            # Correct these to 0 here.

            if y_pred < 0:

                y_pred = 0

            predicted_new_fatalities = int(y_pred)

            

            # Write fatality prediction to data frame

            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'NewFatalities'] = predicted_new_fatalities

            

            # Update cumulative fatalities

            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'Fatalities'] = df[

                            (df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date-timedelta(1))

            ]['Fatalities'].values[0] + predicted_new_fatalities



            # Write case prediction to data frame

            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'NewCases'] = predicted_new_cases



            # Update cumulative confirmed cases

            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'ConfirmedCases'] = df[

                            (df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date-timedelta(1))

            ]['ConfirmedCases'].values[0] + predicted_new_cases



            # Update features according to new prediction

            # ActiveCases is the sum of NewCases in a rolling ten day window.

            # Hence subtract the value 10 days agao (which is lost from the ten day window)

            # and add the new predicted value.

            new_cases_ten_days_ago = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date-timedelta(10))]['NewCases'].values[0]



            updated_active_cases = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date-timedelta(1))

                                     ]['ActiveCases'].values[0] - new_cases_ten_days_ago + predicted_new_cases





            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'ActiveCases'] = updated_active_cases



            # SocDist

            # Assume this takes the same value as the day before

            # Unless after 2020-04-15, in which case assume all locations have value 1

            if prediction_date < datetime(2020, 4, 16):

                updated_socdist = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date-timedelta(1))]['SocDist'].values[0]

            else:

                updated_socdist = 1



            df.loc[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] == prediction_date),

                            'SocDist'] = updated_socdist



            # Lag features

            for i in range (min_lag, max_lag):

                feat = 'ActiveCases-{}'.format(str(i))

                if (prediction_date + timedelta(i)) <= latest:

                    df.loc[(df['Country_Region'] == p_c[1]) &

                         (df['Province_State'] == p_c[0]) &

                         (df['Date'] == prediction_date + timedelta(i)),

                         feat] = updated_active_cases

                feat = 'SocDist-{}'.format(str(i))

                if (prediction_date + timedelta(i)) <= latest:

                    df.loc[(df['Country_Region'] == p_c[1]) &

                         (df['Province_State'] == p_c[0]) &

                         (df['Date'] == prediction_date + timedelta(i)),

                         feat] = updated_socdist

predict(df_train_ex_public,

        LAST_PUB_PRED_DATE,

        FIRST_PUB_PRED_DATE,

        public_cases_model,

        public_fatalities_model,

        cases_scaler_model = cases_scaler_model,

        cases_poly_transform_model = public_cases_poly_transform_model,

        fatalities_scaler_model = fatalities_scaler_model,

        fatalities_poly_transform_model = public_fatalities_poly_transform_model)
def store_predictions(df, earliest, latest):

    """

    Function to store predictions.

    

    Input Parameters:

    df       - dataframe to read data values from

    earliest - earliest date to read/write

    latest   - latest date to read/write

    

    Returns:

    None

    """

    

    province_country_groups = df.groupby(['Province_State', 'Country_Region'])

    for p_c in list(province_country_groups.groups.keys()):

        province_country_group = province_country_groups.get_group(p_c)

        df_test.loc[(df_test['Country_Region'] == p_c[1]) &

                            (df_test['Province_State'] == p_c[0]) &

                            (df_test['Date'] >= earliest) &

                            (df_test['Date'] <= latest),

                            'ConfirmedCases'] = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] >= earliest) &

                            (df['Date'] <= latest)]['ConfirmedCases'].values

        

        df_test.loc[(df_test['Country_Region'] == p_c[1]) &

                            (df_test['Province_State'] == p_c[0]) &

                            (df_test['Date'] >= earliest) &

                            (df_test['Date'] <= latest),

                            'Fatalities'] = df[(df['Country_Region'] == p_c[1]) &

                            (df['Province_State'] == p_c[0]) &

                            (df['Date'] >= earliest) &

                            (df['Date'] <= latest)]['Fatalities'].values
store_predictions(df_train_ex_public, FIRST_STORED_PUB_PRED_DATE, LAST_PUB_PRED_DATE)
df_test.head()
# Set training data

training_features = []

for i in range(min_lag, max_lag):

    feat = 'ActiveCases-{}'.format(str(i))

    training_features.append(feat)

    feat = 'SocDist-{}'.format(str(i))

    training_features.append(feat)



X = df_train_ex[df_train_ex['Date'] <= LAST_PRIV_TRAIN_DATE][training_features]

y = df_train_ex[df_train_ex['Date'] <= LAST_PRIV_TRAIN_DATE]['NewCases'].to_list()



# Scaling

scaler = MinMaxScaler()

cases_scaler_model = scaler.fit(X)

X_scaled = cases_scaler_model.transform(X)



# Number of rows

num_rows = X_scaled.shape[0]
private_cases_model, private_cases_poly_transform_model, predictions, test_values = train_model(X_scaled, y, int(num_rows*0.95), num_rows)
plt.figure(figsize=(18, 9))

plt.plot(list(np.arange(len(predictions)))[-1000:], predictions[-1000:], label = 'Predicted New Cases')

plt.plot(list(np.arange(len(predictions)))[-1000:], test_values[-1000:], label = 'Actual New Cases')

plt.xlabel('Index')

plt.ylabel('Predicted and Actual New Cases')

plt.legend()

plt.title('Quick Plot of Predicted and Actual New Cases')

plt.show()
# Set training data

training_features = []

for i in range(min_lag, max_lag):

    feat = 'ActiveCases-{}'.format(str(i))

    training_features.append(feat)

    feat = 'SocDist-{}'.format(str(i))

    training_features.append(feat)



X = df_train_ex[df_train_ex['Date'] <= LAST_PRIV_TRAIN_DATE][training_features]

y = df_train_ex[df_train_ex['Date'] <= LAST_PRIV_TRAIN_DATE]['Fatalities'].to_list()



# Scaling

scaler = MinMaxScaler()

fatalities_scaler_model = scaler.fit(X)

X_scaled = fatalities_scaler_model.transform(X)



# Number of rows

num_rows = X_scaled.shape[0]
private_fatalities_model, private_fatalities_poly_transform_model, predictions, test_values = train_model(X_scaled, y, int(num_rows*0.95), num_rows)
plt.figure(figsize=(18, 9))

plt.plot(list(np.arange(len(predictions)))[-1000:], predictions[-1000:], label = 'Predicted New Fatalities')

plt.plot(list(np.arange(len(predictions)))[-1000:], test_values[-1000:], label = 'Actual New Fatalities')

plt.xlabel('Index')

plt.ylabel('Predicted and Actual New Fatalities')

plt.legend()

plt.title('Quick Plot of Predicted and Actual New Fatalities')

plt.show()
predict(df_train_ex,

        LAST_PRIV_PRED_DATE,

        FIRST_PRIV_PRED_DATE,

        private_cases_model,

        private_fatalities_model,

        cases_scaler_model = cases_scaler_model,

        cases_poly_transform_model = private_cases_poly_transform_model,

        fatalities_scaler_model = fatalities_scaler_model,

        fatalities_poly_transform_model = private_fatalities_poly_transform_model)
store_predictions(df_train_ex, FIRST_STORED_PRIV_PRED_DATE, LAST_PRIV_PRED_DATE)
df_test.to_csv('submission.csv', columns = ['ForecastId', 'ConfirmedCases', 'Fatalities'], index = False)
predicted_cases = df_test[(df_test['Date'] >= FIRST_STORED_PUB_PRED_DATE) &

                         (df_test['Date'] <= LAST_PRIV_TRAIN_DATE)]['ConfirmedCases'].values

predicted_fatalities = df_test[(df_test['Date'] >= FIRST_STORED_PUB_PRED_DATE) &

                         (df_test['Date'] <= LAST_PRIV_TRAIN_DATE)]['Fatalities'].values
test_cases = df_train[df_train['Date'] >= FIRST_STORED_PUB_PRED_DATE]['ConfirmedCases'].values

test_fatalities = df_train[df_train['Date'] >= FIRST_STORED_PUB_PRED_DATE]['Fatalities'].values
cases_rmsle = np.sqrt(mean_squared_log_error(test_cases, predicted_cases))

print("RMSLE on confirmed cases: {}".format(cases_rmsle))



fatalities_rmsle = np.sqrt(mean_squared_log_error(test_fatalities, predicted_fatalities))

print("RMSLE on fatalities: {}".format(fatalities_rmsle))



print("Overall (mean) RMSLE: {}".format((cases_rmsle+fatalities_rmsle)/2))
sample_locations = [('None', 'United Kingdom'), ('None', 'Korea, South'), ('None', 'Germany')]

test_p_c_groups = df_train.groupby(['Province_State', 'Country_Region'])

pred_p_c_groups = df_test.groupby(['Province_State', 'Country_Region'])

for loc in sample_locations:

    test_grp = test_p_c_groups.get_group(loc)

    pred_grp = pred_p_c_groups.get_group(loc)

    test = test_grp[test_grp['Date'] >= FIRST_STORED_PUB_PRED_DATE]

    pred = pred_grp[(pred_grp['Date'] >= FIRST_STORED_PUB_PRED_DATE) &

                     (pred_grp['Date'] <= LAST_PRIV_TRAIN_DATE)]

    plt.figure(figsize=(12, 6))

    plt.plot(test['Date'], test['ConfirmedCases'], label = 'True Confirmed Cases')

    plt.plot(pred['Date'], pred['ConfirmedCases'], label = 'Predicted Confirmed Cases')

    plt.xlabel('Date')

    plt.ylabel('Number of confirmed cases')

    plt.xticks(rotation='vertical')

    plt.legend()

    plt.title('Graph Comparing True and Predicted Confirmed Cases in {0}'.format(loc))

    plt.show()

    

    plt.figure(figsize=(12, 6))

    plt.plot(test['Date'], test['Fatalities'], label = 'True Fatalities')

    plt.plot(pred['Date'], pred['Fatalities'], label = 'Predicted Fatalities')

    plt.xlabel('Date')

    plt.ylabel('Number of fatalities')

    plt.xticks(rotation='vertical')

    plt.legend()

    plt.title('Graph Comparing True and Predicted Fatalities in {0}'.format(loc))

    plt.show()

    
sample_locations = [('None', 'United Kingdom'), ('New York', 'US'), ('None', 'Italy')]

pred_p_c_groups = df_test.groupby(['Province_State', 'Country_Region'])

for loc in sample_locations:

    pred_grp = pred_p_c_groups.get_group(loc)

    pred = pred_grp[(pred_grp['Date'] >= FIRST_STORED_PRIV_PRED_DATE) &

                     (pred_grp['Date'] <= LAST_PRIV_PRED_DATE)]

    plt.figure(figsize=(12, 6))

    plt.plot(pred['Date'], pred['ConfirmedCases'], label = 'Predicted Confirmed Cases')

    plt.xlabel('Date')

    plt.ylabel('Number of confirmed cases')

    plt.xticks(rotation='vertical')

    plt.legend()

    plt.title('Graph Showing Predicted Confirmed Cases in {0}'.format(loc))

    plt.show()

    

    plt.figure(figsize=(12, 6))

    plt.plot(pred['Date'], pred['Fatalities'], label = 'Predicted Fatalities')

    plt.xlabel('Date')

    plt.ylabel('Number of fatalities')

    plt.xticks(rotation='vertical')

    plt.legend()

    plt.title('Graph Showing Predicted Fatalities in {0}'.format(loc))

    plt.show()