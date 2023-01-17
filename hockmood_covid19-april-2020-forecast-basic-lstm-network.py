# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import datetime



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import MinMaxScaler



from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from keras.layers import Dropout

from keras.layers import RepeatVector

from keras.layers import TimeDistributed



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/train.csv')
display(train_df.head())

display(train_df.info())
print('We have', len(train_df.Country_Region.unique()), 'countries/regions in the dataset.')

print('We have', len(train_df.Province_State.unique()), 'provinces/states in the dataset.')
timetrend_sick = sns.lineplot(train_df['Date'], train_df['ConfirmedCases'])
timetrend_deceased = sns.lineplot(train_df['Date'], train_df['Fatalities'])
# Add a new column to be able to uniquely distinguish countries/regions



train_df['UniqueRegion'] = np.where(train_df['Province_State'].isna(), train_df['Country_Region'], train_df['Country_Region'] + ' - ' + train_df['Province_State'])
# Show trends for 10 countries with most confirmed cases



top10_most_cases = train_df.loc[train_df['Date'] == train_df['Date'].max()][['UniqueRegion','ConfirmedCases']].sort_values(by='ConfirmedCases', ascending=False).head(10)

top10_most_cases_df = train_df.loc[train_df['UniqueRegion'].isin(top10_most_cases['UniqueRegion'].values)]



main_df = pd.DataFrame()



for i, top10_country in enumerate (top10_most_cases_df['UniqueRegion'].unique()):

    if i == 0:

        main_df = top10_most_cases_df.loc[top10_most_cases_df['UniqueRegion'] == top10_country][['Date', 'ConfirmedCases']].sort_values(by='Date')

        main_df = main_df.rename({'ConfirmedCases': top10_country}, axis='columns')



    else:

        temp_df = top10_most_cases_df.loc[top10_most_cases_df['UniqueRegion'] == top10_country][['Date', 'ConfirmedCases']]

        temp_df = temp_df.rename({'ConfirmedCases': top10_country}, axis='columns')

        main_df = pd.merge(main_df, temp_df, on=['Date'])



main_df = main_df.set_index('Date')

main_df.plot(figsize=(20,10))
# Calculate number of new sick per day



unique_regions = np.sort(train_df['UniqueRegion'].unique())

train_df['SickPerDay'] = 0



baseline_length = len(train_df.loc[train_df['UniqueRegion'] == 'Afghanistan']) # Country chosen arbitrarily



for unique_region in unique_regions:

    len_country = len(train_df.loc[train_df['UniqueRegion'] == unique_region])

    len_diffs = len(train_df.loc[train_df['UniqueRegion'] == unique_region]['ConfirmedCases'].diff())

    if len_country > baseline_length or len_diffs > baseline_length:

        raise NameError('Too many rows for country {}'.format(unique_region))

    train_df['SickPerDay'].loc[(train_df['UniqueRegion'] == unique_region)] = train_df.loc[train_df['UniqueRegion'] == unique_region]['ConfirmedCases'].diff()

    

train_df['SickPerDay'] = train_df['SickPerDay'].fillna(0)



# Show an example

display(train_df.loc[train_df['UniqueRegion'] == 'Czechia'].tail())
# Transform main data into a horizontal dataframe



def transform_horizontally(input_df, value_column):



    horizontal_df = pd.DataFrame()



    for i, uniqueRegion in enumerate (unique_regions):

        if i == 0:

            horizontal_df = input_df.loc[input_df['UniqueRegion'] == uniqueRegion][['Date', value_column]].sort_values(by='Date')

            horizontal_df = horizontal_df.rename({value_column: uniqueRegion}, axis='columns')



        else:

            temp_df = input_df.loc[train_df['UniqueRegion'] == uniqueRegion][['Date', value_column]]

            temp_df = temp_df.rename({value_column: uniqueRegion}, axis='columns')

            horizontal_df = pd.merge(horizontal_df, temp_df, on=['Date'])

            

    return horizontal_df
confirmed_horizontal_df = transform_horizontally(train_df, 'ConfirmedCases').sort_values(by='Date')

fatalities_horizontal_df = transform_horizontally(train_df, 'Fatalities').sort_values(by='Date')





display(confirmed_horizontal_df.head())

display(confirmed_horizontal_df.shape)



display(fatalities_horizontal_df.head())

display(fatalities_horizontal_df.shape)
# Convert dataframes into numpy arrays



np_confirmed = confirmed_horizontal_df.drop(columns=['Date']).to_numpy()

np_fatalities = fatalities_horizontal_df.drop(columns=['Date']).to_numpy()
# Scale the values (better performance of LSTM)



scaler = MinMaxScaler(feature_range = (0, 1))



np_confirmed_scaled = scaler.fit_transform(np_confirmed)

np_fatalities_scaled = scaler.fit_transform(np_confirmed)
# Split a multivariate sequence into samples

# Credits to: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting



def split_sequences(sequences, n_steps_in, n_steps_out):

    X, y = list(), list()

    for i in range(len(sequences)):

        # find the end of this pattern

        end_ix = i + n_steps_in

        out_end_ix = end_ix + n_steps_out

        # check if we are beyond the dataset

        if out_end_ix > len(sequences):

            break

        # gather input and output parts of the pattern

        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]

        X.append(seq_x)

        y.append(seq_y)

    return np.array(X), np.array(y)
n_steps_in = 30

n_steps_out = 1



X_confirmed, y_confirmed = split_sequences(np_confirmed_scaled, n_steps_in, n_steps_out)

X_fatalities, y_fatalities = split_sequences(np_fatalities_scaled, n_steps_in, n_steps_out)
assert X_confirmed.shape == X_fatalities.shape

assert y_confirmed.shape == y_fatalities.shape



n_features = X_confirmed.shape[2]
# Define model for confirmed cases



model_confirmed = Sequential()

model_confirmed.add(LSTM(1000, activation='relu', input_shape=(n_steps_in, n_features)))

model_confirmed.add(RepeatVector(n_steps_out))

model_confirmed.add(LSTM(2000, activation='relu', return_sequences=True))

model_confirmed.add(Dropout(0.2))

model_confirmed.add(LSTM(2000, activation='relu', return_sequences=True))

model_confirmed.add(Dropout(0.2))

model_confirmed.add(LSTM(2000, activation='relu', return_sequences=True))

model_confirmed.add(Dropout(0.2))

model_confirmed.add(LSTM(1000, activation='relu', return_sequences=True))

model_confirmed.add(TimeDistributed(Dense(n_features)))

model_confirmed.compile(optimizer='adam', loss='mse')



model_confirmed.summary()
history_confirmed = model_confirmed.fit(X_confirmed, y_confirmed, epochs=200)
plt.plot(history_confirmed.history['loss'][30:])

plt.title('Confirmed cases loss')

plt.show()
X_confirmed_pred = np_confirmed_scaled[-n_steps_in-1:-n_steps_out].reshape((1, n_steps_in, n_features))

y_confirmed_pred = model_confirmed.predict(X_confirmed_pred)
# Display targets and predictions side by side



comparison_df = pd.DataFrame()

comparison_df['Target'] = list(np_confirmed[-1])

comparison_df['Prediction'] = [int(x) for x in scaler.inverse_transform(y_confirmed_pred[0])[0].astype(int)]

comparison_df
# Define model for confirmed cases



model_fatalities = Sequential()

model_fatalities.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))

model_fatalities.add(RepeatVector(n_steps_out))

model_fatalities.add(LSTM(100, activation='relu', return_sequences=True))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(LSTM(100, activation='relu', return_sequences=True))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(LSTM(100, activation='relu', return_sequences=True))

model_fatalities.add(Dropout(0.2))

model_fatalities.add(LSTM(100, activation='relu', return_sequences=True))

model_fatalities.add(TimeDistributed(Dense(n_features)))

model_fatalities.compile(optimizer='adam', loss='mse')



model_fatalities.summary()
history_fatalities = model_fatalities.fit(X_fatalities, y_fatalities, epochs=100)
plt.plot(history_fatalities.history['loss'])

plt.title('Confirmed cases loss')

plt.show()
X_fatalities_pred = np_fatalities_scaled[-n_steps_in-1:-n_steps_out].reshape((1, n_steps_in, n_features))

y_fatalities_pred = model_fatalities.predict(X_fatalities_pred)
# Display targets and predictions side by side



comparison_df = pd.DataFrame()

comparison_df['Target'] = list(np_fatalities[-1])

comparison_df['Prediction'] = [int(x) for x in scaler.inverse_transform(y_fatalities_pred[0])[0].astype(int)]

comparison_df
# Read test set



test_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/test.csv')

test_df['UniqueRegion'] = np.where(test_df['Province_State'].isna(), test_df['Country_Region'], test_df['Country_Region'] + ' - ' + test_df['Province_State'])
# Predict for the necessary number of days



num_days_to_predict = datetime.datetime.strptime(test_df['Date'].max(), '%Y-%m-%d') - datetime.datetime.strptime(train_df['Date'].max(), '%Y-%m-%d')

num_days_to_predict.days



# Copy the train set for the purposes of prediction

X_test_confirmed = np_confirmed_scaled.copy()

X_test_fatalities = np_fatalities_scaled.copy()



def predict_for_test_set(model, X_test):

    for day in range(num_days_to_predict.days):

        X_pred_temp = X_test[-n_steps_in:].reshape((1, n_steps_in, n_features))

        y_pred_temp = model.predict(X_pred_temp)

        X_test = np.append(X_test, y_pred_temp[0], axis=0)

    return X_test



X_test_confirmed = predict_for_test_set(model_confirmed, X_test_confirmed)

X_test_fatalities = predict_for_test_set(model_fatalities, X_test_fatalities)



assert X_test_confirmed.shape == X_test_fatalities.shape

print('We have', X_test_confirmed.shape[0], 'days after predicting.')
# Copy predicted values into test dataframe



test_final = pd.merge(test_df, train_df, how='left', on=['Date', 'UniqueRegion'])

X_test_confirmed_inversed = scaler.inverse_transform(X_test_confirmed)

X_test_fatalities_inversed = scaler.inverse_transform(X_test_fatalities)

dummy_df = pd.DataFrame(test_final.loc[(test_final['UniqueRegion'] == unique_regions[0]) & ((test_final['ConfirmedCases'].isna()) | (test_final['Fatalities'].isna()))].sort_values(by='Date')['Date'])



for i, unique_region in enumerate(unique_regions):

    df_temp = dummy_df.copy()

    assert len(X_test_confirmed_inversed[-num_days_to_predict.days:,i]) == len(df_temp)

    assert len(X_test_fatalities_inversed[-num_days_to_predict.days:,i]) == len(df_temp)

    df_temp['ConfirmedCasesTemp'] = X_test_confirmed_inversed[-num_days_to_predict.days:,i]

    df_temp['FatalitiesTemp'] = X_test_fatalities_inversed[-num_days_to_predict.days:,i]

    df_temp['UniqueRegion'] = unique_region

    test_final = pd.merge(test_final, df_temp, how='left', on=['Date', 'UniqueRegion'])

    try:

        test_final['ConfirmedCases'] = np.where((test_final['UniqueRegion'] == unique_region) & test_final['ConfirmedCases'].isna(), test_final['ConfirmedCasesTemp'], test_final['ConfirmedCases'])

        test_final['Fatalities'] = np.where((test_final['UniqueRegion'] == unique_region) & test_final['Fatalities'].isna(), test_final['FatalitiesTemp'], test_final['Fatalities'])

    except:

        test_final['ConfirmedCases'] = np.where((test_final['UniqueRegion'] == unique_region) & test_final['ConfirmedCases'].isna(), None, test_final['ConfirmedCases'])

        test_final['Fatalities'] = np.where((test_final['UniqueRegion'] == unique_region) & test_final['Fatalities'].isna(), None, test_final['Fatalities'])

        

    #display(test_final.head(50))

    if 'ConfirmedCasesTemp' in test_final.columns:

        test_final = test_final.drop(columns=['ConfirmedCasesTemp'])

    if 'FatalitiesTemp' in test_final.columns:

        test_final = test_final.drop(columns=['FatalitiesTemp'])

        





assert not test_final['ConfirmedCases'].isnull().values.any()

assert not test_final['Fatalities'].isnull().values.any()
test_final['ConfirmedCases'] = test_final['ConfirmedCases'].astype(int)

test_final['Fatalities'] = test_final['Fatalities'].astype(int)
test_final.loc[test_final['UniqueRegion'] == 'Czechia']
sub_sample_df = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-2/submission.csv')

sub_sample_df
submission_df = test_final[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission_df.to_csv('submission.csv', index=False)