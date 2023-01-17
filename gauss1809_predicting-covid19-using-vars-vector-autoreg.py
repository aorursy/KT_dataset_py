### IMPORTS



from datetime import datetime, timedelta



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import matplotlib

%matplotlib inline

matplotlib.rcParams['figure.figsize'] = [16, 8]

import seaborn as sns



from statsmodels.graphics.tsaplots import plot_acf

from statsmodels.tsa.api import VAR

from statsmodels.tsa.vector_ar.var_model import forecast



### DATA LOADING



train_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/train.csv')

train_data['Date'] = train_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())

test_data = pd.read_csv('/kaggle/input/covid19-global-forecasting-week-4/test.csv')

test_data['Date'] = test_data['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').date())

data = pd.concat([train_data,test_data])

ys = ["ConfirmedCases", "Fatalities"]

for y in ys:

    test_data[y] = np.nan
first_test_date = test_data['Date'].min()

print(first_test_date)
last_test_date = test_data['Date'].max()

test_dates = (last_test_date - first_test_date).days + 1

print(last_test_date)
### COUNTRIES & STATES

# countries = data['Country_Region'].unique()



# for country in countries:

#     print('-'*20)

#     print(country)

#     print('-'*20)

#     states = data.loc[data['Country_Region'] == country, 'Province_State'].unique()

#     for state in states:

#         print(state)
# PREDICTION FUNCTION (USING VAR)



def predict_cases_and_fatalities(country,data):

    print('-'*20)

    print(country)

    print('-'*20)

    states = data.loc[data['Country_Region'] == country, 'Province_State'].unique()

    for state in states:

        print('')

        print('State:')

        print(state)

        print('')

        if pd.isna(state):

            data_cs = data[(data['Country_Region'] == country) & (data['Province_State'].isna())].copy()

        else:

            data_cs = data[(data['Country_Region'] == country) & (data['Province_State'] == state)].copy()





        data_cs['Diff_ConfirmedCases'] = data_cs['ConfirmedCases'].diff().fillna(0)

        data_cs['Diff_Fatalities'] = data_cs['Fatalities'].diff().fillna(0)



        train_data_cs = data_cs[data_cs['Date'] < first_test_date].copy()

        test_data_cs = data_cs[(data_cs['Date'] >= first_test_date) & (data_cs['Date'] <= last_test_date) & (data_cs['ConfirmedCases'].isna())].copy()

        test_data_cs_expost = data_cs[(data_cs['Date'] >= first_test_date) & (data_cs['Date'] <= last_test_date) & ~(data_cs['ConfirmedCases'].isna())].copy()



        model = VAR(endog = train_data_cs[['Diff_ConfirmedCases','Diff_Fatalities']], dates = train_data_cs['Date'], freq='D')

        results = model.fit()

        predictions = results.forecast(results.endog[-results.k_ar:], test_dates)

        test_data_cs['Diff_ConfirmedCases'] = [np.max(x[0],0) for x in predictions]

        test_data_cs['Diff_Fatalities'] = [np.max(x[1],0) for x in predictions]

        test_data_cs['ConfirmedCases'] = test_data_cs['Diff_ConfirmedCases'].cumsum() + train_data_cs.loc[train_data_cs['Date'] == first_test_date - timedelta(days=1), 'ConfirmedCases'].iloc[0]

        test_data_cs['Fatalities'] = test_data_cs['Diff_Fatalities'].cumsum() + train_data_cs.loc[train_data_cs['Date'] == first_test_date - timedelta(days=1), 'Fatalities'].iloc[0]



        data_cs = pd.concat([train_data_cs,test_data_cs])



        if pd.isna(state):

            test_data.loc[(test_data['Country_Region'] == country) & (test_data['Province_State'].isna()),ys] = test_data_cs[ys]

        else:

            test_data.loc[(test_data['Country_Region'] == country) & (test_data['Province_State'] == state),ys] = test_data_cs[ys]



    return data_cs, test_data_cs_expost, state
### COMPUTE PREDICTIONS



countries = data['Country_Region'].unique()



for country in countries: 

    data_cs, test_data_cs_expost, state = predict_cases_and_fatalities(country,data)
# SUBMIT SOLUTION



submission = test_data[['ForecastId', 'ConfirmedCases', 'Fatalities']]

submission.to_csv('submission.csv', index=False)

print("Your submission was successfully saved!")
# TEST FOR SINGLE COUNTRIES

country = 'Denmark'



data_cs, test_data_cs_expost, state = predict_cases_and_fatalities(country,data)



for y in ys:

    fig, ax = plt.subplots()

    ax.plot(data_cs['Date'], data_cs[y])

    ax.plot(test_data_cs_expost['Date'], test_data_cs_expost[y])

    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))

    plt.title(f' {y} (country: {country}, state: {state})')