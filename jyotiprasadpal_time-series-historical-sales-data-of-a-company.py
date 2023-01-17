!pip install pmdarima
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from pmdarima import auto_arima

from sklearn.metrics import mean_squared_error, mean_absolute_error
data = pd.read_csv('../input/Historical_Data.csv')

data.head()
data.info()
data.Country_Code.unique()
data['Date'] = pd.to_datetime(data.Date.astype('str'), errors='raise')

data['Month'] = data['Date'].dt.month_name()

data['Year'] = data['Date'].dt.year



data['Country_Code'] = data.Country_Code.astype('category')

print(data.info(), end='\n\n')

print(data.head())
#1. Print number of days which sold more than 3 units.

grouped_df = data.groupby(['Date'], as_index=False)['Sold_Units'].sum()

print(grouped_df.head(5), end='\n\n')

print('Number of days which sold more than 3 units:', grouped_df.loc[grouped_df.Sold_Units >3, :].shape[0])
#2. Print sales of the country(FR) in the month of August

df_FR_Aug = data.loc[((data.Country_Code == 'FR') & (data.Month=='August')), :]

print('Sales of the country(FR) in the month of August:', np.sum(df_FR_Aug.Sold_Units))
#3. Print total units sold in the country(AT). 

df_AT = data.loc[data.Country_Code == 'AT', :]

print('Total units sold in the country(AT):', np.sum(df_AT.Sold_Units))
def preprocess(country):        

    #set observed = True because Country_Code is category column

    df = data.groupby(['Country_Code', 'Date'], as_index=False, observed=True)['Sold_Units'].sum()        

    df_country = df.loc[df.Country_Code==country, :].set_index(['Date', 'Country_Code']).unstack(fill_value=0).asfreq('D', fill_value=0).stack().sort_index(level=1).reset_index()

    df_country_sorted = df_country.sort_values(['Date'], ascending=True)

    return (country, df_country_sorted)



#preprocess data for each country

country_dfs = {}

for country in data.Country_Code.unique():

    country, df = preprocess(country)

    country_dfs[country] = df
print('Starting date of sale for ‘FR’:', country_dfs['FR'].loc[0, 'Date'])
df_AT = country_dfs['AT']

df_AT_non_selling = df_AT.loc[df_AT['Sold_Units'] == 0, :]

print("Number of non-selling days for the country('AT'):", df_AT_non_selling.shape[0])



fig, ax = plt.subplots(figsize=(25, 5))

sns.lineplot(x='Date', y='Sold_Units', data=df_AT, ax=ax)
def fit_predict_auto_arima(train, test):

    #fit the model using training data

    model = auto_arima(train['Sold_Units'], 

                       seasonal=False, 

                       stationary=False, 

                       trace=False, 

                       error_action='ignore', 

                       suppress_warnings=True,

                       random_state=1)

    model.fit(train['Sold_Units'])



    #predict using test data

    forecast = model.predict(n_periods=len(test))

    forecast = pd.DataFrame({'Date': test.Date, 'Prediction': forecast}, index = test.index)



    #calculate rmse

    rmse = np.round(np.sqrt(mean_squared_error(test['Sold_Units'], forecast['Prediction'])), 3)

    

    #calculate mean absolue error

    mae = np.round(mean_absolute_error(test['Sold_Units'], forecast['Prediction']), 3)



    #plot the predictions for test set       

    fig, ax = plt.subplots(figsize=(25, 5))

    ax.plot('Date', 'Sold_Units', data=train, label='Train')

    ax.plot('Date', 'Sold_Units', data=test, label='Valid')

    ax.plot('Date', 'Prediction', data=forecast, label='Prediction')

    ax.legend()

    ax.set_title(f'Sold units for country {country} on daily basis')

    

    return (model, rmse, mae)
#for each country fit and predict    

country_error_map = {}

country_models_map = {}

for country in data.Country_Code.unique():

    df_country = country_dfs[country]

    #divide data into train and test set.

    train = df_country.loc[:len(df_country)-10, :]

    test = df_country.loc[len(df_country)-10:, :]



    #fit and predict

    model, rmse, mae = fit_predict_auto_arima(train, test)

    print(f'For country {country}, RMSE={rmse}, MAE={mae}')

    country_error_map[country] = mae

    country_models_map[country] = model
#print the model summary of country AT

country_models_map['AT'].summary()