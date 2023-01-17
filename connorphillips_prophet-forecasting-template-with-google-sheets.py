# data manipulation

import pandas as pd

# mathetmatical functions

import numpy as np

# time series forecasting

from fbprophet import Prophet

# google sheet importation

import gspread

from oauth2client.service_account import ServiceAccountCredentials

# time manipulation

from datetime import datetime, timedelta

from dateutil.relativedelta import relativedelta

# calendar manipulation

import calendar

# outlier detection

import outlier_detection
# Google api scope

scope = ['https://spreadsheets.google.com/feeds',

         'https://www.googleapis.com/auth/drive']

# Oauth2 credentials

credentials = ServiceAccountCredentials.from_json_keyfile_name('*your-account-id*.json', scope)

# authorize Google account with gspreads

gc = gspread.authorize(credentials)
# google sheet variables

sheet_id = '*your-google-sheet-id'

sheet_raw_data = 'raw_data'

sheet_sessions = 'sessions'

sheet_leads = 'leads'

sheet_sals = 'sals'
# load google sheet by sheet id

sheet = gc.open_by_key(sheet_id).worksheet(sheet_raw_data)

# store sheet in array

df_raw = pd.DataFrame(sheet.get_all_values(),columns=sheet.get_all_values()[0])

# exclude header from row count

df_raw = df_raw.drop(df_raw.index[0])

# convert date column to datetime format

df_raw['date'] = pd.to_datetime(df_raw['date'])



print(df_raw.describe())
### Matplotlib



import matplotlib.pyplot as plt

%matplotlib inline



x = np.array(df_raw['date'])

y = np.array(df_raw['sessions'])

type(y)

print(y)



# Sessions

#plt.plot(x,y)

#plt.show()

#plt.title('Scatter Plot - Sessions')

#plt.xlabel('Date')

#plt.ylabel('Sessions')

#plt.show()
# convert 'sessions' to integer

df_raw['sessions'] = pd.to_numeric(df_raw['sessions']).round(0).astype(int)

# run through outlier function

df = df_raw.iloc[outlier_detection.reject_outliers_iqr(df_raw['sessions'])]



print(df.describe())
# retrieve last day of month

def getLastDay(specificDate):

    formattedDate = datetime.strptime(specificDate, '%m/%d/%y')

    return calendar.monthrange(formattedDate.year,formattedDate.month)[1]

# set dynamic date values

today = datetime.strftime(datetime.now(), '%m/%d/%y')

yesterday = datetime.strftime(datetime.now() - timedelta(1), '%m/%d/%y')

yesterday_gs_format = datetime.strftime(datetime.now() - timedelta(1), '%Y-%m-%d')



print("Today: " + today)

print("Yesterday: " + yesterday)

print("Yesterday Google Sheet Format: " + yesterday_gs_format)
# set custom date values

customStartDate = '1/1/18' #set in m/d/y format

customEndDate = yesterday #set in m/d/y format
# date range for CUSTOM sample data

dateRange = pd.date_range(customStartDate,customEndDate)



# set customRange to custom_df

data_set_df = df[df['date'].isin(dateRange)]



session_ds = data_set_df

lead_ds = data_set_df

sal_ds = data_set_df



print(data_set_df.head(2))

print(data_set_df.tail(2))
# set days remaining in month to forecast for CUSTOM sample data

last_day = getLastDay(customEndDate)

daysRemaining = last_day - datetime.strptime(customEndDate, '%m/%d/%y').day



print(last_day)

print(daysRemaining)
# set 'ts'

session_ds = session_ds.assign(ds = session_ds['date'])

# set 'y'

session_ds = session_ds.assign(y = session_ds['sessions'])



session_ds.head(1)
session_model = Prophet()

session_model.add_country_holidays(country_name='US')

session_model.fit(session_ds)
session_future = session_model.make_future_dataframe(periods=daysRemaining, freq='D')

session_future.tail()
session_forecast = session_model.predict(session_future)

session_model.plot(session_forecast)
# set 'ts'

lead_ds = lead_ds.assign(ds = lead_ds['date'])

# set 'y'

lead_ds = lead_ds.assign(y = lead_ds['leads'])



lead_ds.head(1)
lead_model = Prophet()

lead_model.fit(lead_ds)
lead_future = lead_model.make_future_dataframe(periods=daysRemaining, freq='D')

lead_future.tail()
lead_forecast = lead_model.predict(lead_future)

lead_model.plot(lead_forecast)
# set 'ts'

sal_ds = sal_ds.assign(ds = sal_ds['date'])

# set 'y'

sal_ds = sal_ds.assign(y = sal_ds['sals'])



sal_ds.head(1)
sal_model = Prophet()

sal_model.fit(sal_ds)
sal_future = sal_model.make_future_dataframe(periods=daysRemaining, freq='D')

sal_future.tail()
sal_forecast = sal_model.predict(sal_future)

sal_model.plot(sal_forecast)
sal_forecast.tail()
session_forecast_sheets = list(session_forecast['yhat'].tail(daysRemaining))

lead_forecast_sheets = list(lead_forecast['yhat'].tail(daysRemaining))

sal_forecast_sheets = list(sal_forecast['yhat'].tail(daysRemaining))

days_remaining_dates = session_forecast['ds'].tail(daysRemaining)

days_remaining_first_date = days_remaining_dates.iloc[0]

days_actual_last_date = days_remaining_first_date - timedelta(days=1)
# load google sheet by sheet id

#sheet = gc.open_by_key(sheet_id).worksheet(sheet_sessions)

# store sheet in array

#df_raw = pd.DataFrame(sheet.get_all_values(),columns=sheet.get_all_values()[0])

#column = sheet.col_values(1)
#cell_list = sheet.findall(str(days_actual_last_date.date()))
#lookup_col = cell_list[0].col

#lookup_row = cell_list[1].row + 1 #Adding one to the row lookup so forecasted values are appended to current day to end of month

#print(lookup_col)

#print(lookup_row)
#forecasting_range = sheet.range(lookup_row,lookup_col,31,lookup_col)

#print(forecasting_range)

#print(len(forecasting_range))
def forecastedValueImport(gid, gsheet, forecast, yesterday, days_in_month):

    sheet = gc.open_by_key(gid).worksheet(gsheet)

    # store sheet in array

    df_raw = pd.DataFrame(sheet.get_all_values(),columns=sheet.get_all_values()[0])

    #df_raw = pd.DataFrame(sheet.get_all_values())

    

    # set date column to column variable

    column = sheet.col_values(1)

    print(column)

    

    print(str(yesterday.date()))

    

    print(type(str(yesterday.date())))

    

    # Find cells containing date matching yesterday

    cell_list = sheet.findall(yesterday_gs_format)

    print("Yesterday Cell")

    print(cell_list)

    

    # Set sheet column and row values for the location of yesterday

    lookup_col = cell_list[0].col

    lookup_row = cell_list[1].row + 1 #Adding one to the row lookup so forecasted values are appended to current day to end of month

    print(lookup_col)

    print(lookup_row)

    

    # Set last row based on days in month

    if(days_in_month == 31):

        last_day_row = 32

    elif(days_in_month == 30):

        last_day_row = 31

    elif(days_in_month == 29):

        last_day_row = 30

    elif(days_in_month == 28):

        last_day_row = 29

    else:

        last_day_row = 30

    

    # Pull range from Google Sheets where forecasted values are going to be pushed to

    forecasting_range = sheet.range(lookup_row,lookup_col,last_day_row,lookup_col)

    

    print(len(forecasting_range))

    print(len(forecast))

    

    

    

    # Loop through the range setting the value of the cell to the forecasted values in the list

    for i in range(0,len(forecasting_range)):

        forecasting_range[i].value = forecast[i]

    

    # Update the Google Sheet with the forecasted values

    sheet.update_cells(forecasting_range)
forecastedValueImport(sheet_id, sheet_sessions, session_forecast_sheets, days_actual_last_date, last_day)
forecastedValueImport(sheet_id, sheet_leads, lead_forecast_sheets, days_actual_last_date, last_day)
forecastedValueImport(sheet_id, sheet_sals, sal_forecast_sheets, days_actual_last_date, last_day)