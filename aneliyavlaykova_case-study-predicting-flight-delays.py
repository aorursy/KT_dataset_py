import pandas as pd



df = pd.read_csv('../input/flights.csv', low_memory=False)

print('Dataframe dimensions:', df.shape)

#____________________________________________________________

# gives some infos on columns types and number of null values

tab_info=pd.DataFrame(df.dtypes).T.rename(index={0:'column type'})

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()).T.rename(index={0:'null values (nb)'}))

tab_info=tab_info.append(pd.DataFrame(df.isnull().sum()/df.shape[0]*100)

                         .T.rename(index={0:'null values (%)'}))

df.info()

#df = df.dropna()

#df = df[['YEAR', 'MONTH', 'DAY', 'AIRLINE', 'FLIGHT_NUMBER', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT' ]].dropna()

tab_info
airports = pd.read_csv("../input/airports.csv")

airports
airlines = pd.read_csv("../input/airlines.csv")

airlines
pd.set_option('display.max_columns', 500)

df.head(10)
df_Jan = df[df['MONTH'] == 1]

df_Jan
df_Jan['DATE'] = pd.to_datetime(df_Jan[['YEAR','MONTH', 'DAY']])
df_Jan_nocan = df_Jan[df_Jan['CANCELLED'] == 0]
df_Jan['CANCELLED'].unique()
df_Jan_nocan.isnull().sum().sort_values()
df_Jan_nocan[df_Jan_nocan['DEPARTURE_TIME'].isnull() == True]