#Import everything, fix dates

import pandas as pd 

dateparse = lambda x: pd.datetime.strptime(x, '%m/%d/%Y %I:%M:%S %p')

dateparse2 = lambda x: pd.datetime.strptime(x, '%m/%d/%Y')



df = pd.read_csv('crime.csv', parse_dates=['Date'], date_parser=dateparse)

wx = pd.read_csv('ChicagoWX_Clean.csv', parse_dates=['date'], date_parser=dateparse2)

police = pd.read_csv('PD_Locations.csv')