import datetime as dt

from pandas_datareader import data
# We would like all available data from 01/01/2000 until Today.

start_date = '2011-01-01'

end_date =  dt.datetime.today()
# User pandas_reader.data.DataReader to load the desired data. As simple as that.

panel_data = data.DataReader('ASHOKLEY.NS', 'yahoo', start_date, end_date)
panel_data.head()