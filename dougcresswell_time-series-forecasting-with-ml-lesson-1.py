import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os # Miscellaneous operating system interfaces
fpath = '../input/daily-total-female-births-in-california-1959/daily-total-female-births-CA.csv'
df = pd.read_csv(fpath, names=['date', 'births'], header=0, parse_dates=['date'], index_col='date')
df.head()
df.tail(10)
df.info()
print('There are {} elements in the DataFrame'.format(df.size))
df.count()
# Select all date from December 1959 (month 12)
df['1959-12']
# Make a boolean mask. start_date and end_date can be datetime.datetimes, np.datetime64s, pd.Timestamps, or even datetime strings:
start_date = '1959-03-23'
end_date = '1959-04-02'
mask = (df.index >= start_date) & (df.index <= end_date)
df[mask]

# Alternate approaches using .loc method
# df.loc['1959-03-23':'1959-04-02']
# df.loc[start_date:end_date]
df.loc['1959-01-24']
# Create data frame
hourly = pd.DataFrame()

# Create random integer values using numpy.random
avg = df['births'].div(24).mean()
stdev = df['births'].div(24).std()
hourly['births'] = np.random.normal(loc=avg, scale=stdev, size=(24*365)).astype(int)

# Create datetimes, one per hour for each day in 1959
hourly.index = pd.date_range('1/1/1959', periods=(24*365), freq='H')

# Select all rows from a single date
hourly.loc['1959-07-04']
df.describe()
# Get a list 'filename' of all practice files
folder = '../input/time-series-practice-datasets/'
filename = os.listdir(folder)
filename
# Create a new DataFrame 'df', referencing a file name from the list
df = pd.read_csv(folder+filename[4])
df.head()
dfxl = pd.read_excel(folder+filename[3])
dfxl.head()
