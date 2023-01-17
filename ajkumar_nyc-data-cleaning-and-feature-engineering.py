# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from math import floor # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd;#for dataprocessing.

import matplotlib.pyplot as plt;

from IPython.display import display;

%matplotlib inline
def readData(filename):

    data = pd.read_csv(filename, parse_dates=[2,3]);

    display(data.head(3));

    return data;
data = readData('../input/train.csv');
display(data.isnull().sum());
print("Avaliable dataset is from", data['pickup_datetime'].min(), "to", data['pickup_datetime'].max());

print("Total rows in dataset is",len(data));
data.boxplot(column=['trip_duration']);
data = data[data.trip_duration < 20000];

data.boxplot(column=['trip_duration']);
data = data[data.trip_duration < 2000];

data.boxplot(column=['trip_duration']);
data.boxplot(column=['pickup_longitude', 'dropoff_longitude']);
data = data[ data.pickup_longitude > -80];
data.boxplot(column=['pickup_longitude', 'dropoff_longitude']);
PICKUP_LOWER_LIMIT = data['pickup_longitude'] > -74;

PICKUP_UPPER_LIMIT = data['pickup_longitude'] < -73.92;

DROP_OFF_LOWER_LIMIT = data['dropoff_longitude'] > -74;

DROP_OFF_UPPER_LIMIT = data['dropoff_longitude'] < -73.92;

data = data[PICKUP_LOWER_LIMIT & PICKUP_UPPER_LIMIT & DROP_OFF_LOWER_LIMIT & DROP_OFF_UPPER_LIMIT];

pd.DataFrame.boxplot(data, column=['pickup_longitude', 'dropoff_longitude']);
data.boxplot(column=['pickup_latitude', 'dropoff_latitude']);
PICKUP_LOWER_LIMIT = data['pickup_latitude'] > 40.7;

PICKUP_UPPER_LIMIT = data['pickup_latitude'] < 40.82;

DROP_OFF_LOWER_LIMIT = data['dropoff_latitude'] > 40.7;

DROP_OFF_UPPER_LIMIT = data['dropoff_latitude'] < 40.82;

data = data[PICKUP_LOWER_LIMIT & PICKUP_UPPER_LIMIT & DROP_OFF_LOWER_LIMIT & DROP_OFF_UPPER_LIMIT];

pd.DataFrame.boxplot(data, column=['pickup_latitude', 'dropoff_latitude']);
print("Total samples in our dataset are ",len(data))
# Print data types to check type of each column and remaining column.

print(data.dtypes)
data.set_index('pickup_datetime', inplace=True, drop=False, append=False);
data['pickup_hour'] = data.index.hour
display(data.head(3));
data['isPeak'] = (data['pickup_hour'].isin([7,8,9,18,19,20,21])).astype(int);
display(data.head(2));
# Extracting date from datetime.
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar;



us_hol_cal = calendar();

holidays = us_hol_cal.holidays(start = data['pickup_datetime'].min(), end=data['pickup_datetime'].max());

data['isWeekend'] = (data['pickup_datetime'].dt.dayofweek > 5).astype(int);

data['isUSHoliday'] = (data['pickup_datetime'].isin(holidays)).astype(int);
data['isHoliday'] = data['isWeekend'] | data['isUSHoliday'];
display(data.head(3));
# Histogram of longitude and latitude values for dividing them in different categories

data.hist(column=['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'], figsize=(15,6));
latitude_step=0.02

latitude_transform = lambda lat: floor(lat/latitude_step)*latitude_step;

data['pickup_lat'] = data.pickup_latitude.map(latitude_transform);

data['dropoff_lat'] = data.dropoff_latitude.map(latitude_transform);



longitude_step = 0.01;

longitude_transform = lambda lon: floor(lon/longitude_step)*longitude_step;

data['pickup_lon'] = data.pickup_longitude.map(longitude_transform);

data['dropoff_lon'] = data.dropoff_longitude.map(longitude_transform);
display(data.head(2));
print(data.dtypes);
# update dtype from object to int.

data['store_and_fwd'] = (data['store_and_fwd_flag'].map(lambda value: value=='Y')).astype(int);
display(data.tail(1));
# drop redundant columns;

data.drop(['pickup_datetime', 'dropoff_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag', 'pickup_hour', 'isWeekend', 'isUSHoliday'], axis=1, inplace=True);
print(data.dtypes);
display(data.head(5));