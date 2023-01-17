%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
sns.set(color_codes=True)

def string_to_datetime(s, fmt='%Y-%m-%d'):
    if s != s:
        return np.nan
    year, month, day = s.split('-')
    try:  
        d = pd.datetime(int(year), int(month), int(day))
    except ValueError:
        d = pd.datetime(2017, 1, 1)
    d = min([max([d, pd.datetime(2013, 1, 1)]), pd.datetime(2017, 1, 1)])
    return d
# Read and transform dates
train = pd.read_csv("../input/train.csv", usecols=['date_time', 'is_booking', 'srch_ci', 'srch_co'],
                   parse_dates=['date_time'], nrows=10**6)
train['srch_ci'] = train['srch_ci'].apply(string_to_datetime)
train['srch_co'] = train['srch_co'].apply(string_to_datetime)
train.info()
train_bookings = train[train['is_booking'] == 1].drop('is_booking', axis=1)
train_clicks = train[train['is_booking'] == 0].drop('is_booking', axis=1)
del train
test_bookings = pd.read_csv("../input/test.csv", usecols=['date_time', 'srch_ci', 'srch_co'],
                   parse_dates=['date_time'], nrows=10**6)
test_bookings['srch_ci'] = test_bookings['srch_ci'].apply(string_to_datetime)
test_bookings['srch_co'] = test_bookings['srch_co'].apply(string_to_datetime)
test_bookings.info()
f = plt.figure()
plt.hist(train_bookings['date_time'].values, bins=100, alpha=0.5, normed=True, label='train bookings')
plt.hist(test_bookings['date_time'].values, bins=50, alpha=0.5, normed=True, label='test bookings')
plt.hist(train_clicks['date_time'].values, bins=100, alpha=0.5, normed=True, label='train clicks')
plt.title('Search time distribution')
plt.legend(loc='best')
f.savefig('SearchTime.png', dpi=300)
plt.show()

f = plt.figure()
plt.hist(train_bookings['srch_ci'].values, bins=100, alpha=0.5, normed=True, label='train bookings')
plt.hist(test_bookings['srch_ci'].dropna().values, bins=50, alpha=0.5, normed=True, label='test bookings')
plt.hist(train_clicks['srch_ci'].dropna().values, bins=100, alpha=0.5, normed=True, label='train clicks')
plt.title('Checkin time')
plt.legend(loc='best')
f.savefig('CheckinTime.png', dpi=300)
plt.show()
