# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data_flight = pd.read_csv('../input/BrFlights2/BrFlights2.csv', 
                          sep = ',', header = 0, encoding = 'latin1')

data_flight.head()
data_flight.columns = [
    'flight_no', 'airline_company', 'route_type', 
    'departure_schedule', 'departure_actual', 'arrival_schedule', 'arrival_actual',
    'is_flown', 'just_code', 
    'airport_orig', 'city_orig', 'status_orig', 'country_orig',
    'airport_dest', 'city_dest', 'status_dest', 'country_dest',
    'lon_dest', 'lat_dest', 'lon_orig', 'lat_orig'
]

data_flight.dtypes
data_flight['is_flown'] = data_flight['is_flown'].astype('category')
data_flight['status_orig'] = data_flight['status_orig'].astype('category')
data_flight['status_dest'] = data_flight['status_dest'].astype('category')
data_flight['just_code'] = data_flight['just_code'].astype('category')
data_flight['route_type'] = data_flight['route_type'].astype('category')

from datetime import datetime
# 2016-01-30T08:58:00Z
data_flight['departure_actual'] = pd.to_datetime(data_flight['departure_actual'], format = '%Y-%m-%dT%H:%M:%SZ', errors = 'coerce')
import random
import datetime
import math

seed = 123
## define testing size
test_size = 100 #math.floor(math.sqrt(len(data_flight)))
## testing for string values
test_result_flight_no = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'flight_no']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['flight_no'] == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_flight_no[i] = time_delta.total_seconds() * 1000

test_result_flight_no_val = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'flight_no']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['flight_no'].values == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_flight_no_val[i] = time_delta.total_seconds() * 1000
## testing float values
test_result_lon_dest = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'lon_dest']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['lon_dest'] == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_lon_dest[i] = time_delta.total_seconds() * 1000

test_result_lon_dest_val = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'lon_dest']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['lon_dest'].values == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_lon_dest_val[i] = time_delta.total_seconds() * 1000
## testing categorical values
test_result_status_dest = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'status_dest']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['status_dest'] == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_status_dest[i] = time_delta.total_seconds() * 1000

test_result_status_dest_val = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_flight_no = data_flight.loc[random_ind, 'status_dest']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['status_dest'].values == rand_flight_no]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_status_dest_val[i] = time_delta.total_seconds() * 1000
## testing categorical values
test_result_departure_actual = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_departure_actual = data_flight.loc[random_ind, 'departure_actual']

    time_before = datetime.datetime.now()
    data_flight[data_flight['departure_actual'] == rand_departure_actual]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_departure_actual[i] = time_delta.total_seconds() * 1000

test_result_departure_actual_val = [None] * test_size
for i in range(0, test_size):
    random.seed(i)
    random_ind = random.randrange(0, len(data_flight))
    rand_departure_actual = data_flight.loc[random_ind, 'departure_actual']
    
    time_before = datetime.datetime.now()
    data_flight[data_flight['departure_actual'].values == rand_departure_actual]
    time_after = datetime.datetime.now()
    time_delta = time_after - time_before
    test_result_departure_actual_val[i] = time_delta.total_seconds() * 1000
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# set figure size
fig = plt.figure(figsize=(12,12))

def draw_the_plot(y, mean, label, color):
    plt.scatter(range(1, test_size + 1), y, label = 'time consumed' + label, marker = '.', color = color, alpha = .3)
    plt.plot(range(1, test_size + 1), [mean] * test_size, label = 'mean' + label , color = color, linestyle='--')

def draw_plot(title, y, mean, y_vals, mean_vals):
    plt.title(title)
    draw_the_plot(y, mean, '', 'red')
    draw_the_plot(y_vals, mean_vals, ' by val', 'blue')
    plt.legend(loc='upper right')
    plt.tight_layout()

# sub-plot 1
ax = plt.subplot(221)
draw_plot('string', test_result_flight_no, np.array(test_result_flight_no).mean(), 
          test_result_flight_no_val, np.array(test_result_flight_no_val).mean())   

# sub-plot 2
plt.subplot(222)
draw_plot('float', test_result_lon_dest, np.array(test_result_lon_dest).mean(), 
          test_result_lon_dest_val, np.array(test_result_lon_dest_val).mean())   

# sub-plot 3
plt.subplot(223)
draw_plot('categorical', test_result_status_dest, np.array(test_result_status_dest).mean(), 
          test_result_status_dest_val, np.array(test_result_status_dest_val).mean())

# sub-plot 4
plt.subplot(224)
draw_plot('datetime', test_result_departure_actual, np.array(test_result_departure_actual).mean(), 
          test_result_departure_actual_val, np.array(test_result_departure_actual_val).mean())

## show the plot
plt.show()
import seaborn as sns

def pinch(method, result_time):
    results = pd.DataFrame({
        'method': [method] * test_size,
        'time': result_time
    })
    return results

def draw_plot(title, y):
    plt.title(title)
    sns.swarmplot(data = pd.concat(results), x = 'method', y = 'time')
    x = plt.gca().axes.get_xlim()
    plt.plot(x, len(x) * [y[y['method'] == 'direct'].agg('mean')], sns.xkcd_rgb["blue"])
    plt.plot(x, len(x) * [y[y['method'] == 'by_val'].agg('mean')], sns.xkcd_rgb["orange"])
    plt.legend(loc='upper right')
    plt.tight_layout()

fig = plt.figure(figsize=(12,12))

ax = plt.subplot(221)
results = [pinch('direct', test_result_flight_no) , pinch('by_val', test_result_flight_no_val)]
draw_plot('string', pd.concat(results))


plt.subplot(222)
results = [pinch('direct', test_result_lon_dest) , pinch('by_val', test_result_lon_dest_val)]
draw_plot('float', pd.concat(results))

plt.subplot(223)
results = [pinch('direct', test_result_status_dest) , pinch('by_val', test_result_status_dest_val)]
draw_plot('categorical', pd.concat(results))

plt.subplot(224)
results = [pinch('direct', test_result_departure_actual) , pinch('by_val', test_result_departure_actual_val)]
draw_plot('datetime', pd.concat(results))
speedup_char = (np.mean(test_result_flight_no) - np.mean(test_result_flight_no_val)) / np.mean(test_result_flight_no) * 100
speedup_float = (np.mean(test_result_lon_dest) - np.mean(test_result_lon_dest_val)) / np.mean(test_result_lon_dest) * 100
speedup_cat = (np.mean(test_result_status_dest) - np.mean(test_result_status_dest_val)) / np.mean(test_result_status_dest) * 100
speedup_datetime = (np.mean(test_result_departure_actual) - np.mean(test_result_departure_actual_val)) / np.mean(test_result_departure_actual) * 100

print('string: {0:.2f}%'.format(speedup_char))
print('float: {0:.2f}%'.format(speedup_float))
print('categorical: {0:.2f}%'.format(speedup_cat))
print('datetime: {0:.2f}%'.format(speedup_datetime))