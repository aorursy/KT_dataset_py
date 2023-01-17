
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# Read all data files into dataframes
report = pd.read_csv('../input/report.csv', index_col='index')
glasses = pd.read_csv('../input/glasses.csv', index_col='DATE')
smartphone = pd.read_csv('../input/smartphone.csv', index_col='timestamp')
smartwatch = pd.read_csv('../input/smartwatch.csv', index_col='timestamp')

cdf = smartwatch.append(glasses.append(smartphone, sort=True), sort=True)


def catchNaN(x):
    try:
        eval(x)
        return True
    except:
        return False

# heart_rate = pd.DataFrame(smartwatch.loc[smartwatch['source'] == "heart_rate"])


#activity = pd.DataFrame(smartphone.loc[smartphone['source'] == "activity"])
# heart_rate['2017-06-29 1:00:12.0':'2017-06-29 2:00:12.0']['values'].map(clean).plot()
#heart_rate['2017-07-01':'2017-07-03']['values'].plot()
#reduced = pd.DataFrame([eval(x)[0] for x in heart_rate['values']])
# heart_rate['reduced'] = reduced



# Convert all sources to a column of values
sources = set(cdf['source'].tolist())
for s in sources:
    cdf[s] = cdf['values'].where(cdf['source'] == s)
cdf = cdf.drop(columns=['source', 'index'])
    

# Lambda functions for heart_rate data
heart_rate = lambda x: x if not catchNaN(x) else float(eval(x)[0])
heart_rate_accuracy = lambda x: x if not catchNaN(x) else int(eval(x)[1])

# Split heart rate data using lambda functions
cdf['heart_rate_accuracy'] = cdf['heart_rate'].map(heart_rate_accuracy)
cdf['heart_rate'] = cdf['heart_rate'].map(heart_rate)

# Lambda functions for 3-value measurements
valx = lambda x: x if not catchNaN(x) else float(eval(x)[0])
valy = lambda x: x if not catchNaN(x) else float(eval(x)[1])
valz = lambda x: x if not catchNaN(x) else float(eval(x)[2])

# Splitter for accelerometer
cdf['accelx'] = cdf['accelerometer'].map(valx)
cdf['accely'] = cdf['accelerometer'].map(valy)
cdf['accelz'] = cdf['accelerometer'].map(valz)
cdf = cdf.drop(columns=['accelerometer'])

# Lambda functions for any single integer and float
to_num = lambda x: x if not catchNaN(x) else float(eval(x)[0])

# Converter for steps
cdf['steps_counter'] = cdf['step_counter'].map(to_num)
cdf['steps_detector'] = cdf['step_detector'].map(to_num)
 
# Converter for battery
cdf['battery'] = cdf['battery'].map(to_num)

# Converter for pressure
cdf['pressure'] = cdf['pressure'].map(to_num)

cdf
