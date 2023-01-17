import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



from matplotlib import pyplot as plt

import seaborn as sbn

%matplotlib inline
global_temperatures = pd.read_csv("../input/GlobalTemperatures.csv", infer_datetime_format=True, index_col='dt', parse_dates=['dt'])

print (global_temperatures.info())
global_temperatures[global_temperatures.index.year > 2000]['LandAverageTemperature'].plot(figsize=(13,7))
global_temperatures.groupby(global_temperatures.index.year)['LandAverageTemperature'].mean().plot(figsize=(13,7))
global_temperatures.groupby(global_temperatures.index.year)['LandAverageTemperatureUncertainty'].mean().plot(figsize=(13,7))