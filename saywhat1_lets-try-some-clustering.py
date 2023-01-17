import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
globalTempdf = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv", parse_dates = True, infer_datetime_format = True)
globalTempdf['dt'] = pd.to_datetime(globalTempdf.dt)
ZurichTempdf = globalTempdf[globalTempdf.City == 'Zurich'].dropna()
ZurichTempdf.set_index(ZurichTempdf.dt, inplace = True)
ZurichTempdf.tail()
zurByYeardf = ZurichTempdf['1/1/1900':'8/1/2013'].resample('A').dropna()
g = sns.tsplot(zurByYeardf.AverageTemperature)
plot = plt.plot(zurByYeardf.index, pd.stats.moments.ewma(zurByYeardf.AverageTemperature, com = 14.5))
plot1 = plt.plot(zurByYeardf.index, pd.stats.moments.ewma(zurByYeardf.AverageTemperatureUncertainty, com = 9.5))
