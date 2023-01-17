%matplotlib inline

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 


datas = pd.read_csv("../input/3-Airplane_Crashes_Since_1908.txt")

print(datas.head(2))
#datas.hist(column='Date')

#gb = datas.groupby("Date")
#print(gb.describe().sum())

dates_only = datas[['Date']]
grouped = dates_only.groupby(lambda x: dates_only['Date'][x].split('/')[2]).count()
#print(grouped)

plot_year = grouped.plot(kind='bar', grid=True, by=grouped, title="Plane crashed per year", figsize=(16, 8 ), )
plot_year.set_ylabel("Planes crashed")
plot_year.set_xlabel("Years")
plot_year.legend(["Years"])


