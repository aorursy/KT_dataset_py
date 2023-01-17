# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import datetime
import matplotlib.dates as mdates
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# The present example uses the following data set and it is based on the more complex notebook developed for that dataset.
# https://www.kaggle.com/manualrg/spanish-electricity-market-demand-gen-price
#
# https://www.kaggle.com/manualrg/daily-electricity-demand-forecast-machine-learning

path = "/kaggle/input/spanish-electricity-market-demand-gen-price/spain_energy_market.csv"
data = pd.read_csv(path, sep=",", parse_dates=["datetime"])
data = data[data["name"]=="Demanda programada PBF total"]#.set_index("datetime")
data["date"] = data["datetime"].dt.date
data.set_index("date", inplace=True)
data = data[["value"]]
data = data.asfreq("D")
data = data.rename(columns={"value": "energy"})
data.info()
data[0:6]

# write your code here
fig, ax = plt.subplots()
plt.plot(data)
# call the plot method on data and put the title
plt.title('Energy Demand')
# change the y label
plt.ylabel('MWh')
plt.xlabel('date')
# show the plot
plt.legend(['energy'],loc=2)
plt.xlim(datetime.date(2014,1,1),datetime.date(2018,12,31))
myFmt = mdates.DateFormatter('%Y')
ax.xaxis.set_major_formatter(myFmt)
ax.xaxis.set_major_locator(plt.MaxNLocator(7))
plt.show()
# Further practice here
