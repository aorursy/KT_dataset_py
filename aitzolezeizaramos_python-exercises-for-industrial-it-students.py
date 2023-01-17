# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load

import matplotlib.pyplot as plt # plotting

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

#data.info()
data[0:6]

# write your code here

# call the plot method on data and put the title



# change the y label



# show the plot

# Further practice here
