# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



a = pd.read_csv("../input/Indian petrolium product prices - Diesel.csv", parse_dates=True)

date = a['Date']

delhi =a["Delhi"]

mumbai = a["Mumbai"]

a['Date2'] = pd.to_datetime(a['Date'])

a['Year'] = a['Date2'].dt.year

a['Month'] = a['Date2'].dt.month

a['Day'] = a['Date2'].dt.day

a.head()

myplot= plt.plot(delhi,mumbai,a["Year"])

plt.xlabel("price in Rs")

plt.ylabel("year")

plt.show()