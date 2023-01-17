# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/Group8.csv")

data2 = pd.read_csv("/kaggle/input/Group9.csv")
data.head()
data.info()
data.isnull()
data.drop('field2', axis=True)
data.describe()
data.replace('NaN', 'False')
data["created_at"]=pd.to_datetime(data["created_at"])
data_1 = data.groupby(["created_at"]).agg({"field1":'sum'})

data_1.head()
print("Total number of field1: ", data_1["field1"].iloc[-1])
data.replace('nan', '0')
plt.figure(figsize=(12,6))

plt.plot(data["entry_id"],marker=" ",label="Time")

plt.xlabel("Date")

plt.ylabel("Value")

plt.xticks(rotation=90)

plt.title("RTC")

plt.legend()
data2.head()
data2.info()
data2.notnull()
data2.replace('NaN', 'False')
plt.figure(figsize=(12,6))

plt.plot(data2["entry_id"],marker=" ",label="Time")

plt.xlabel('created_at')

plt.ylabel("Value")

plt.xticks(rotation=90)

plt.title("RTC")

plt.legend()
fig, (ax1, ax2) = plt.subplots(1, 2,figsize=(20,6))

ax1.plot(data["entry_id"],label='Timer')

ax1.axhline(data["entry_id"].mean(),linestyle='--',color='black',label="Mean Timer")

ax1.set_title("Real Time")

ax1.set_ylabel("Value")

ax1.set_xlabel("Timestamp")

ax1.legend()

for tick in ax1.get_xticklabels():

    tick.set_rotation(90)

ax2.plot(data2["entry_id"],label="Timer")

ax2.axhline(data2["entry_id"].mean(),linestyle='--',color='black',label="Mean Timer")

ax2.set_title("Delay")

ax2.set_ylabel("Value")

ax2.set_xlabel("Timestamp")

ax2.legend()

for tick in ax2.get_xticklabels():

    tick.set_rotation(90)