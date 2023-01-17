# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(os.path.join(dirname, filename))

df
df_namibia = df.loc[df["location"] == "Namibia"]

df_namibia = df_namibia[["total_cases", "total_deaths"]]

df_namibia
row_list = list(range(1,(df_namibia.shape[0])+1))

df_namibia["days"] = row_list

df_namibia
df_namibia.set_index(["days"], inplace = True, drop = True) 

df_namibia
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(17,5))

ax1 = fig.add_subplot(1,2,1)

ax2 = fig.add_subplot(1,2,2)



ax1.plot(df_namibia["total_cases"])

ax2.plot(df_namibia["total_deaths"])



ax1.set_title("Namibia covid-19 confirmed cases")

ax2.set_title("Namibia covid-19 deaths")



ax1.set_xlabel("Number of days")

ax2.set_xlabel("Number of days")



ax1.set_ylabel("Cases")

ax2.set_ylabel("Deaths")

df_namibia["death_rate"] = df_namibia["total_deaths"] / df_namibia["total_cases"] * 100

df_namibia["death_rate"].dropna()
import matplotlib.pyplot as plt

fig_dr = plt.figure(figsize=(25,5))

ax_dr = fig_dr.add_subplot(1,2,1)

ax_dr.plot(df_namibia["death_rate"])

ax_dr.set_title("Namibia covid-19 death rate")

ax_dr.set_xlabel("Number of days")

ax_dr.set_ylabel("Percentage")
