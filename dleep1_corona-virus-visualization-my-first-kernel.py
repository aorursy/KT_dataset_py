# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

data.info()
data.corr()
plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(), annot =True, linewidth=.5, fmt= '.1f')

plt.show()
data.head()
data.columns


data.Confirmed.plot(kind="line", color="g", label="Confirmed", linewidth=2, alpha=0.5, grid=True, linestyle=":")

data.Recovered.plot(color="b", label="Recovered", linewidth=2, alpha=0.5, grid=True, linestyle="-.")

plt.legend(loc="upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Confirmed - Recovered Line Plot")

plt.show()
data.Confirmed.plot(kind="line", color="g", label="Confirmed", linewidth=2, alpha=0.5, grid=True, linestyle=":")

data.Deaths.plot(color="r", label="Deaths", linewidth=2, alpha=0.5, grid=True, linestyle="-.")

plt.legend(loc="upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Confirmed - Deaths Line Plot")

plt.show()
data.plot(kind="scatter", x="Confirmed", y="Deaths", alpha=0.5, color="blue")

plt.xlabel("Confirmed")

plt.ylabel("Deaths")

plt.title("Scatter Plot")

plt.show()