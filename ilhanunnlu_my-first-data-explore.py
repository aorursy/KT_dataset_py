# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/2015.csv')
data.info()
data.head(10)
data.corr()
liste = ["country", "region", "happiness_rank", "happiness_score", "standard_error", "economy", "family", "health", "freedom", "trust", "generosity", "dystopia_residual"]
data.columns = liste
print(data.columns)
data.happiness_score.plot(kind="line", color = "red", linewidth = 1, alpha = 1, grid = True, linestyle = ":", figsize = (10,10))
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Happiness Rank Line Plot")
plt.show()
data.plot(kind = "scatter", x = "health", y = "economy", alpha = 0.7, color = "red", figsize = (10,10))
plt.xlabel("health")
plt.ylabel("economy")
plt.title("Scatter Plot")
plt.show()
data.freedom.plot(kind = "hist", bins = 70, figsize = (10,10))
plt.title("Histogram")
plt.show()
west_europe = data[data.region == "Western Europe"]
print(west_europe.country)
for each in west_europe.country:
    print(each)

filtered_data = data[np.logical_and(data.economy > 1, data.region == "Western Europe")]
print(filtered_data.country)
