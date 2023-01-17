# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# data visualization libraries
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# QQ Plot
import scipy.stats as stats
import pylab as py

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
worldwide = pd.read_csv("/kaggle/input/covid19-worldwide-08112020/08-11-2020.csv", index_col=3)
worldwide.head()
worldwide.tail()
worldwide["Last_Update"] = pd.to_datetime(worldwide["Last_Update"])
worldwide["Last_Update"] = worldwide["Last_Update"].dt.strftime("%m-%d-%Y")
worldwide["Last_Update"].tail()
worldwide.shape
print("There are {0} rows and {1} columns in our csv file.".format(worldwide.shape[0], worldwide.shape[1]))
worldwide.info()
worldwide.describe()
# Describe the categorical features
worldwide.describe(include=object)
worldwide.rename(columns={"Long_": "Long", "Case-Fatality_Ratio": "Case_Fatality_Ratio"}, inplace=True)
worldwide.columns
worldwide["Last_Update"].value_counts()
worldwide.isnull().sum()
# Heatmap of null values
sns.heatmap(worldwide.isnull())
cont = ["Confirmed", "Deaths", "Recovered", "Active", "Incidence_Rate", "Case_Fatality_Ratio"]
worldwide[cont].boxplot(return_type="axes", figsize=(25,10))
sns.boxplot(x=worldwide["Recovered"])
sns.scatterplot(x="Recovered", y="Confirmed", data=worldwide)
# QQ Plot of Recovered 
stats.probplot(worldwide["Recovered"], dist="norm", plot=py)
py.show()
# Standard deviation of each column
worldwide.std(axis=0)
sns.boxplot(x=worldwide["Active"])
sns.scatterplot(x="Active", y="Confirmed", data=worldwide)
# QQ Plot of Recovered 
stats.probplot(worldwide["Active"], dist="norm", plot=py)
py.show()
# Remove the outlier from Recovered
worldwide = worldwide[worldwide["Recovered"] != worldwide["Recovered"].max()]
len(worldwide["Recovered"])
sns.boxplot(x=worldwide["Recovered"])
# Remove all rows with negative Active cases
worldwide = worldwide[worldwide["Active"] >= 0]
worldwide.shape
sns.boxplot(x=worldwide["Active"])
worldwide.shape
print("After removing the outliers, we now have {0} rows and {1} columns in our csv file.".format(worldwide.shape[0], worldwide.shape[1]))
# Read times series data
confirmed = pd.read_csv("../input/csse-covid-19-time-series/time_series_covid19_confirmed_US.csv")
confirmed.shape
confirmed.columns
confirmed.head()
deaths = pd.read_csv("../input/csse-covid-19-time-series/time_series_covid19_deaths_US.csv")
deaths.shape
deaths.columns
recovered = pd.read_csv("../input/csse-covid-19-time-series/time_series_covid19_recovered_global.csv")
recovered.head()
recovered = recovered[recovered["Country/Region"] == "US"]
recovered.tail()
recovered.columns
print("The number of columns in confirmed, deaths, and recovered dataframes are {0}, {1}, and {2} respectively.".format(confirmed.shape[1], deaths.shape[1], recovered.shape[1]))
identifiers = ["Province_State", "Country_Region", "Lat", "Long_"]
confirmed_melt = confirmed.melt(id_vars=identifiers, value_vars=confirmed.columns[11:], var_name="Date", value_name="Confirmed") 
confirmed_melt.tail()
confirmed_melt.info()
recovered_melt = recovered.melt(id_vars=recovered.columns[0:4], value_vars=recovered.columns[4:], var_name="Date", value_name="Recovered") 
recovered_melt.rename(columns={"Province/State":"Province_State", "Country/Region":"Country_Region", "Long":"Long_"}, inplace=True)
recovered_melt.head
deaths_melt = deaths.melt(id_vars=identifiers, value_vars=confirmed.columns[11:], var_name="Date", value_name="Deaths")
deaths_melt.tail()
print(confirmed_melt.shape)
print(recovered_melt.shape)
print(deaths_melt.shape)
# Lets replace NaNs with empty values in these three dataframes.
confirmed_melt = confirmed_melt.replace(np.nan, '', regex=True)
deaths_melt = deaths_melt.replace(np.nan, '', regex=True)
#recovered_melt = recovered_melt.replace(np.nan, '', regex=True)
# Merge confirmed_melt and deaths_melt first

merged = confirmed_melt.merge(
  right=deaths_melt, 
  how='left',
  on=['Province_State', 'Country_Region', 'Date', 'Lat', 'Long_']
)
# Merging full_table and recovered_df_long
merged = merged.merge(
  right=recovered_melt, 
  how='left',
  on=['Province_State', 'Country_Region', 'Date', 'Lat', 'Long_']
)

merged.head(10)