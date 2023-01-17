# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
corruption_file = "/kaggle/input/corruption-index/index.csv"
df_corruption_index = pd.read_csv(corruption_file)
df_corruption_index = df_corruption_index.iloc[:,:8]
df_corruption_index.head()
# Used documentation from https://stackabuse.com/python-data-visualization-with-matplotlib/ to 

#resize the figure so we can see all the countries listed in a large horizontal bar graph
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 20
fig_size[1] = 40
plt.rcParams["figure.figsize"] = fig_size

#show bar graph
ax_corruption_bar_graph = df_corruption_index.plot(x = "Country", y = "Corruption Perceptions Index (CPI)", kind = 'barh')
# Return the figure size to something more managable for future plotting
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 10
plt.rcParams["figure.figsize"] = fig_size
poverty_stats_series_file = "/kaggle/input/poverty-and-equity-database/povstats-csv-zip-242-kb-/PovStatsSeries.csv"
poverty_stats_country_file = "/kaggle/input/poverty-and-equity-database/povstats-csv-zip-242-kb-/PovStatsCountry.csv"
poverty_stats_country_series_file = "/kaggle/input/poverty-and-equity-database/povstats-csv-zip-242-kb-/PovStatsCountry-Series.csv"

poverty_stats_data_file = "/kaggle/input/poverty-and-equity-database/povstats-csv-zip-242-kb-/PovStatsData.csv"


df = pd.read_csv(poverty_stats_data_file)
df.head()
df_indexed = df.set_index(["Country Code", "Indicator Code"])
df_indexed
df_grouped = df.groupby("Indicator Code")
df_grouped = df_grouped.get_group("SI.POV.GINI")
df_grouped = df_grouped.set_index(["Country Code"])
df_grouped
year_range = range(1974,2019,1)
country_codes = df_grouped.index
minimum_gini_series = pd.Series(index=country_codes, name="Minimum observed GINI index")
maximum_gini_series = pd.Series(index=country_codes, name="Maximum observed GINI index")
mean_gini_series = pd.Series(index=country_codes, name="Mean observed GINI index")
num_observations_series = pd.Series(index=country_codes, name="Number of estimations of GINI index")

for country in country_codes:
    
    #will be used to compute mean observed gini
    successful_gini_observations = 0
    total_gini = 0

    #conceptually, the Gini index ranges from 0 to 100, these are therefore conceptual extremes of minimum/maximum
    minimum_gini = 100
    maximum_gini = 0
    mean_gini = 0

    
    country_series = df_grouped.loc[country]
    for year in year_range:
        gini_this_year = country_series.loc[str(year)]
        if pd.notna(gini_this_year):
            successful_gini_observations = successful_gini_observations + 1
            total_gini = total_gini + gini_this_year
            if gini_this_year < minimum_gini:
                minimum_gini = gini_this_year
                #print(minimum_gini)
            if gini_this_year > maximum_gini:
                maximum_gini = gini_this_year
                #print(maximum_gini)
    
    if successful_gini_observations > 0:
        mean_gini = total_gini / successful_gini_observations

    minimum_gini_series.loc[country] = minimum_gini
    maximum_gini_series.loc[country] = maximum_gini
    mean_gini_series.loc[country] = mean_gini
    num_observations_series.loc[country] = int(successful_gini_observations)
#simplify the data frame now that we have summary statistics
df_grouped = df_grouped.iloc[:,:3]
df_grouped["Mean GINI"] = mean_gini_series
df_grouped["Min GINI"] = minimum_gini_series
df_grouped["Max GINI"] = maximum_gini_series
df_grouped["Number of observations"] = num_observations_series
df_grouped
#For some countries, we simply lack any helpful data about inequality. We can pick these out because Mean GINI is still 0.
for country in country_codes:
    row = df_grouped.loc[country]
    if row.loc["Number of observations"] == 0:
        df_grouped = df_grouped.drop([country])
df_grouped
df_merged = df_grouped.merge(right=df_corruption_index,how='inner',on='Country Code')
df_merged
import seaborn as sns
from sklearn import linear_model

linear_regressor = LinearRegression() 

x_corruption = df_merged["Corruption Perceptions Index (CPI)"]
y_gini = df_merged["Mean GINI"]

sns.lmplot(data = df_merged, x = "Corruption Perceptions Index (CPI)", y = "Mean GINI")
sns.lmplot(data = df_merged, x = "Corruption Perceptions Index (CPI)", y = "Min GINI")
sns.lmplot(data = df_merged, x = "Corruption Perceptions Index (CPI)", y = "Max GINI")