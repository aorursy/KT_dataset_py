# math operations
from numpy import inf

# time operations
from datetime import timedelta

# for numerical analyiss
import numpy as np

# to store and process data in dataframe
import pandas as pd

# basic visualization package
import matplotlib.pyplot as plt

# advanced ploting
import seaborn as sns

# interactive visualization
import plotly.express as px
import plotly.graph_objs as go
# import plotly.figure_factory as ff
#from plotly.subplots import make_subplots

# for offline ploting
from plotly.offline import plot, iplot, init_notebook_mode
init_notebook_mode(connected=True)

# hide warnings
import warnings
warnings.filterwarnings('ignore')
# list files
# ==========

!ls ../input/corona-virus-report
# Full data
# =========

full_table = pd.read_csv('../input/covid19-clean2/covid_19_clean_complete.csv')
full_table.rename(columns = {'Province/State': 'Province_State', 'Country/Region': 'Country_Region'})
# Deep dive into the DataFrame
# Examine DataFrame (object type, shape, columns, dtypes)
full_table.info()
# Extract Brazil Data
# ===================

brazil_df = full_table[full_table.Country_Region == 'Brazil']
brazil_df.head(10)
# Extract India Data
# ===================

india_df = full_table[full_table.Country_Region == 'India']
india_df.head(10)
# merge Brazil and India data

brazil_india_df = pd.concat([brazil_df, india_df])
brazil_india_df[brazil_india_df.Country_Region=='Brazil'].head(10)
brazil_india_df[brazil_india_df.Country_Region=='India'].head(10)
# plot Brazil and India Confirmed Cases
# =====================================

sns.lineplot(data=brazil_india_df, x="Date", y="Confirmed", hue="Country_Region")
# plot Brazil and India Deaths Cases
# ==================================
sns.lineplot(data=brazil_india_df, x="Date", y="Deaths", hue="Country_Region")
# plot Brazil and India Recovered Cases
# =====================================

sns.lineplot(data=brazil_india_df, x="Date", y="Recovered", hue="Country_Region")
# plot Brazil and India Active Cases
# ==================================

sns.lineplot(data=brazil_india_df, x="Date", y="Active", hue="Country_Region")
# Worldometer data
# ================

worldometer_data = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

# Replace missing values '' with NAN and then 0
# What are the alternatives? Drop or impute. Do they make sense in this context?
worldometer_data = worldometer_data.replace('', np.nan).fillna(0)
worldometer_data['Case Positivity'] = round(worldometer_data['TotalCases'] / worldometer_data['TotalTests'], 2)
worldometer_data['Case Fatality'] = round(worldometer_data['TotalDeaths'] / worldometer_data['TotalCases'], 2)

# Case Positivity is infinity when there is zero TotalTests due to division by zero
worldometer_data[worldometer_data["Case Positivity"] == inf] = 0

# Qcut is quantile cut. Here we specify three equally sized bins and label them low, medium, and high, respectively.
worldometer_data['Case Positivity Bin'] = pd.qcut(worldometer_data['Case Positivity'], q=3,
                                                  labels=["low", "medium", "high"])

# Population Structure
worldometer_pop_struc = pd.read_csv('../input/covid19-worldometer-snapshots-since-april-18/population_structure_by_age_per_contry.csv')

# Replace missing values with zeros
worldometer_pop_struc = worldometer_pop_struc.fillna(0)
# worldometer_pop_struc.info()

# Merge worldometer_data with worldometer_pop_struc
# Inner means keep only common key values in both datasets
worldometer_data = worldometer_data.merge(worldometer_pop_struc, how='inner', left_on='Country/Region',
                                          right_on='Country')

# Keep observations where column "Country/Region" is not 0
worldometer_data = worldometer_data[worldometer_data["Country/Region"] != 0]

# Inspect worldometer_data's metadata
worldometer_data.info()


# Inspect Data
# worldometer_data.info()
# worldometer_data.tail(20)
# worldometer_data["Case Positivity"].describe()

print("India", worldometer_data[worldometer_data.Country == 'India'][['Population','TotalCases', 'TotalDeaths', 'ActiveCases']])
print("Brazil", worldometer_data[worldometer_data.Country == 'Brazil'][['Population','TotalCases', 'TotalDeaths', 'ActiveCases']])