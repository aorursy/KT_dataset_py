# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# VARIABLE DESCRIPTIONS:

# UNID: ISO numeric country code (used by the United Nations)

# WBID: ISO alpha country code (used by the World Bank)

# SES: Socioeconomic status score (percentile) based on GDP per capita and

# educational attainment (n=174)

# country: Short country name

# year: Survey year

# SES: Socioeconomic status score (1-99) for each of 174 countries

# gdppc: GDP per capita: Single time-series (imputed)

# yrseduc: Completed years of education in the adult (15+) population

# popshare: Total population shares
# Load data

filename='/kaggle/input/globses/GLOB.SES.csv'

ses_df=pd.read_csv(filename,encoding='ISO-8859-1')

ses_df.head()
# Review N/As in dataframe via function

# Source: https://towardsdatascience.com/cleaning-missing-values-in-a-pandas-dataframe-a88b3d1a66bf



def assess_NA(data):

    """

    Returns a pandas dataframe denoting the total number of NA values and the percentage of NA values in each column.

    The column names are noted on the index.

    

    Parameters

    ----------

    data: dataframe

    """

    # pandas series denoting features and the sum of their null values

    null_sum = data.isnull().sum()# instantiate columns for missing data

    total = null_sum.sort_values(ascending=False)

    percent = ( ((null_sum / len(data.index))*100).round(2) ).sort_values(ascending=False)

    

    # concatenate along the columns to create the complete dataframe

    df_NA = pd.concat([total, percent], axis=1, keys=['Number of NA', 'Percent NA'])

    

    # drop rows that don't have any missing data; omit if you want to keep all rows

    df_NA = df_NA[ (df_NA.T != 0).any() ]

    

    return df_NA
# Review columns with N/As

assess_NA(ses_df)
# Replace NAs with mean values

ses_df['yrseduc'] = ses_df['yrseduc'].fillna( ses_df['yrseduc'].mean() )
# Explore data for cleaning

ses_df.describe()
from pandas.plotting import scatter_matrix



# Review correlation

scatter_matrix(ses_df,figsize=(15,20))

print(ses_df.corr())
# Analysis

# Question 1: How do the other variables correlate with the socioeconomic status?



corr = ses_df.corr()

corr["SES"].sort_values(ascending=False)
# Analysis

# Question 2: Which countries have experienced the most growth in their socioeconomic status, GDP and years of education to the present day?
# Gather most recent entires (2010) into a new dataframe

recent_df = ses_df.loc[ses_df["year"]==2010]

recent_df.head()
# Establish a function that to measure growth for different measures, between the most recent year and lowest value year



def measure_growth(measure):



    growth = []

    lowest = []



    for x in recent_df["country"]:

        low = ses_df[ses_df['country']==x].nsmallest(1, measure)[measure].values

        y = recent_df[measure].loc[recent_df['country'] == x].values - low

        growth.append(y)

        lowest.append(low)

        print(f"{x}: {y}")



    # Convert array to float

    growth = np.array(growth, dtype=float)

    lowest = np.array(lowest, dtype=float)



    # Add growth values into 2010 dataframe

    recent_df[f"Lowest {measure}"] = lowest

    recent_df[f"{measure} Growth"] = growth

# Run functions for the three measures

measure_growth("SES")

measure_growth("gdppc")

measure_growth("yrseduc")
# SES

recent_df.sort_values(by="SES Growth",ascending=False).head()
# GDP

recent_df.sort_values(by="gdppc Growth",ascending=False).head()
# Years of education

recent_df.sort_values(by="yrseduc Growth",ascending=False).head()