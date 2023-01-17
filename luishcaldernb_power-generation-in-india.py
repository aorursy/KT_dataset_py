import pandas as pd

import pandas_profiling as pp

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

!pip install -q pandas-profiling[notebook]
df1 = pd.read_csv('../input/daily-power-generation-in-india-20172020/file.csv', parse_dates = ['Date'], thousands = ',', decimal = '.')

df2 = pd.read_csv('../input/daily-power-generation-in-india-20172020/State_Region_corrected.csv')
df1.info()
df2.info()
##

# Data dictionary in df1:

#

# Thermal Generation Actual (in MU)    --> TG_A

# Thermal Generation Estimated (in MU) --> TG_E

# Nuclear Generation Actual (in MU)    --> NG_A

# Nuclear Generation Estimated (in MU) --> NG_E

# Hydro Generation Actual (in MU)      --> HG_A

# Hydro Generation Estimated (in MU)   --> HG_E



df1 = df1.rename(columns = {

    'Thermal Generation Actual (in MU)': 'TG_A', 'Thermal Generation Estimated (in MU)': 'TG_E',

    'Nuclear Generation Actual (in MU)': 'NG_A', 'Nuclear Generation Estimated (in MU)': 'NG_E',

    'Hydro Generation Actual (in MU)': 'HG_A', 'Hydro Generation Estimated (in MU)': 'HG_E'

    })



##

# Data dictionary in df2:

#

# State / Union territory (UT) --> State_UT

# Area (km2)                   --> Area_km2

# Region                       --> Region

# National Share (%)           --> Nat_share_%



df2 = df2.rename(columns = {

    'State / Union territory (UT)': 'State_UT',

    'Area (km2)': 'Area_km2', 

    'National Share (%)': 'Nat_share_%'

    })
df1.sample(n = 3)
df2.sample(n = 3)
pd.isna(df1).head(n = 4)

#It is observed that columns NG_A & NG_E (Nuclear Generation), show 'na' data as 'True':
df1 = df1.fillna(0.000)



# Number of non-missing values:



df1.count()
df1.describe(include = [np.number], percentiles = [0.10, 0.25, 0.5, 0.75, 0.9]).T
df1.memory_usage(deep = True)
df1.select_dtypes(include = ['object']).nunique()
df2.select_dtypes(include = ['object']).nunique()
df1['Region'] = df1['Region'].astype('category')

df2['Region'] = df2['Region'].astype('category')

df1.dtypes    # df2 also changed to 'category'
df1.memory_usage(deep = True)

# The amount of memory used went down in the 'Region' column.
df2['Region'] = df2['Region'].replace({'Northeastern': 'NorthEastern'})
df1.set_index('Date', inplace = True)

df1.head(n = 3)
df2.set_index('State_UT', inplace = True)

df2.head(n = 3)
#pp.ProfileReport(df1)

#Warning results:



#TG_E is highly correlated with TG_A	High correlation

#TG_A is highly correlated with TG_E	High correlation

#HG_E is highly correlated with HG_A	High correlation

#HG_A is highly correlated with HG_E	High correlation

#Date is uniformly distributed	Uniform

#Region is uniformly distributed	Uniform

#NG_A has 1857 (40.1%) Zeros

#NG_E has 1857 (40.1%) Zeros
# pp.ProfileReport(df2)

#Warning results:



#Nat_share_% is highly correlated with Area_km2	High correlation

#Area_km2 is highly correlated with Nat_share_%	High correlation

#State_UT has unique values	Unique

#Area_km2 has unique values	Unique
df3 = df1.groupby('Region')[['TG_A', 'TG_E', 'NG_A', 'NG_E', 'HG_A', 'HG_E']].sum()

df3.head(n=5)
g = sns.PairGrid(df1, height = 3,

    x_vars = ['TG_A', 'NG_A', 'HG_A'],

    y_vars = ['Region'])

g.map(sns.barplot)

g.fig.suptitle('India: Power Generation by Region, 2017-2020', y = 1.2)