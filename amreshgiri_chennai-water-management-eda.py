# Loading the required libraries 



import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 

import numpy as np

%matplotlib inline
# Loading the required datasets and using the 'Date' as index 



df_chennai_rs = pd.read_csv('../input/chennai_reservoir_levels.csv', parse_dates=['Date'], index_col='Date')

df_chennai_rainfall = pd.read_csv('../input/chennai_reservoir_rainfall.csv', parse_dates=['Date'], index_col='Date')
# Exploring the reservoirs dataset 



df_chennai_rs.head()
# Statistical facts about the reservoirs dataset 



df_chennai_rs.describe()
# More info about the reservoirs dataset like varible type, number of observations, etc



df_chennai_rs.info()
# Plotting for POONDI with date for reservoir quantity from 2004 to 2018



sns.set(rc={'figure.figsize':(15,7)})

poondi = df_chennai_rs[['POONDI']]

poondi.plot()

plt.show()
# Plotting for CHOLAVARAM with date for reservoir quantity from 2004 to 2018



sns.set(rc={'figure.figsize':(15,7)})

cholavaram = df_chennai_rs['CHOLAVARAM']

cholavaram.plot()

plt.show()
# Plotting for REDHILLS with date for reservoir quantity from 2004 to 2018



sns.set(rc={'figure.figsize':(15,7)})

redhills = df_chennai_rs[['REDHILLS']]

redhills.plot()

plt.show()
# Plotting for CHEMBARAMBAKKAM with date for reservoir quantity from 2004 to 2018



sns.set(rc={'figure.figsize':(15,7)})

chembarambakkam = df_chennai_rs[['CHEMBARAMBAKKAM']]

chembarambakkam.plot()

plt.show()
# Plotting all the reservoirs' storage (in million cubic feet) for years 2004 to 2018 in a single plot



sns.set(rc={'figure.figsize':(15,7)}) 

All = df_chennai_rs[['POONDI', 'CHOLAVARAM', 'CHEMBARAMBAKKAM', 'REDHILLS']]

All.plot()

plt.xticks(rotation=30)

plt.show()
# Statistical facts about the rainfall data 



df_chennai_rainfall.describe()
# More info about the rainfall data 



df_chennai_rainfall.info()
# Plotting rainfall data for all the places on a single plot to better understand the patterns



sns.set(rc={'figure.figsize':(15,10)}) 

All = df_chennai_rainfall[['POONDI', 'CHOLAVARAM', 'CHEMBARAMBAKKAM', 'REDHILLS']]

All.plot()

plt.xticks(rotation=30)

plt.show()
# Combining rainfall and reservoir data for each location 



# POONDI



df_POONDI = pd.DataFrame({

        'Rainfall': df_chennai_rainfall.POONDI * 5,

        'Reservoir': df_chennai_rs.POONDI

    })



# CHEMBARAMBAKKAM 



df_CHEMBARAMBAKKAM = pd.DataFrame({

        'Rainfall': df_chennai_rainfall.CHEMBARAMBAKKAM * 5,

        'Reservoir': df_chennai_rs.CHEMBARAMBAKKAM

    })



# CHOLAVARAM 



df_CHOLAVARAM = pd.DataFrame({

        'Rainfall': df_chennai_rainfall.CHOLAVARAM * 5,

        'Reservoir': df_chennai_rs.CHOLAVARAM

    })



# REDHILLS 



df_REDHILLS = pd.DataFrame({

        'Rainfall': df_chennai_rainfall.REDHILLS * 5,

        'Reservoir': df_chennai_rs.REDHILLS

    })
# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.

# CHEMBARAMBAKKAM



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_years_CHEMBARAMBAKKAM = df_CHEMBARAMBAKKAM['01-01-2015':'01-01-2018']

df_three_years_CHEMBARAMBAKKAM.plot()

plt.show()
# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.

# POONDI



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_years_POONDI = df_POONDI['01-01-2015':'01-01-2018']

df_three_years_POONDI.plot()

plt.show()
# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.

# REDHILLS



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_years_REDHILLS = df_REDHILLS['01-01-2015':'01-01-2018']

df_three_years_REDHILLS.plot()

plt.show()
# Plotting 3 years rainfall and reservoirs quantity data from 2015 to 2018 to better understand the patterns.

# CHOLAVARAM



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_years_CHOLAVARAM = df_CHOLAVARAM['01-01-2015':'01-01-2018']

df_three_years_CHOLAVARAM.plot()

plt.show()
# Plotting for time period between October 2015 to January 2016 for CHOLAVARAM



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_months_CHOLAVARAM = df_CHOLAVARAM['10-01-2015':'01-01-2016']

df_three_months_CHOLAVARAM.plot()

plt.show()
# Plotting for time period between October 2015 to January 2016 for POONDI



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_months_POONDI = df_POONDI['10-01-2015':'01-01-2016']

df_three_months_POONDI.plot()

plt.show()
# Plotting for time period between October 2015 to January 2016 for REDHILLS



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_months_REDHILLS = df_REDHILLS['10-01-2015':'01-01-2016']

df_three_months_REDHILLS.plot()

plt.show()
# Plotting for time period between October 2015 to January 2016 for CHEMBARABAKKAM 



sns.set(rc={'figure.figsize':(15,10)}) 

df_three_months_CHEMBARAMBAKKAM = df_CHEMBARAMBAKKAM['10-01-2015':'01-01-2016']

df_three_months_CHEMBARAMBAKKAM.plot()

plt.show()
# POONDI



df_POONDI.plot()

plt.show()
# CHEMBARAMBAKKAM 



df_CHEMBARAMBAKKAM.plot()

plt.show()
# CHOLAVARAM 



df_CHOLAVARAM.plot()

plt.show()
# REDHILLS 



df_REDHILLS.plot()

plt.show()