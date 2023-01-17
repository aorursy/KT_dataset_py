# Data Analysis

import numpy as np 

import pandas as pd 



# Data Visualisation 

import matplotlib.pyplot as plt 

import seaborn as sns
hedge_data = pd.read_csv('../input/python/edhec-hedgefundindices.csv',

                           header=0, index_col=0, parse_dates=True)

hedge_data.shape
# Lets look at the monthly data

hedge_data.head()
# Convert to percent 

hedge_data = hedge_data/100

hedge_data.index = hedge_data.index.to_period('M')

hedge_data
# Collect mean and median data 

pd.concat([hedge_data.mean(), hedge_data.median(), hedge_data.mean() > hedge_data.median()], 

          axis="columns")
# Make a skewness function 

def skewness(r):

    '''

        ARGS:

            Series or Dataframe

        

        RETURNS: 

            Float or a series data with the calculated skewness

    '''

    

    # Calculate the demeaned returns 

    demeaned_r = r - r.mean()

    

    # Use the population standard deviation, ddof=0

    sigma_r = r.std(ddof=0)

    

    # Calculate the expectation of the demeaned returns raised to the third power

    exp = (demeaned_r**3).mean()

    

    # Calcualte the skew

    skew = exp/sigma_r**3

    return skew
# Calculate the skewness and sort the returns 

skewness(hedge_data).sort_values()
# Using the stats library 

import scipy.stats as st

st.skew(hedge_data)
# Make a kurtosis function 

def kurtosis(r):

    '''

        ARGS:

            Series or Dataframe

        

        RETURNS: 

            Float or a series data with the calculated kurtosis

    '''

    

    # Calculate the demeaned returns 

    demeaned_r = r - r.mean()

    

    # Use the population standard deviation, ddof=0

    sigma_r = r.std(ddof=0)

    

    # Calculate the expectation of the demeaned returns raised to the third power

    exp = (demeaned_r**4).mean()

    

    # Calcualte the skew

    kurt = exp/sigma_r**4

    return kurt
# Calculate the kurtosis and sort the returns 

kurtosis(hedge_data).sort_values(ascending=False)
# Using the built in function 

st.jarque_bera(hedge_data['CTA Global'])
# Function to apply the Jarque Bera test and 

# return whether the hedge is normally distributed or not 

def is_normal(r, level=0.01):

    '''

        ARG

            Series data

        RETURN

            True, if hypothesis of normality is accepted, False otherwise 

    '''

    statistic, p_val = st.jarque_bera(r)

    return p_val > level
is_normal(hedge_data)
hedge_data.aggregate(is_normal)