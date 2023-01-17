# Import libraries 

import pandas as pd 

import numpy as np 



import matplotlib.pyplot as plt 

import seaborn as sns
# Load the data

hfi_data = pd.read_csv('../input/python/edhec-hedgefundindices.csv',

                           header=0, index_col=0, parse_dates=True)

hfi_data.shape
hfi_data.head()
# Convert to percentages 

hfi_data = hfi_data/100

hfi_data.index = hfi_data.index.to_period('M')

hfi_data.head()
# Calculate the standard deviation 

std = hfi_data.std(ddof=0)

std.sort_values(ascending=False)
# Calculate the standard deviation for returns which have negative values 

semi_std = hfi_data[hfi_data<0].std(ddof=0)

semi_std.sort_values(ascending=False)
comparison = pd.concat([std, semi_std], axis=1)

comparison.columns = ["Standard Deviation", "Semi-Deviation"]

comparison.plot.bar(title="Standard Deviation vs Semideviation")
np.percentile(hfi_data, 5, axis=0)
def var_historic(r, level=5):

    '''

    ARG

        r: Dataframe with the returns

        level: percentile level

    Returns 

        percentile for each column

    '''

    # Check the type of data

    if isinstance(r, pd.DataFrame):

        return r.aggregate(var_historic, level=level)

    

    elif isinstance(r, pd.Series):

        return -np.percentile(r, level)

    else: 

        raise TypeError("Expected r to be a series or dataframe")
var_historic(hfi_data, level=5) 
from scipy.stats import norm
# Compute the z score assuming the data is gaussian 

# Percent point function

z = norm.ppf(0.05)

z
# Compute the gaussian VaR

var_gauss = -(hfi_data.mean() + z*hfi_data.std(ddof=0))

var_gauss
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

    

    # Calculate the expectation of the demeaned returns raised to the fourth power

    exp = (demeaned_r**4).mean()

    

    # Calcualte the skew

    kurt = exp/sigma_r**4

    return kurt
# Update z 

k = kurtosis(hfi_data)

s = skewness(hfi_data)

z = norm.ppf(0.05)

z = (z + (z**2 - 1)*s/6 + (z**3 - 3*z)*(k-3)/24 - (2 * z**3 - 5*z)*(s**2)/36)
mcf_var = -(hfi_data.mean() + z*hfi_data.std(ddof=0))

mcf_var
# Calculate the skewness and kurtosis 

stats = pd.concat([s, k], axis=1)

stats.columns = ["Skewness", "Kurtosis"]

stats
# Compare all three by plotting 

results = [var_gauss, mcf_var, var_historic(hfi_data, level=5)]

comparison=pd.concat(results, axis=1)

comparison.columns = ["Gaussian", "Cornish-Fisher", "Historic"]

comparison


# Plot the comparison DataFrame

ax = comparison.plot.bar(title="Hedge Fund Indices: VaR")

ax.set_xlabel("Indices")

ax.set_ylabel("Value at Risk")
# Create a cvar function 

def cvar_historic(r, level=5):

    """

        Computes the conditional VaR of Series or DataFrame

    """

    

    if isinstance(r, pd.Series): 

        is_beyond = r <= -var_historic(r, level=5) 

        return -r[is_beyond].mean()

    elif isinstance(r, pd.DataFrame):

        return r.aggregate(cvar_historic, level=level)

    else: 

        raise TypeError("Expected a dataframe or series")
cvar_historic(hfi_data, level=5)