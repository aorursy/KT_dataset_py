# Data Analysis
import numpy as np 
import pandas as pd 

# Data Visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns
monthly_data = pd.read_csv('../input/python/Portfolios_Formed_on_ME_monthly_EW.csv',
                           header=0, index_col=0, parse_dates=True, na_values=-99.99)
monthly_data.shape
# Let's look at the data
monthly_data.head()
# Extract the data
returns = monthly_data[['Lo 10', 'Hi 10']]
returns.columns = ['SmallCap', 'LargeCap']
returns.head()
# Convert Returns to percentages 
returns = returns/100
returns
# Make a line plot of the returns
returns.plot.line()
returns.index = pd.to_datetime(returns.index, format="%Y%m")
returns
returns.index = returns.index.to_period('M')
returns
# Plot the data again 
returns.plot.line()
# Get the data in the year 1960 
returns['1960']
# Compute the wealth index by starting with 1000 dollars
# The starting value won't matter with drawdowns

wealth_index = 1000*(1+returns['LargeCap']).cumprod()
wealth_index.head()
# Plot the wealth index over time 
wealth_index.plot.line()
# Compute the previous peaks 
previous_peaks = wealth_index.cummax()
previous_peaks.head()
# Plot the previous peaks
previous_peaks.plot.line()
# Calculate the drawdown in percentage
drawdown = (wealth_index - previous_peaks)/previous_peaks
# Plot the drawdown 
drawdown.plot.line()
drawdown.head()
# Get the worst drawdown 
drawdown.min()
drawdown.idxmin()
# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {drawdown['1975':].min()} on {drawdown['1975':].idxmin()}")
# Get the worst drawdown in the 90s 
print(drawdown['1990':'1999'].min())
print(drawdown['1990':'1999'].idxmin())
# Combine Plots 
wealth_index.plot.line()
previous_peaks.plot.line()

# Make a drawdown function
def compute_drawdown(return_series: pd.Series):
    '''
        ARGS: 
            Takes in a series of returns
            
        RETURNS:
            Wealth index
            Previous Peaks 
            Percent Drawdowns            
    '''
    
    # Calculate the wealth previous peaks and drawdowns
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    
    # Create a dataframe 
    drawdown_data = pd.DataFrame({'Wealth': wealth_index, 
                                  'Peaks': previous_peaks,
                                  'Drawdown': drawdowns})
    return drawdown_data
# Get the data for small cap stocks 
small_cap_drawdowns = compute_drawdown(returns['SmallCap'])
small_cap_drawdowns
# Lets plot the wealth and the peaks 
small_cap_drawdowns[['Wealth', 'Peaks']].plot.line()
small_cap_drawdowns['Drawdown'].plot.line()
small_cap_drawdowns['Drawdown'].min()
small_cap_drawdowns['Drawdown'].idxmin()
# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {small_cap_drawdowns['Drawdown']['1975':].min()} on {small_cap_drawdowns['Drawdown']['1975':].idxmin()}")
# Get the worst drawdown since 1975
print(f"The worst drawdown since 1975 was {drawdown['1975':].min()} on {drawdown['1975':].idxmin()}")
