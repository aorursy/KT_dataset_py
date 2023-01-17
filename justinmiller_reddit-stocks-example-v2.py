# Example use of Reddit Stock Data
# Simulating an automated trading algorithm based on reddit discourse and suggestions.
# Exploring and optimizing policies to maximize stability and profits.
# Modules
import numpy as np 
import pandas as pd 
import json
from matplotlib import pyplot as plt

# Kaggle Getting Data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Loading json as dict
# Runtime ~10 Seconds

with open('/kaggle/input/reddit-pennystock-data/saps.json') as f:
  data = json.load(f)

# Displaying Subreddits 
subList = list(data.keys())
print(subList)
# Reserializing Pandas DataFrames and Series
# Runtime ~ 15 Seconds

# For each subreddit
for sub in subList:
    data[sub]['raw']['postData'] = pd.DataFrame(data = np.array(data[sub]['raw']['postData']), columns = ['ticker','title','text','flair','unix'])
    
    # For each financial data type
    for finType in ['inter','intra']:
    
        # Get a list of post keys
        keyList = list(data[sub]['raw'][finType].keys())
    
        # For each keys, get the financial data for that post
        for key in keyList:
            data[sub]['raw'][finType][key] = np.array(data[sub]['raw'][finType][key])
            data[sub]['raw'][finType][key] = pd.Series(data[sub]['raw'][finType][key][:,1], index = data[sub]['raw'][finType][key][:,0])

# For the example, lets focus on r/stocks
# Feel free to explore all 7 subreddits in sublist
sub = subList[1]
print("The channel of focus is: " + sub)

# The intra and inter raw data include a key that matches with the row of postData from the postData matrix
# These keys can be found in the metadata (here we will focus only on interday trades)
# Feel free to play around with intraday trades, just swap 'inter' with 'intra' if otherwise.
keys = data[sub]['md']['inter']['keys']
print("The first 10 interday keys are:")
print(keys[:10])

# Show example post with post and financial data
# The linking index between post and finData is rowNum: that is the index of the post and also the index of the key used
# Feel free to play around with rowNum, explore some more of the 1500+ interday and 300+ intraday posts for each subreddits.
rowNum = 1500
print("Post Data:")
print(data[sub]['raw']['postData'].loc[rowNum])
print("Interday Data:")
print(data[sub]['raw']['inter'][keys[rowNum]])
# To run more in depth analysis on the data, we can iteratively loop through posts and profits
# To do this, we need to assign a time frame and a policy of which posts to include in a theoretical "trade"

# As an example policy, let's simulate our profits if we bought and held stocks labeled by the author as 'advice'
policy = data[sub]['raw']['postData'].flair == 'Advice'
newKeys = data[sub]['raw']['postData'].loc[policy].index.astype(str)
print(str(len(newKeys)) + " keys fit this policy in overall post data.")


# Initializing profit list
profitList = np.array([])

# For each key
for key in newKeys:
    
    # Get the most recent profit (indicating full hold)
    # Add try, except since some posts dont have interday data
    try:
        profit = data[sub]['raw']['inter'][key].iloc[-1]
        profitList = np.append(profitList, profit)
    except:
        pass
    
print(str(len(profitList)) + " of those keys have interday data.")
print("All profits from those holdings: ")
print(profitList)
print("Total profit of policy:")
print(np.mean(profitList))
# We can also explore the change in profit over time for such a policy
# To do so, we use the unix timestamps associated with each profit

# First, get all times listed in these keys
timeList = np.array([])

# Loop through keys
for key in newKeys:
    
    # Again, use try, except since some data do not have interday data
    try:
        times = data[sub]['raw']['inter'][key].index
        timeList = np.append(times, timeList)
    except:
        pass
    
# Make timelist unique
timeList = np.unique(timeList)
print("Plotting performance over " + str(len(timeList)) + " different timestamps:")

# Create mean profit time series 
meanProfit = pd.Series(index = timeList, data = np.zeros(len(timeList)))

# For each time
for time in meanProfit.index:
    
    profitList = np.array([])
    
    # Loop through keys
    for key in newKeys:

        # Again, use try, except since some data do not have interday data
        # Getting profits at that time
        try:
            profit = data[sub]['raw']['inter'][key].loc[time]
            profitList = np.append(profitList, profit)
            
        except:
            pass
        
    # Save mean profit to time series
    meanProfit.loc[time] = np.nanmean(profitList)
    
# Plot Results
plt.plot(meanProfit)
plt.title("Policy Performance over Time")
plt.ylabel("Cumulative Profit")
plt.xlabel("Time (UNIX)")

# While this is all for the example, I would reccomend exploring more complicated policies 
# that incorporate techniques like dynamic trades and inter-channel reccomendations.

# For more info:
# See https://github.com/justinmiller33/SAPS-public for data collection scripts.
# Questions, Comments or Lucrative Ideas? Contact miller.justi@northeastern.edu
# Live extraction of posts? See reddit's pushshift api.