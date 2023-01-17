# Example notebook to display setup and usage of reddit-pennystocks-data
# Loading and Formatting data

# Modules
import numpy as np 
import pandas as pd 

# Kaggle Getting Data
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
# Loading in files
df = pd.read_csv("/kaggle/input/reddit-pennystock-data/postData.csv")
fData = pd.read_csv("/kaggle/input/reddit-pennystock-data/finData.csv")

# Function to format files from raw extract
def format(df, fData):
    
    # Fix index for post data
    df.index = df['Unnamed: 0'].astype(str)
    df.index.name = 'Index'
    df = df.drop(columns = 'Unnamed: 0')
    
    # Set string time as index for financial data
    fData.index = fData.Datetime
    fData = fData.drop(columns = 'Datetime')
    
    return df, fData

# Run function
df, fData = format(df, fData) 

print(str(len(df)) + " total posts")
# SIMPLE EXAMPLE: Getting performance of stocks based off of a condition
# Here, we will explore posts with external links
# Feel free to play around with it

# Declaring condition
condition = (df.linked == 1)

# Some other examples you can check out:
"""
condition = df.text.str.contains('buy')
condition = df.title.str.contains('!')
condition = df.flair == "Discussion"
condition = df.flair == "Catalyst"
Also, combinations of features!
"""

# Detailing posts
print(str(sum(condition)) + " posts satisfy condition")

# Function to get performance of of a boolean array
def getPerformance(df, fData, condition):

    # Getting indices that satisfy the condition
    idx = df[condition].index
    
    # Setting to conditional fData
    cfData = fData[idx]
    
    # Creating dict for profits at each hold time
    performance = {}
    # Initializing performance with lists for every 5 minute interval over the 
    for interval in np.linspace(5, 390, 78):
        performance[int(interval)] = []
        
    # Looping through posts
    for post in cfData.columns:
        
        # Getting performance list for that post
        perList = cfData[post]
        
        # Getting index where post was made
        # Include try-except for corner cases
        try:
            postIndex = np.where(np.isnan(perList)==False)[0][0]
        except:
            print("Missed Post")
            continue
        
        # Loop through performances
        for i in range(len(perList)):
            
            # If that performance is not nan
            if not np.isnan(perList[i]):
                
                # Get that value's location in the performance dict
                perKey = 5 * (i-2) + 5
                
                # Assign performance to that key
                # Include try-except for corner cases
                try:
                    performance[perKey].append(perList[i])
                except:
                    print("Missed Starter")
                    continue
    
    # Default all first performances to 0
    performance[0] = []
    performance[0].append(0)
    return performance                
        
# Run function        
performance = getPerformance(df, fData, condition)   
print("performance calculated")
# Displaying mean perfomance over time
from matplotlib import pyplot as plt

# List of times
x = list(performance.keys())

# Initializing performances
y = np.zeros(len(x))
keys = list(performance.keys())

# Getting means for each time range (must adjust for percent change... -.5 is not the same as +.5)
for i in range(len(keys)):
    y[i] = 100* (np.mean(1 + np.array(performance[keys[i]])) - 1)

# Plot chart
plt.plot(x,y)
plt.title("Simulated Performance")
plt.xlabel("Minutes Held")
plt.ylabel("Percent Change")