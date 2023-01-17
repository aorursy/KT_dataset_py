# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # graphs and plotting
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/dataanalytics-moreclean/DataAnalytics_clean.csv") #Load the clean data
df.head() #Take a peek at the dataframe
#These each give a pivot table comparing price estimates by state. However, High is sorted by the high estimates and Low is sorted by the low estimate
High = pd.pivot_table(df, values = ["high_estimate"], index = ["State"], aggfunc = np.mean).sort_values(by = "high_estimate")
Low = pd.pivot_table(df, values = ["low_estimate"], index = ["State"], aggfunc = np.mean).sort_values(by = "low_estimate")
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4)) # A Window for the plot to everything can actually be readable
High.plot.bar(ax = axes[0], color="orange", ylim = (40,115)) #Plotting the histogram for the high estimate
Low.plot.bar(ax = axes[1], ylim = (25,70)) #Plotting the histogram for the low estimate
#A pivot table to check the salary estimates of companies that have key competitors or not
High = pd.pivot_table(df, values = ["low_estimate","high_estimate"], index = ["Has_Competitor"], aggfunc = np.mean).sort_values(by = "high_estimate",ascending = False)
print(High) #Print the two pivot tables with some spacing between
#This data was pulled into a pivot table to segment it out for ease of creating the plot. 
Industry = pd.pivot_table(df, values = "high_estimate", index = ["Industry"], aggfunc = np.mean).sort_values(by = "high_estimate")
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (12,24)) # A Window for the plot to everything can actually be readable
Industry.plot.barh(ax = axes, color = "green") #Plotting the histogram, horizontal for increased readability
#Pivot tables for high and low estimates, relating each to company revenue
High = pd.pivot_table(df, values = ["high_estimate"], index = ["Revenue"], aggfunc = np.mean).sort_values(by = "high_estimate", ascending = False)
Low = pd.pivot_table(df, values = ["low_estimate"], index = ["Revenue"], aggfunc = np.mean).sort_values(by = "low_estimate", ascending = False)
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4)) # A Window for the plot to everything can actually be readable
High.plot.bar(ax = axes[0], color = "orange",ylim = (85,96)) #Plotting the high estimate histogram
Low.plot.bar(ax = axes[1], color = "Blue", ylim = (50,57)) #Plotting the low estimate histogram
#Pivot tables for high and low estimates, relating each to company size
High = pd.pivot_table(df, values = ["high_estimate"], index = ["Size"], aggfunc = np.mean).sort_values(by = "high_estimate", ascending = False)
Low = pd.pivot_table(df, values = ["low_estimate"], index = ["Size"], aggfunc = np.mean).sort_values(by = "low_estimate", ascending = False)
fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (12,4)) # A Window for the plot to everything can actually be readable
High.plot.bar(ax = axes[0], color = "orange",ylim = (85,96)) #Plotting the high estimate histogram
Low.plot.bar(ax = axes[1], color = "Blue", ylim = (50,57)) #Plotting the low estimate histogram
#A pivot table to check which companies were wordier
Verbose = pd.pivot_table(df, values = ["Description Length"], index = ["Size"], aggfunc = np.mean).sort_values(by = "Description Length",ascending = False)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (12,4)) # A Window for the plot to everything can actually be readable
Verbose.plot.bar(ax = axes, color = "green", ylim = (2000,4500)) #Plotting verbosity histogram
#A pivot table to compare rating to company size
Rating = pd.pivot_table(df, values = ["Rating"], index = ["Size"], aggfunc = np.mean).sort_values(by = "Rating",ascending = False)
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (12,4)) # A Window for the plot to everything can actually be readable
Rating.plot.bar(ax = axes, color = "green", ylim = (3,4)) #Plotting the rating histogram