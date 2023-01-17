# Importing Python Libraries

import pandas as pd

import matplotlib.pyplot as plt
# Performing Analytics to get Percentage of Success for each Main Category of Projects



# Importing the Kickstarter Dataset

ds = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")



# Store dataset without undefined and live projects from dataset

compProj = ds[(ds["state"] != "undefined") & (ds["state"] != "live")]



# Store the total number of projects per category

totProjCount = compProj["main_category"].value_counts()



# Store dataset with only successful projects

susProj = compProj[compProj["state"]=="successful"]



# Store number of successful projects per category

susProjCount = susProj["main_category"].value_counts()



# Store percentage of success for each category

percentSuccess = susProjCount/totProjCount*100



# Print Out Percentage of Success Results

print(percentSuccess)
# Line Graph

percentSuccess.plot()
# Bar Graph for main_category

percentSuccess.plot(kind='bar')
percentSuccess.plot(kind='pie')
percentSuccess.plot(kind='area')
#Horizontal Bar Graph

percentSuccess.plot(kind='barh')
# Horizontal Bar Graph with Inverted Y Axis

percentSuccess.plot(kind='barh').invert_yaxis()
# Changing Colors of Bars

percentSuccess.plot(kind='barh', color="red")
# Adding Percent of Failure - Extensions

percentFail = 100-percentSuccess



# Creating a new DataFrame to contain both information

newDF = pd.DataFrame({'percentSuccess': percentSuccess, 'percentFail':percentFail}, index=percentSuccess.index)

print(newDF)
# Plotting Bar Graph with Both Percent Success and Failure

newDF.plot(kind="bar")
# Plot stacked bar graph

newDF.plot(kind="bar", stacked=True)
# Graphing Both Percentage on Separate Graphs

newDF.plot(kind="bar", rot=0, subplots=True)