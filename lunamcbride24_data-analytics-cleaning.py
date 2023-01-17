# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as plt # graphs and plotting
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/data-analyst-jobs/DataAnalyst.csv") #Pull in the DataAnalyst CSV
df.rename(columns = {"Unnamed: 0":"Index"}, inplace = True) #Replace the weird name for the index column
df.head() #View the dataframe
values = df["Salary Estimate"] #Puts the salary estimate 

#For loop to check for null values
for i in range(0,len(df.index)):
    if values[i] == "-1": #If the estimate is null
        df.drop(df.index[i],inplace=True) #Drop the line with the null value
df.reset_index(drop=True, inplace=True) #Reset indecies so we can still loop through
df.head() #Opens up the dataframe to take a peek
estimate = df["Salary Estimate"].apply(lambda x: x.split("(")[0]) #Lambda function to remove the (Glassdoor est.) from the row
estimate = estimate.replace({"\$" : "","K" : ""},regex = True) #Takes out the dollar sign and k off of the numbers
df["Salary Estimate"] = estimate #Puts the variable removing the fluff back into the dataframe
df.head() #Take a peek at the dataframe
df["low_estimate"] = df["Salary Estimate"].apply(lambda x: x.split("-")[0]).astype(int) #Gets the low salary estimate as a new column
df["high_estimate"] = df["Salary Estimate"].apply(lambda x: x.split("-")[1]).astype(int) #Gets the high salary estimate as a new column
df.head() #Take a peek at the dataframe
states = df["Location"].apply(lambda x: x.split(", ")[1]) #Pull the state from the location
df["State"] = states.apply(lambda x: "CO" if "Arapahoe" in x else x) #Arapahoe comes from "Greenwood Village, Arapahoe, CO" in the data
df["State"].value_counts() #Make sure there is no problems in the state data
df["Easy Apply"] = df["Easy Apply"].apply(lambda x: False if x=="-1" else True) #Fixed Easy Apply to True/False
df["Competitors"] = df["Competitors"].apply(lambda x: "None" if x=="-1" else x) #Changed competitors from -1 to None
df.head() #Take a peek at the data
df["Company Name"] = df["Company Name"].astype(str).apply(lambda x: x.split("\n")[0]) #Removed the rating off the end of the company name
df["Job Description"] = df["Job Description"].replace({"\n" : " "}, regex=True) #Turned the description into one long string, deleting \n characters
df.head() #Take a peek at the dataframe
df["Description Length"] = df["Job Description"].apply(lambda x: len(x)) #Get the length of the description using a lambda function
df.head() #Take a peek at the data
df["Has_Competitor"] = df["Competitors"].apply(lambda x: False if x=="None" else True) #Created a new row to see if the company has any competitors
print(df["Has_Competitor"].value_counts()) #Display the counts of the true or false
df.Industry = df.Industry.apply(lambda x: "Undefined" if x=="-1" else x) #another cleaning aspect it seems I overlooked, but fixed here
print(df["Industry"].value_counts()) #Display the counts of each industry
df["Revenue"] = df["Revenue"].apply(lambda x: "Unknown / Non-Applicable" if x=="-1" else x) #Fixing the null values in the revenue column
print(df["Revenue"].value_counts()) #Display the counts of each revenue class
df["Size"] = df["Size"].apply(lambda x: "Unknown" if x=="-1" else x) #Update null values in the Size area
print(df["Size"].value_counts()) #Display the counts of each size class
df = df.drop(["Index"], axis=1) #Drop the index column, as it is not really needed in analysis
df.head()
df.to_csv(r"./DataAnalytics_clean.csv", index = False) #Sends the data to a new csv