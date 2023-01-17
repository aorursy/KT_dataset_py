# Starter code provided by Kaggle



#Importing Python libraries and storing with nicknames

import numpy as np

import pandas as pd



#Printing out the filenames in the included dataset

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Importing the Kickstarter Dataset

ds = pd.read_csv("/kaggle/input/kickstarter-projects/ks-projects-201801.csv")



# Quick snapshot of the dataset

ds.info()
# Display only information for two features

ds[["main_category","state"]]
# Get number of projects in each state

ds["state"].value_counts()
# Remove undefined and live projects from dataset

compProj = ds[(ds["state"] != "undefined") & (ds["state"] != "live")]



#Print out the number of projects in each state in filtered dataset

compProj["state"].value_counts()
# Print the total number of projects per category

totProjCount = compProj["main_category"].value_counts()

print(totProjCount)
#Filter dataset with only successful projects

susProj = compProj[compProj["state"]=="successful"]



#Print out number of successful projects per category

susProjCount = susProj["main_category"].value_counts()

print(susProjCount)
#Print percentage of success for each category

susProjCount/totProjCount*100
# Sort percentage of success in Ascending Order

percentSuccess = susProjCount/totProjCount*100

percentSuccess.sort_values()