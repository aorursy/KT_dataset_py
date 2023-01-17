# Importing Python Libraries

import pandas as pd

import matplotlib.pyplot as plt
# In this section we have already provided all of the code to calculate the percentage of success for each "main category" of Kickstarter Projects.

# We review each of these steps in detailed in our Data Playground activity. 



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