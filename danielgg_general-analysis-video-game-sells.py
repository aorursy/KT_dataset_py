#Import libraries 

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

#Let's import the dataset

vg = pd.read_csv('../input/Video_Games_Sales_as_at_22_Dec_2016.csv')

#Check shape of the table

print(vg.shape)
#Let's take a look at the data

vg.head(5)
#Second Look at the Data =)

vg.describe()
#Check for NaN Values on the Dataset

vg.isnull().any()
#Check for format on Columns

vg.info()
#Best Selling Title

#Calculate Max Value for Global Sales

vg["Global_Sales"].max()

#Get Value on Global Sales and Show Column Name for it

vg[vg["Global_Sales"]== 82.530000000000001]["Name"]
# Max Selling by Year

vg.groupby(["Year_of_Release"])["Global_Sales"].max()

#How can I get the title too?
#Correlation between Rating and Revenue

vg[['Global_Sales','Critic_Score']].corr() 

#There is not correlation between score and sells
#How many Total Sells by Year

Total_Sells_By_Year = vg.groupby(["Year_of_Release"]).sum()["Global_Sales"]

Total_Sells_By_Year