# In this section start by importing the data into a dataframe variable 

# Pandas for data manipulation 
import pandas as pd

# Seaborn for plotting
import seaborn as sns 

# MatplotLib Pyplot - For Math plots
import matplotlib.pyplot as plt

# Importing the deliveries.csv with .read_csv method
ipl_deliveries = pd.read_csv("../input/ipldata/deliveries.csv")

#.shape to quickly see the structure of the data 

ipl_deliveries.shape
ipl_deliveries.head(10) # 
#Importing csv into variable 
ipl_matches = pd.read_csv("../input/ipldata/matches.csv")

#Finding out the total number of rows and colums 
ipl_matches.shape

#See 10 records 
ipl_matches.head(10)
ipl_deliveries.corr()
ipl_matches.corr()
# Correlation map generated using seaborn

cor_ipl_deliveries = ipl_deliveries.corr()
plt.figure(figsize=(20,10)) # This determines size of the plot
sns.heatmap(cor_ipl_deliveries, annot=True)
plt.show()

# Correlation map generated using seaborn

ipl_matches

cor_ipl_matches = ipl_matches.corr()
plt.figure(figsize=(20,10)) # This determines size of the plot
sns.heatmap(cor_ipl_matches, annot=True)
plt.show()
