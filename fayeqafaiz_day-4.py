import pandas as pd
#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
import seaborn as sns
#Import Library Pandas
import pandas as pd
#I am working in Windows environment
#Reading the dataset in a dataframe using Pandas
df = pd.read_csv("../input/plant-1-generation-data/Plant_1_Generation_Data.csv")  
print (df.head(3))  #Print first three observations
df.sort_values(by=['DC_POWER','AC_POWER'])
df.sort_values(by='DC_POWER', ascending=False)
#Iterate row wise data
for item in df.index: 
     print(df['SOURCE_KEY'][item], df['TOTAL_YIELD'][item]) 
#label based
for i in range(0,10) : 
   print(df.loc[i,"SOURCE_KEY"], df.loc[i,"TOTAL_YIELD"]) 

#index based
for i in range(0,10) : 
   print(df.iloc[i,2], df.iloc[i,5])
for index, row in df.iterrows(): 
    print (row["SOURCE_KEY"], row["TOTAL_YIELD"])
Remove_duplicate=df.drop_duplicates(['SOURCE_KEY'])
print(Remove_duplicate)
df.describe()
#missing values
df.isnull()
sns.distplot(df['DC_POWER']);
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
