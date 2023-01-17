# Loading Libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Import needed packages to process data
import sklearn
import matplotlib as mpl
import matplotlib.pyplot as plt
# Loading Data Set

csv_path = "../input/chronic-disease/U.S._Chronic_Disease_Indicators.csv"
df_src=pd.read_csv(csv_path, low_memory=False)

# Visualizing first 5 lines of Data

df_src.head()

# Data Understanding

df_src.info()
# check completely filled features

(df_src.count()/len(df_src))*100
# defining an indexed function that explores the features of each columnt

def df_values(df):
    for i in range(0, len(df.columns)):
        print("***** Feature:", df.columns[i], "*****")
        print (df.iloc[:,i].value_counts())
        print("\n ")

df_values(df_src)


# dropping unmeaningful columns by index

df_src.drop([2,4,7,11,12,14,15,18,19,20,21,23,24,26,27,28,29,30,31,32,33])


# Dropping rows with muissing value in the Data Value column

df_src.dropna(subset=['DataValue'])
# Getting insights about diseases analyzing the 'Topic' subset

CountStatus=df_src['Topic'].value_counts()
print (CountStatus)

CountStatus.plot.barh()
#selecting the amount of dollars values

df_exp= df_src[df_src['DataValueUnit']=='$']
df_exp.info()
# evaluating sanitary expenses by location 
df_exp['LocationDesc'].value()
