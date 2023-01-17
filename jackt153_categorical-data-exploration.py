# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
Train_Path = "../input/train.csv"
Column_Header = pd.read_csv(Train_Path, nrows=55)
Column_Header.info()

df = pd.read_csv(Train_Path, usecols=(Column_Header.select_dtypes(include="object").columns))
DV = pd.read_csv(Train_Path, usecols=["HasDetections"])
df.drop(["MachineIdentifier"], axis=1, inplace=True)
for i, col in enumerate(df.columns):

    try:
        Value = (pd.isna(df[col]).value_counts().loc[True]/len(df))*100

        print(col, " % of null values: ",Value,"%")
        
        Missing_Value = df[col].mode().values

        df[col].fillna(Missing_Value[0], axis=0, inplace=True)
        print("Updated; ", col)
        
    except:
        pass
    
df_C = pd.concat((df,DV),axis=1)

for i, col in enumerate(df_C.select_dtypes(include="object").columns):
        
    df_Desc = df_C[col].describe(include='all')
    
    if df_Desc["unique"] > 20:
        #pass
        #print(i, " ", col)

        Temp = df_C[["HasDetections",col]]
        #Creates column of 1s
        Temp["Val"] = 1
        
        #Creates a pivot table with catergories as rows and DV as columns
        Pivot = pd.pivot_table(Temp, values=["Val"], index=[col], 
                       columns=["HasDetections"], aggfunc=np.sum)
        
        #Plots this on a bar graph sort by 0 and with top 25 values
        Total = Pivot.sort_values([('Val', 0)], ascending=False).head(25)
        
       
        plt.figure(i, figsize=(7,5))
        DV_1_G = plt.barh(Total['Val', 1].index, Total['Val', 1].values, alpha=0.4)
        DV_0_G = plt.barh(Total['Val', 0].index, Total['Val', 0].values, alpha=0.4)
        #plt.xticks( rotation='vertical' )
        plt.title(col)
        plt.legend((DV_1_G, DV_0_G), ("DV = 1", "DV = 0"))
        plt.show()
        
        #print(DV_1)
        
    else:
        #print(i, " ", col)
        
        DV_1 = df_C[col].loc[df_C["HasDetections"]==1].value_counts()
        DV_1 = (DV_1/DV_1.sum())*100
        
        DV_0 = df_C[col].loc[df_C["HasDetections"]==0].value_counts()
        DV_0 = (DV_0/DV_0.sum())*100
        
        plt.figure(i, figsize=(7,5))
        DV_1_G = plt.barh(DV_1.index, DV_1.values, alpha=0.4)
        DV_0_G = plt.barh(DV_0.index, DV_0.values, alpha=0.4)
        plt.title(col)
        plt.legend((DV_1_G, DV_0_G), ("DV = 1", "DV = 0"))
        plt.show()
        
        #print(DV_1)
        