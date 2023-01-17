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
df_05_07 = pd.read_csv("../input/accidents_2005_to_2007.csv")

df_09_11 = pd.read_csv("../input/accidents_2009_to_2011.csv")

df_12_14 = pd.read_csv("../input/accidents_2012_to_2014.csv")



df = pd.concat([df_05_07, df_09_11, df_12_14], axis=0)
print("Number of Duplicates: ",(df.duplicated().sum()))

df["Dup_Flag"] = df.duplicated()

df = df.loc[df["Dup_Flag"]==False]

df.drop(("Dup_Flag"), axis=1, inplace=True)
df.describe()
df.info()
percent_missing = df.isnull().sum() * 100 / len(df)

missing_value_df = pd.DataFrame(data=percent_missing.values, index=percent_missing.index)

missing_value_df.columns = ["Missing Value %"]

missing_value_df
df["Accident_Severity"].value_counts().plot("barh", alpha=0.75, figsize=(6,4), title="Distribution of Accident Severity")
Day_of_Week_D = df["Day_of_Week"].value_counts().sort_index()/len(df)*100

Day_of_Week_D.index = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

plt.bar(Day_of_Week_D.index, Day_of_Week_D.values, alpha=0.75, color="orange")

plt.title("Distribution of Accidents by Day of the Week")

for i, val in enumerate(Day_of_Week_D.values):

    plt.text(i-0.3, val-1,  str( "{:.{}f}".format( val, 1 )), color='blue', fontweight='bold')

    
Acc_By_Date = pd.DataFrame()

Acc_By_Date["Accident_Severity"] = df["Accident_Severity"]

Acc_By_Date["Date"] = pd.to_datetime(df["Date"])

Acc_By_Date["Hour"] = df.Time.str.slice(0,2)

Acc_By_Date["Year"] = df["Year"]

Acc_By_Date["Day_of_Week"] = df["Day_of_Week"]
Total_Acc = Acc_By_Date["Date"].value_counts()



plt.figure(1, figsize=(7,7))

sn.distplot(Total_Acc.values)

plt.title("Distribution of Total Accidents Per Day")

plt.text(100, 0.004,'Mean:'+ str( "{:.{}f}".format( Total_Acc.mean(), 1 )), fontsize=12)

plt.text(100, 0.0038,'Median:'+ str( "{:.{}f}".format( Total_Acc.median(), 1 )), fontsize=12)

plt.text(100, 0.0036,'STD:'+ str( "{:.{}f}".format( Total_Acc.std(), 1 )), fontsize=12)

plt.show()

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14,14))

count = 0



for i, col in enumerate(Acc_By_Date["Year"].unique()):



    Total_Acc_by_Year = Acc_By_Date["Date"].loc[Acc_By_Date["Year"]==col].value_counts()

    

    ax[i//3,count].set_ylim([0.0,0.007])

    sn.distplot(Total_Acc_by_Year.values,  ax=ax[i//3,count])

    ax[i//3,count].title.set_text("Year:"+str(col))

    ax[i//3,count].text(100, 0.006,'Mean:'+ str( "{:.{}f}".format( Total_Acc_by_Year.mean(), 1 )), fontsize=12)

    ax[i//3,count].text(100, 0.0055,'Median:'+ str( "{:.{}f}".format( Total_Acc_by_Year.median(), 1 )), fontsize=12)

    ax[i//3,count].text(100, 0.0050,'STD:'+ str( "{:.{}f}".format( Total_Acc_by_Year.std(), 1 )), fontsize=12)

    

    count = count + 1

    if count == 3: count = 0

plt.show()
Acc_By_Hour = Acc_By_Date["Hour"].value_counts().sort_index()



Acc_By_Hour = (Acc_By_Hour/Acc_By_Hour.sum())*100



plt.figure(1, figsize=(8,5))

plt.bar(Acc_By_Hour.index, Acc_By_Hour.values, alpha=0.75)

plt.title("Distribution of Accidents by Hour")

plt.ylabel("%")

plt.xlabel("Hour of the Day (24hr)")

Day_Of_Week_Dict = {1:"Sun", 2:"Mon", 3:"Tue", 4:"Wed", 5:"Thu", 6:"Fri", 7:"Sat"}



fig, ax = plt.subplots(nrows=4, ncols=2, figsize=(16,16))

count = 0



for i, col in enumerate(Acc_By_Date["Day_of_Week"].sort_values().unique()):

    Acc_By_Hour_By_Day = Acc_By_Date["Hour"].loc[Acc_By_Date["Day_of_Week"]==col].value_counts().sort_index()

    Acc_By_Hour_By_Day = (Acc_By_Hour_By_Day/Acc_By_Hour_By_Day.sum())*100

    

    ax[i//2, count].bar(Acc_By_Hour_By_Day.index, Acc_By_Hour_By_Day.values, alpha=0.75)

    ax[i//2, count].title.set_text("Distribution of Accidents by Hour for:" + str(Day_Of_Week_Dict.get(col)))

    ax[i//2, count].set_ylim([0,11])

    ax[i//2, count].set_ylabel("%")



    count = count + 1

    if count == 2: count = 0

        

ax[3,1].set_yticks([])

ax[3,1].set_xticks([])

plt.show()

#Number_of_Casualties

plt.figure(1, figsize=(7,7))

sn.distplot(df["Number_of_Vehicles"].values)

plt.title("Distribution of Number of Vehicles involved")

plt.show()

#Number_of_Casualties

plt.figure(1, figsize=(7,7))

sn.distplot(df["Number_of_Casualties"].values)

plt.title("Distribution of Number of Casualties involved")

plt.show()

plt.figure(1, figsize=(7,7))

sn.scatterplot(df["Number_of_Casualties"].values, df["Number_of_Vehicles"].values)

plt.title("Vehicles vs Casualties")

plt.xlabel("Number_of_Casualties")

plt.ylabel("Number_of_Vehicles")

plt.show
Object_Col = df.drop(["Accident_Index", "Date", "Time"], axis=1).select_dtypes("object").columns



Object_Col = list(Object_Col)



Object_Col.append("Speed_limit")

Object_Col.append("Local_Authority_(District)")

Object_Col.append("1st_Road_Class")

Object_Col.append("2nd_Road_Class")

Object_Col.append("Police_Force")

Object_Col.append("Urban_or_Rural_Area")

for i, col in enumerate(Object_Col):



    Agg = df[col].value_counts().sort_values()

    

    Agg = (Agg/Agg.sum())*100

    

    if len(Agg) < 20:

    

        plt.figure(i, figsize=(7,5))

        plt.barh(Agg.index.astype(str), Agg.values, alpha=0.75,  color="orange")

        plt.title(str(col))

        plt.rc('font', size=12)    

        plt.show()

        

    else:

        

        Agg = Agg.head(20)

        plt.figure(i, figsize=(7,5))

        plt.barh(Agg.index.astype(str), Agg.values, alpha=0.75, color="orange")

        plt.title(str(col) + " top 20")

        plt.rc('font', size=12)    

        plt.show()
Acc_Sev_Dict = {1:"Fatal",

                2:"Serious",

                3:"Slight"}



for j, col in enumerate(Object_Col):

    

    plt.figure(j, figsize=(7,5))

    

    if len(df[col].unique()) >20:

        pass

    

    else:

        

        for i in range(1,4):

            DV = df[col].loc[df["Accident_Severity"]==i].value_counts()

            DV = (DV/DV.sum())*100

            plt.barh(DV.index.astype(str), DV.values, alpha=0.40, label=Acc_Sev_Dict[i])

    

        plt.legend()

        plt.title(col)

        plt.show()