# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/data.csv")
df.columns
#Check on the data type of each columns

set(df.dtypes)

#df.dtypes[df.dtypes == "int64"]
'''Integer Data Type'''

df.dtypes[df.dtypes == "int64"].iloc[2:]
'''Float Data Type'''

df.dtypes[df.dtypes == "float64"]
'''Data type other than "int64" or "float64" '''

'''The most of the object features are in string format'''

df.dtypes[df.dtypes != "int64"] [df.dtypes != "float64"]


"%d Known Clubs in the dataset"%len(list(set(df.Club[df.Club.isnull() == False])))
from collections import Counter

import matplotlib.pyplot as plt



adf=pd.DataFrame.from_dict(Counter(df.Club[df.Club.isnull() == False]), orient='index').reset_index()

adf.columns = ["Club", "player_cnt"]

fig = plt.figure(figsize= (5,10))

ax = fig.add_subplot(111)





bp= ax.boxplot(adf["player_cnt"], showmeans=True,showcaps=True, showbox=True, meanline=True , patch_artist=True)

#Change the style of Box

for box in bp["boxes"]:

    box.set(color = "purple", linewidth=2)

    box.set(facecolor = "yellow")

#Change the style of the cap

for cap in bp["caps"]:

    cap.set(color="red", linewidth=4)

#Change the style of whiskers

for whisker in bp['whiskers']:

    whisker.set(color='blue', linewidth=2)    

    

ax.set_xticklabels(["All Teams"], fontsize = 15)

plt.title("Distribution of Number of players")

plt.ylabel("Number of Players", fontsize= 20)

plt.ylim(top=35, bottom=17)



IQR = np.percentile(adf.player_cnt, 75) - np.percentile(adf.player_cnt, 25)

plt.text(x=1+0.05, y=int(adf.player_cnt.min()),s=int(adf.player_cnt.min()))



plt.text(x=1+0.05, y=np.percentile(adf.player_cnt, 75)+0.2,s=int(np.percentile(adf.player_cnt, 75)))

plt.text(x=1+0.05, y=np.percentile(adf.player_cnt, 25)-0.3,s=int(np.percentile(adf.player_cnt, 25)))

plt.text(x=1+0.05, y=np.percentile(adf.player_cnt, 25)-1.5*IQR+0.2,s=int(np.percentile(adf.player_cnt, 25)-1.5*IQR))

plt.text(x=1+0.05, y=33,s=int(np.percentile(adf.player_cnt, 75)+1.5*IQR))





np.percentile(adf.player_cnt, 75)

plt.show()

"%d Positions"%len(list(set(df["Position"])))

list(set(df["Position"]))
#df.Value.str.extract('(\d+)')

#df.Value.str.extract(r'(\W+)?([Mk])?(\d+)')

dCol=df.Value.str.extract(r'(\d+)', expand=True)

MKCol=df.Value.str.extract('([MK])', expand=True)

df1=pd.concat([dCol,MKCol], axis=1)

df1.columns = ["digit","M_K"]

#df1["M_K"][df1["M_K"] != "K"][df1["M_K"] != "M"]="1"

df1.M_K[df1.M_K.isnull()] =int(1)

df1["M_K"][df1["M_K"] == "M"]=int(1000000)

df1["M_K"][df1["M_K"] == "K"]=int(1000)

df1["Calculated_Value_in_Mil"]=df1.digit.astype(int) * df1.M_K.astype(int)

df1.Calculated_Value_in_Mil = df1.Calculated_Value_in_Mil.astype(int) // 1000000
import matplotlib.pyplot as plt

df2=pd.concat([df, df1], axis=1)

p=df2[["Name", "Position", "Calculated_Value_in_Mil"]].sort_values(by=["Calculated_Value_in_Mil"], ascending=False).head(20)





plt.figure(figsize=(15,5))

plt.bar(x=p.Name, height=p.Calculated_Value_in_Mil, color = "purple", edgecolor="black",alpha=0.5)

plt.xticks(rotation=45, ha='right', fontsize=15)

plt.xlabel("Name of Player", fontsize = 15)

plt.ylabel("Market Values (M)", fontsize = 15)

plt.title("Top 20 Market Values Player", fontsize = 25)



for i in range(p.shape[0]):

    plt.text(x=float(i)-0.35,y=p.iloc[i].Calculated_Value_in_Mil*0.5,s=p.iloc[i].Position, fontsize=15

            )

plt.show()
#print(df2[df2.Club.isnull() ==True].shape)

#print(df2[df2.Club.isnull() ==False].shape)

#print(df2.shape)



p=df2.groupby("Club")["Calculated_Value_in_Mil"].sum()

p=pd.DataFrame(p).sort_values(by= "Calculated_Value_in_Mil", ascending= False).head(20)



fig=plt.figure(figsize=(15,5))

ax=fig.add_subplot(111)

ax.bar(x=p.index, height=p.Calculated_Value_in_Mil, color="yellow", edgecolor="black")

ax.set_xticklabels(labels=p.index,rotation=45,ha="right")

ax.set_title("Top 20 Clubs with total Players' Market Value", fontsize=20)

ax.set_xlabel("Clubs", fontsize=15)

ax.set_ylabel("Total Market Values",fontsize=15)

plt.show()
fig=plt.figure(figsize = (15,4))

ax=fig.add_subplot(111)

ax.boxplot(df1.Calculated_Value_in_Mil, vert=False)

ax.set_yticklabels(["All Player"], fontsize = 15)

ax.set_xticklabels(labels=np.arange(0,130,20), fontdict={"fontsize":30}, minor=True)

plt.title("Distribution of Players' Market Values", fontsize=20)

plt.xlabel(" Market Values (M)", fontsize = 15)

plt.show()
import seaborn as sns

print(df2.dtypes[df2.dtypes== "int64"])



adf=df2.loc[:,["Age","Overall","Potential","Special", "Calculated_Value_in_Mil"]].corr()

sns.heatmap(adf, annot=True)

plt.title("Correlation of Integer Type Variables vs Market values")

plt.show()
'''Correlation bettwen'''

c=list(df2.columns[df2.dtypes == "float64"]) + ["Calculated_Value_in_Mil"]

adf=df2.loc[:,c]



plt.figure(figsize=(15,15))

sns.heatmap(adf.corr())

plt.title("Correlation between Floatt Features vs Market Values", fontsize=20)

plt.show()

#df2.loc[:,"LS":"RB"]