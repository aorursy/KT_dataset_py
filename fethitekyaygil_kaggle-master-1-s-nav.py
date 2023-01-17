# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_2015=pd.read_csv("/kaggle/input/world-happiness/2015.csv")

df_2016=pd.read_csv("/kaggle/input/world-happiness/2016.csv")

df_2017=pd.read_csv("/kaggle/input/world-happiness/2017.csv")

df_2018=pd.read_csv("/kaggle/input/world-happiness/2018.csv")

df_2019=pd.read_csv("/kaggle/input/world-happiness/2019.csv")



df_2015["Year"]=2015

df_2016["Year"]=2016

df_2017["Year"]=2017

df_2018["Year"]=2018

df_2019["Year"]=2019
df_1=pd.concat([df_2015,df_2016,df_2017])

df_2=pd.concat([df_2018,df_2019])
scores=df_1["Happiness Score"].dropna()

happinessScores=df_1["Happiness.Score"].dropna()

appended=scores.append(happinessScores)

df_1["Happiness Score"]=appended
gdpPerCapitas=df_1["Economy (GDP per Capita)"].dropna()

economyGdpPerCapitas=df_1["Economy..GDP.per.Capita."].dropna()

appended=gdpPerCapitas.append(economyGdpPerCapitas)

df_1["Economy (GDP per Capita)"]=appended
health1=df_1["Health (Life Expectancy)"].dropna()

health2=df_1["Health..Life.Expectancy."].dropna()

appended=health1.append(health2)

df_1["Health (Life Expectancy)"]=appended
dystopiaResidual1=df_1["Dystopia Residual"].dropna()

dystopiaResidual2=df_1["Dystopia.Residual"].dropna()

appended=dystopiaResidual1.append(dystopiaResidual2)

df_1["Dystopia Residual"]=appended
trustGovernmentCorruption1=df_1["Trust (Government Corruption)"].dropna()

trustGovernmentCorruption2=df_1["Trust..Government.Corruption."].dropna()

appended=trustGovernmentCorruption1.append(trustGovernmentCorruption2)

df_1["Trust (Government Corruption)"]=appended
happinessRank1=df_1["Happiness Rank"].dropna()

happinessRank2=df_1["Happiness.Rank"].dropna()

appended=happinessRank1.append(happinessRank2)

df_1["Happiness Rank"]=appended
df_1.drop(columns=["Happiness.Score","Happiness.Rank","Economy..GDP.per.Capita.","Health..Life.Expectancy.","Dystopia.Residual","Trust..Government.Corruption.","Standard Error"],inplace=True)
df_1.rename(columns={"Economy (GDP per Capita)":"GDP per capita","Happiness Rank":"Overall Rank","Happiness Score":"Score"},inplace=True)

df_2.rename(columns={"Freedom to make life choices":"Freedom","Country or region":"Country","Perceptions of corruption":"Trust (Government Corruption)"},inplace=True)
#!pip install ycimpute

!pip install ycimpute==0.1.1
merged_df=pd.merge(df_1,df_2,on=["Country","Score","GDP per capita","Freedom","Trust (Government Corruption)","Generosity","Year"],how="outer")

merged_df["IsTurkey"]=merged_df["Country"]=="Turkey"

merged_df["IsTurkey"]=merged_df["IsTurkey"].astype("int64")
from ycimpute.imputer import knnimput

merged_df_cont=merged_df.select_dtypes(["int64","float64"])

np_arr_merged_df=np.array(merged_df_cont)

knned_merged_df=knnimput.KNN(k=4).complete(np_arr_merged_df)

knned_df=pd.DataFrame(knned_merged_df,columns=merged_df_cont.columns)
df_turkey=knned_df[(knned_df["IsTurkey"] ==1.0)]

df_turkey.drop(columns=["IsTurkey","Overall rank","Overall Rank"],inplace=True)
grouped_df=df_turkey.groupby("Score").sum()
grouped_df.Year.plot()
sns.barplot(x="Year",y=grouped_df.index,data=grouped_df)
grouped_df_resetted=grouped_df.reset_index()
import matplotlib.pyplot as plt



grouped_df_resetted.plot(x="Score", subplots=True,layout=(10,4),figsize=(10,20),legend=True)

plt.legend()



plt.show()
from sklearn.preprocessing import StandardScaler

scaled_grouped_np=StandardScaler().fit_transform(grouped_df.values)

scaled_grouped_df=pd.DataFrame(scaled_grouped_np,columns=grouped_df.columns,index=grouped_df.Year)

scaled_grouped_df.sort_index()

scaled_grouped_df=scaled_grouped_df.abs()
scaled_grouped_df=scaled_grouped_df.sort_index()

scaled_grouped_df.plot.bar(width=0.8,figsize=(15,5),ylim=(0,2.5),align='center').legend(bbox_to_anchor=(1.2, 0.5))
happiest_2019=df_2019[df_2019["Score"]==df_2019["Score"].max()]

#Overall rank ile de alabilirdik
X=happiest_2019[["GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]]

melted_df_X=pd.melt(X)
sns.barplot(y=melted_df_X.variable,x=melted_df_X.value)