# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# for Data-Vilsualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Importing data 

crime_data_1 = pd.read_csv("../input/RS_Session_246_AS11.csv")

crime_data_2 = pd.read_csv("../input/RS_Session_246_AU_98_1.1.csv")
# Overview of the Data 1

crime_data_1.head()
crime_data_1.tail(10)
# We have rows stating Total number of Crimes in States/UTs 

# Removing them as it not important as of now

crime_data_1.drop([29,37,38],axis=0,inplace=True)

crime_data_1.shape
# Overview of the Data 2

crime_data_2.head()
crime_data_2.tail(10)
crime_data_2.drop([36],axis=0,inplace=True)

crime_data_2.shape
# Making States as Index in both DataFrame

crime_data_1.rename(columns={"States/UTs":"State/UT"},inplace=True)

crime_data_1.set_index(crime_data_1["State/UT"],inplace=True)

crime_data_1.head()
crime_data_1.drop(["State/UT"],axis=1,inplace=True)
crime_data_2.set_index(crime_data_2["State/UT"],inplace=True)

crime_data_2.drop(["State/UT"],axis=1,inplace=True)

crime_data_2.head()
crime_data_1.index
crime_data_2.index
# Index with different name format :

# data1             data2

# A & N Islands     A & N Island

# D&N Haveli        D & N Haveli

# Delhi UT          Delhi

# Making changes in one of the dataFrame

crime_data_1.rename(index={"A & N Islands":"A & N Island","D&N Haveli":"D & N Haveli","Delhi UT" : "Delhi"},inplace=True)
# Merging both DataFrames 

data = pd.concat([crime_data_1,crime_data_2],axis=1)

data.head()
data.shape
# Removing duplicate columns

data.drop(["2015 - Total rape Cases","2016 - Total rape Cases"],axis=1,inplace=True)

data.head()
# Index, Datatype and Memory information

data.info()

# All columns are in proper format
# Figure Size

plt.figure(figsize=(20,20))



# Seaborn LinePlots

sns.lineplot(x=data.index,y=data["2014 - Cases registered"],ci=None,label="2014")

sns.lineplot(x=data.index,y=data["2015 - Cases registered"],ci=None,label="2015")

sns.lineplot(x=data.index,y=data["2016 - Cases registered"],ci=None,label="2016")



# Matplotlib Additional Features

plt.xlabel("States/UTs")

plt.xticks(rotation=90)

plt.title("2014-2015-2016 Total Cases Registered")

plt.ylabel("Total Cases registered")

plt.legend()

plt.figure(figsize=(20,20))



# Seaborn LinePlots

sns.lineplot(x=data.index,y=data["2014 - Total rape Cases"],ci=None,label="2014")

sns.lineplot(x=data.index,y=data["Rape - 2015"],ci=None,label="2015")

sns.lineplot(x=data.index,y=data["Rape - 2016"],ci=None,label="2016")



# Matplotlib Additional Features

plt.xlabel("States/UTs")

plt.xticks(rotation=90)

plt.title("2014-2015-2016 Total Rape Cases")

plt.ylabel("Total Rape Cases")

plt.legend()

plt.figure(figsize=(20,20))



# Seaborn LinePlots

sns.lineplot(x=data.index,y=(data["2014 - Total rape Cases"]/data["2014 - Cases registered"])*100,ci=None,label="2014")

sns.lineplot(x=data.index,y=(data["Rape - 2015"]/data["2015 - Cases registered"])*100,ci=None,label="2015")

sns.lineplot(x=data.index,y=(data["Rape - 2016"]/data["2016 - Cases registered"])*100,ci=None,label="2016")



# Matplotlib Additional Features

plt.xlabel("States/UTs")

plt.xticks(rotation=90)

plt.title("% of Rape Cases during 2014-2016")

plt.ylabel("% of Rape Cases")

plt.legend()
plt.figure(figsize=(20,20))



# Seaborn LinePlots

sns.lineplot(x=data.index,y=(data["Assaults (molestation) - 2015"]/data["2015 - Cases registered"])*100,ci=None,label="2015")

sns.lineplot(x=data.index,y=(data["Assaults (molestation) - 2016"]/data["2016 - Cases registered"])*100,ci=None,label="2016")



# Matplotlib Additional Features

plt.xlabel("States/UTs")

plt.xticks(rotation=90)

plt.title("2015-2016 % of Assaults (Molestation) Cases")

plt.ylabel("% of Assaults( Molestation) Cases")

plt.legend()
plt.figure(figsize=(20,20))



# Seaborn LinePlots

sns.lineplot(x=data.index,y=(data["Murder (women) - 2015"]/data["2015 - Cases registered"])*100,ci=None,label="2015")

sns.lineplot(x=data.index,y=(data["Murder (women) - 2016"]/data["2016 - Cases registered"])*100,ci=None,label="2016")



# Matplotlib Additional Features

plt.xlabel("States/UTs")

plt.xticks(rotation=90)

plt.title("2015-2016 % of Murder Cases")

plt.ylabel("% of Murder Cases")

plt.legend()
State_UT = []

UTs = ["A & N Island","D & N Haveli","Delhi","Puducherry","Chandigarh","Daman & Diu","Lakshadweep"]

for st in data.index:

    if st not in UTs:

        State_UT.append("State")

    else:

        State_UT.append("UT")

temp = pd.concat([data,pd.DataFrame(data={"State_UT":pd.Series(State_UT,index=data.index)})],axis=1)

temp.head()
North_India = ["Delhi","Haryana","Jammu & Kashmir","Himachal Pradesh","Uttar Pradesh","Punjab","Uttarakhand"]

West_India = ["D & N Haveli","Daman & Diu","Goa","Gujarat","Maharashtra","Rajasthan"]

South_India = ["Kerala", "Tamil Nadu", "Karnataka", "Andhra Pradesh","Telangana","Puducherry"]

Middle_India = ["Madhya Pradesh","Chhattisgarh"]

East_India = ["Odisha","West Bengal","Bihar","Jharkhand"]

North_East_India = ["Sikkim","Meghalaya","Assam","Arunachal Pradesh","Nagaland","Manipur","Tripura","Mizoram"]

Island_India = ["A & N Island","Lakshadweep"]
def directionwise(li,col):

    r = []

    for i in li:

        r.append(temp.loc[i,col])

    return pd.Series(r)
def PlotDirectionWise(List,col1,col2,title,ylabel):

    

    plt.figure(figsize=(5,5))



    sns.barplot(x=pd.Series(List),y=(directionwise(List,col1)/directionwise(List,col2))*100,hue=directionwise(List,"State_UT"))



    plt.xlabel("States/UTs")

    plt.xticks(rotation=90)

    plt.title(title)

    plt.ylabel(ylabel)

    plt.legend()
#PlotDirectionWise(North_India,"2014 - Total rape Cases","2014 - Cases registered","% of Rape Cases - 2014","% of Rape Cases")

#PlotDirectionWise(North_India,"Rape - 2015","2015 - Cases registered","% of Rape Cases - 2015","% of Rape Cases")

PlotDirectionWise(North_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(North_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(North_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(West_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(West_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(West_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(South_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(South_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(South_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(Middle_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(Middle_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(Middle_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(East_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(East_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(East_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(North_East_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(North_East_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(North_East_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
PlotDirectionWise(Island_India,"Rape - 2016","2016 - Cases registered","% of Rape Cases - 2016","% of Rape Cases")
PlotDirectionWise(Island_India,"Assaults (molestation) - 2016","2016 - Cases registered","% of Assault Cases - 2016","% of Assault")
PlotDirectionWise(Island_India,"Murder (women) - 2016","2016 - Cases registered","% of Murder Cases - 2016","% of Murders")
def RankCalculator(data):

    # Rape %

    RapePr14 = (data["2014 - Total rape Cases"]/data["2014 - Cases registered"])*100

    RapePr15 = (data["Rape - 2015"]/data["2015 - Cases registered"])*100

    RapePr16 = (data["Rape - 2016"]/data["2016 - Cases registered"])*100

    

    # Assault %

    AssaultPr15 = (data["Assaults (molestation) - 2015"]/data["2015 - Cases registered"])*100

    AssaultPr16 = (data["Assaults (molestation) - 2016"]/data["2016 - Cases registered"])*100

    

    # Murder %

    MurderPr15 = (data["Murder (women) - 2015"]/data["2015 - Cases registered"])*100

    MurderPr16 = (data["Murder (women) - 2016"]/data["2016 - Cases registered"])*100

    

    #StableWt = [50,30,20]

    #IncWt = [60,35,25]

    #DecWt = [45,25,15]

    

    IncDec = []

    AvgPr = []

    for (r14,r15,r16,a15,a16,m15,m16) in zip(RapePr14,RapePr15,RapePr16,AssaultPr15,AssaultPr16,MurderPr15,MurderPr16):

        

        # Coeff of Inc or Dec in Rape %

        lr = 0

        if r15>r14:

            lr = 1

        elif r15<r14:

            lr = -1

    

        if r16>r15:

            lr = lr + 1

        elif r16<r15:

            lr = lr - 1

        

        # Coeff of Inc or Dec in Assaults %

        la = 0

        if a16>a15:

            la = 1

        elif a16<a15:

            la = -1

        

        # Coeff of Inc or Dec in Murders %

        lm = 0

        if m16>m15:

            lm = 1

        elif m16<m15:

            lm = -1

        

        IncDec.append([lr,la,lm])

        AvgPr.append([(r14+r15+r16)/3 , (a15+a16)/2 , (m15+m16)/2])

        

    # print(IncDec)

    # print(AvgPr)

    FinalScore = []

    for ([r,a,m],[ri,ai,mi]) in zip(AvgPr,IncDec):

        cnt = 0

        total = 0

        if ri==-2:

            cnt = 30

            total = 30*r

        elif ri==-1:

            cnt = 40

            total = 40*r

        elif ri==0:

            cnt = 50

            total = 50*r

        elif ri==1:

            cnt = 60

            total = 60*r

        else:

            cnt = 70

            total = 70*r

            

        if ai==-1:

            cnt = cnt+25

            total = total + 25*a

        elif ai==0:

            cnt = cnt+30

            total = total + 30*a

        else:

            cnt = cnt+35

            total = total + 35*a

            

        if mi==-1:

            cnt = cnt+15

            total = total + 15*m

        elif mi==0:

            cnt = cnt+20

            total = total + 20*m

        else:

            cnt = cnt+25

            total = total + 25*m

            

        FinalScore.append(total/cnt)

        

    return pd.DataFrame({"Scores" : pd.Series(FinalScore,index=data.index)})

        

finalScore = RankCalculator(data)
Ranks = finalScore.sort_values("Scores")

Ranks
Ranks.index