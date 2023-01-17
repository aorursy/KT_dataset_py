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
import matplotlib.pyplot as plt 
%matplotlib inline 
from matplotlib import style 
style.use("ggplot")


terror = pd.read_csv("../input/globalterrorismdb_0617dist.csv", encoding = "ISO-8859-1", low_memory = False)
terror.shape
terror.head(3)   ## Notice that month and day has 0 values. We will change it later on. There are many columns which will not be useful for our 
## further analysis, we will remove them too. We will rename columns. Select desired columns and continue our analysis with a smaller but more
## effective dataframe.
a = terror.columns.values.tolist()
a
terror.rename(columns = {"iyear": "Year", "imonth": "Month", "iday": "Day", "country_txt" : "Country", 
                         "region_txt" : "Region", "city" : "City", "attacktype1_txt": "AttackType","targtype1_txt": "Target_Type", "target1" : "target",
                         "gname" : "Group_Name", "individual" : "Individual", "weaptype1_txt" : "Weapon_Type", "nkill" : "Killed", "nwound" : "Wounded",
                         "summary" : "Summary", "motive": "Motive"} , inplace = True )

                         
terror2 = terror.copy()
##lets only pick useful columns for our further analysis.
terror_data = terror2[["eventid", "Year", "Month", "Day", "Country", "Region", "City","latitude", "longitude",
                        "AttackType","Target_Type", "target", "Group_Name","Individual", "Weapon_Type", "Killed", 
                        "Wounded", "Summary", "Motive"]]
terror_data.shape
terror_data.info()

terror_data.isnull().sum()
terror_data["Total_Casualities"] = terror_data.Killed + terror_data.Wounded
terror_data[["Killed", "Wounded", "Total_Casualities"]].head(20) ##to check above operation is doing well.
#I want to remove rows which have Nan values for Killed, Wounded and Casualities columns ( all 3 of them are Nan's)
mask1 = terror_data.Killed.isnull() == True
mask2 = terror_data.Wounded.isnull() == True
mask3 = terror_data.Total_Casualities.isnull() == True
mask4 = mask1 & mask2 & mask3
mask4.head(5)  ## we can assert it from just above table 
##now it is time to remove this rows from my data .
terror_data = terror_data[~mask4]
terror_data.shape
##now lets remove 0's from both Month and Day columns since there is simply no 0th day and month.
terror_data["Day"][terror_data.Day == 0] = 1
terror_data["Month"][terror_data.Month == 0] = 1
##below warning is not a big deal.

terror_data[["Day", "Month"]].head(5)
## lets create a datetime object which we will use it later on as an index. Pretty nifty for analyzing time trends.
terror_data["Date"]  = pd.to_datetime(terror_data[["Year", "Month", "Day"]], errors = "coerce")  ## to create datetime object from existing columns.
terror_data[["Month", "Day", "Year", "Date"]].head(5) 
##lets use our date columns as an index.
terror_data.set_index("Date", inplace = True)

## lets sort the index.
terror_data.sort_index(inplace = True)

terror_data.head(5)   ##to check whether we are ready for deeper analysis or not. 
# It seems we are ready for deeper analysis. We manipulated data in many different ways, we removed Nan values, we dropped unnecessary columns, 
# we renamed columns accordingly, we created a datetime index from existing columns, we created some columns from existing columns, we sorted our data 
# according to the events occurance date. It is time for further investigation. Since we already did a lot of work, rest of the analysis will be 
# more relaxed and insightful hopefully :)  Before going into further analysis I want to keep this clean data in another csv file : 
terror_data.to_csv("Terror_Data_Cleaned.csv")

##lets start with statistical exploration.
terror_data.describe()
## It is time for us to ask advanced questions
## What is the number of total casualities per year: 
s1 = terror_data.groupby("Year")["Total_Casualities"].sum()
s1
## lets visualize the number of casualities due to terror attacks through years.
s1.plot.line()
plt.xlabel("Years")
plt.ylabel("Casualities (person)")
plt.title("Total Casualities in Years")
plt.margins(0.02)
plt.tight_layout()
## Although sometimes terrorism declined globally, it is obvious that it kept a dramatic rising slope over the years. Especially after 2010 ;
# there is a significant rise in terrorist casualities in the world which can be due to  Syrian Civil War which affected many countries later on.
terror_data.Country.nunique()   ## How many countries had attacked by terrorists until now.
## Which 10 country affected most by terrorists.
terror_data.Country.value_counts().sort_values(ascending = False).head(10)
# Lets plot countries that are most affected by terrorist attacks.
s2 = terror_data.Country.value_counts().sort_values(ascending = False).head(10)
s2.plot.bar()
plt.xlabel("Countries")
plt.ylabel("Number of Terror Attacks")
plt.title("Top 10 Terrorist Attacking Countries")
plt.margins(0.02)
plt.tight_layout()
## Lets see how many terrorist attack happened during this period (1970-2016)
s3 = terror_data.groupby("Year")["eventid"].count()
s3.plot.line()
plt.xlabel("Years")
plt.ylabel("Number of Terror Attacks")
plt.title("Number of Terror Attacks in Years")
plt.tight_layout()
plt.margins(0.02)

# Again, it can be seen from the graph that terrorist attacks dramatically rose after 2010. It is almost tripled in a relatively short time period.
s3.idxmax() ## It will give the year of maximum terror attacks occured.    
s3.idxmin() ## It will give the year of minimum terror attacks occured.    
## Which 10 cities are attacked most by the terrorists:
terror_data.groupby("City")["eventid"].count().nlargest(10)
## lets remove "Unknown" city from the series above and then plot the cities that are attacked most by the terrorists.
s4 = terror_data.groupby("City")["eventid"].count().nlargest(10)
del s4["Unknown"]
s4.plot.barh()
plt.xlabel("Number of terrorist attacks")
plt.ylabel("Cities")
plt.title("Cities that attacked most by terrorists")
plt.tight_layout()
## what are the different methods/ attack types terrorists use to attack ? 
terror_data.AttackType.nunique()
## there are 9 different attack strategies terrorists use, lets see their popularity among them.
s5 = terror_data.groupby("AttackType")["eventid"].count()
s6 = terror_data.groupby("AttackType")["eventid"].count().sort_values(ascending = False)
s6
## lets combine the least 4 attack types into the category of "Others":
s6["Others"] = np.sum(s6[-4:])
s6 = s6.sort_values(ascending = False)
for i in s6[-4:].index:
    del s6[i]
s6
##lets visualize it to gather better insights from terrorists most common attack types :
a = plt.pie(s6, autopct = '%1.1f%%', labels = s6.index, startangle = 90)
plt.title("Terrorists Attack Strategies")
plt.tight_layout()
plt.margins(0.05)

## how many different targets chosen by terrorists. 
terror_data["Target_Type"].nunique()
## Terrorist picked 22 different targets to attack. Lets see and visualize what are their main targets?
s7 = terror_data.groupby("Target_Type")["eventid"].count()
s7 = s7.sort_values(ascending = False)
s7.plot.bar()
plt.xlabel("Targets")
plt.ylabel("Number of Attacks")
plt.title("Attack distribution by Targets")

## lets check correlation between Killed people and Wounded people in terror attacks. This is just for statistical purpose, We must see a 
# moderate or even strong correlation here. It is 0.53 which can be easily accepted as moderate correlation. 
terror_data.Killed.corr(terror_data.Wounded) 
## Who are the top 15  active terrorist groups/ caused most number of terrorist attacks in the world : 
terror_data.groupby("Group_Name")["eventid"].count().sort_values(ascending = False).head(15)
s8 = terror_data.groupby("Group_Name")["eventid"].count().sort_values(ascending = False).head(15)
del s8["Unknown"]
## Taliban, Shining Path(SL), Islamic State of Iraq and the Levant (ISIL)  are the most active terror groups globally.
s8.plot.barh()
plt.ylabel("Terrorist Groups")
plt.xlabel("Number of attacks")
plt.title("Most active terrorist groups in the world")
plt.tight_layout()

## Which terrorist groups are the most bloody-minded ? 
s9 = terror_data.groupby("Group_Name")["Total_Casualities"].sum().sort_values(ascending = False).head(10)
del s9["Unknown"]
s9
## Islamic State of Iraq and the Levant (ISIL), Taliban, Boko Haram are the most bloody minded terror groups.
## lets dive deep into terror attacks that happened in Turkey : 
terror_in_Turkey = terror_data[terror_data.Country == "Turkey"]
terror_in_Turkey.shape
terror_in_Turkey.isnull().sum()
## who are the 10 most active terrorist groups in Turkey ?
tr_terror_groups = terror_in_Turkey["Group_Name"].value_counts()[0:10]
tr_terror_groups
tr_terror_groups = tr_terror_groups.drop("Unknown") ## to drop Unknown 
tr_terror_groups.plot.barh()
plt.ylabel("Terrorist Groups")
plt.xlabel("Number of Attacks")
plt.title("Most active terrorist groups in Turkey")
plt.tight_layout()
# It is obvious that PKK is the most annoying / active terrorist group in Turkey. Lets dive for more information 
## about them like when they attacked Turkey first and what is total number of casualities due to them etc.
## lets also investigate their activity through time.
PKK_data = terror_in_Turkey[terror_in_Turkey["Group_Name"] == "Kurdistan Workers' Party (PKK)"]
PKK_data.sort_index(inplace = True)
PKK_data.shape


## We sorted index,when PKK first attacked Turkey ? What are other details about this attack ? 
PKK_data.head(1)
## In 15 August 1984 PKK first attacked Turkey. Eruh was the location.Sadly, 1 people killed, 9 people wounded.
print("Total Casualities due to PKK is: " + str(PKK_data["Total_Casualities"].sum().astype(int)) + " people."  )
print("Total number of people killed by PKK is: " + str(PKK_data["Killed"].sum().astype(int)) + " people.")  #4635 people killed because of PKK terrorists.
print("Total number of people wounded by PKK attacks is: " + str(PKK_data["Wounded"].sum().astype(int)) + " people." ) ## 4502 people wounded because of their attacks.
## lets look their attack numbers through years.
pkk = PKK_data.groupby("Year")["eventid"].count()
pkk.plot.line() 
plt.xlabel("Years")
plt.ylabel("Pkk attacks")
plt.title("Pkk attacks through years")
plt.margins(0.02)
plt.tight_layout() 
## lets see how they damaged Turkey through years by looking total casualities. 
pkk_casualities = PKK_data.groupby("Year")["Total_Casualities"].sum()
pkk_casualities.plot.line()  
plt.xlabel("Years")
plt.ylabel("Pkk damage- Casualities")
plt.title("Pkk casualities through years")
plt.margins(0.02)
plt.tight_layout()
# Lets go back to terror_in_Turkey data and answer the following questions:
# Question 1 : Which cities exposed the most number of terrorist attacks ?
cities = terror_in_Turkey.groupby("City")["eventid"].count().nlargest(10)
cities
del cities["Unknown"]
cities
## What targets had chosen by terrorists mostly in Turkey ?
targets = terror_in_Turkey.groupby("Target_Type")["eventid"].count().nlargest(10)
targets
targets.plot.bar()
plt.xlabel("targets")
plt.ylabel("Number of attacks")
plt.title("Terrorists attack targets in Turkey")
plt.tight_layout()
## Last question : Which terrorist attack caused the most damage in Turkey ? When did this attack happen and where was the location ?
##How many people killed and wounded?
disaster = terror_in_Turkey["Total_Casualities"].idxmax()
disaster

terror_in_Turkey.loc[disaster]
## As seen from above, it was in 10 October 2015, terrorists Attacked Ankara with bombs and explosives, terrorist group was ISIL and unfortunately 
##it caused 105 people to death and 245 people wounded.
 # This was the last phase of my analysis and it was an unpleasant finish just like every terrorist attack happening in the world. 
# I hope one day we manage to live in peace in this world since we are literally living only once ! I hope everybody understands this and 
# try to solve the problems without the need of violance, invasion, insulting and attacking. 
