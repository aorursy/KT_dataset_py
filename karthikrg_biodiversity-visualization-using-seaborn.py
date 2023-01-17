#Import the necessary libraries



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # data visualization

import matplotlib.pyplot as plt # plotting

from collections import Counter #count
#Running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
#DataFrame to hold Park data

df_parks = pd.read_csv(r"../input/parks.csv")
#File Description

#Park Code             National Parks Service park code                                  String

#Park Name             Office park name                                                  String

#State                 US state(s) in which the park is located. Comma-separated         String

#Acres                 Size of the park in acres                                         Numeric

#Latitude              Latitude of the park (centroid)                                   Numeric

#Longitude             Longitude of the park (centroid)                                  Numeric



df_parks.head()
#Adding 2 columns. 

#1. The region the State belongs to. We want to visualise at a higher level.

#2. The expansion of the State abbreviation 



df_parks["Region"]=np.nan

df_parks["State_FullName"] = np.nan
#Fill the regions

# MidWest: South Dakota (SD), Ohio (OH), Michigan (MI), North Dakota (ND), Minnesota (MN)

# Northeast: Maine (ME)

# South: Texas (TX), Florida (FL), South Carolina (SC), Tennessee/North Carolina (TN, NC), Arkansas (AR), Kentucky (KY), Virginia (VA)

# West: Utah (UT), Colorado (CO), New Mexico (NM), California (CA), Oregon (OR), Alaska (AK), California/Nevada (CA, NV), Montana (MT), Nevada (NV), Arizona (AZ), Colorado (CO), Wyoming (WY), Hawaii (HI), Washington (WA), Wyoming, Montana,Idaho (WY, MT, ID)



df_parks["Region"]= np.where((df_parks["State"]=="SD")| (df_parks["State"]=="OH") | (df_parks["State"]=="MI") | (df_parks["State"]=="ND") | (df_parks["State"]=="MN"), "MidWest", df_parks["Region"])

df_parks["Region"]= np.where((df_parks["Region"]=="nan") & ((df_parks["State"]=="ME")), "Northeast", df_parks["Region"])

df_parks["Region"]= np.where((df_parks["Region"]=="nan") & ((df_parks["State"]=="TX")| (df_parks["State"]=="FL") | (df_parks["State"]=="SC")| (df_parks["State"]=="TN, NC") | (df_parks["State"]=="AR")| (df_parks["State"]=="KY")| (df_parks["State"]=="VA")), "South", df_parks["Region"])

df_parks["Region"]= np.where((df_parks["Region"]=="nan") & ((df_parks["State"]=="UT")| (df_parks["State"]=="CO") | (df_parks["State"]=="NM")| (df_parks["State"]=="CA") | (df_parks["State"]=="OR")| (df_parks["State"]=="AK")| (df_parks["State"]=="CA, NV")| (df_parks["State"]=="MT")| (df_parks["State"]=="NV")| (df_parks["State"]=="AZ")| (df_parks["State"]=="WY")| (df_parks["State"]=="HI")| (df_parks["State"]=="WA")| (df_parks["State"]=="WY, MT, ID")), "West",df_parks["Region"])
#Expand the State's abbreviation



df_parks["State_FullName"] = np.where(df_parks["State"]=="ME", "Maine", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="UT", "Utah", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="SD", "South Dakota", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="TX", "Texas", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="FL", "Florida", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="CO", "Colorado", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="NM", "New Mexico", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="CA", "California", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="SC", "South Carolina", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="OR", "Oregon", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="OH", "Ohio", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="AK", "Alaska", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="CA, NV", "California/Nevada", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="MT", "Montana", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="NV", "Nevada", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="AZ", "Arizona", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="TN, NC", "Tennessee/North Carolina", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="WY", "Wyoming", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="HI", "Hawaii", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="AR", "Arkansas", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="MI", "Michigan", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="KY", "Kentucky", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="WA", "Washington", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="VA", "Virginia", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="ND", "North Dakota", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="MN", "Minnesota", df_parks["State_FullName"])

df_parks["State_FullName"] = np.where(df_parks["State"]=="WY, MT, ID", "Wyoming, Montana,Idaho", df_parks["State_FullName"])

#To see Acres under each Region

sns.barplot(x=df_parks["Region"], y=df_parks["Acres"],hue =df_parks["Region"], data=df_parks, errwidth=0)  



#Conclusion: 

#1. The West region has the most number of Acres under it

#2. Northeast region has the least. Are we missing data?
#To see number of parks in each State, under each region

#df_MW_grp_data = df_parks[df_parks["Region"]=="MidWest"].groupby(["State_FullName"])[[ "Acres"]].sum()



#To see number of parks in each "West" region, as this has the highest number of acres

df_MW_grp_data = df_parks[df_parks["Region"]=="West"]



#At the URL below, we can see the names we can pass for Palettes.

#URL: https://matplotlib.org/examples/color/colormaps_reference.html

#To check further: Some (like Inferno) are accepted in lowercase while others (like Grey) are accepted with first letter being capitalised.

sns.countplot(y= df_MW_grp_data["State_FullName"], data = df_MW_grp_data, palette="Greens")



#Conclusion

#1. Alaska has the highest number of parks

#2. There are few states with least number of parks

#3. There are some parks mapped across states. These are valid??
#Sum of acres per state, in "West" region, are got below



print("Park acres (in millions) per state, in West region:")

df_MW_grp_data.groupby("State_FullName", sort="True")["Acres"].sum()/1000000 
#Let us analyze the Species in National Parks now.



#Load Dataset

df_species = pd.read_csv("../input/species.csv")
#Species ID                         National Parks Service park code.                       String

#Park Name                          Park in which the species appears                       String

#Category                           One of Mammal, Bird, etcetera                           String

#Order                              The scientific order the species belongs to             String

#Family                             The scientific family the species belongs to            String

#Scientific Name                    Full scientific species name                            String

#Common Names                       Usual common name(s) for the species. Comma-delimited   String

#Record Status                      Usually "Approved"                                      String

#Occurrence                         Whether or not the species presence in the park has     String

                                   #been confirmed (one of "Present", "Not Confirmed", 

                                   #"Not Present (Historical)").

#Nativeness                         Whether the species is native to the area or a          String

                                   #non-native/invasive

#Abundance                          Commonality of sightings                                String

#Seasonality                        When the species can be found in the park               String

                                   #Blank if the species is found there year-round

#Conservation Status                IUCN species conservation status                        String



df_species.head()
#We create a combined dataset of Parks and Species, to visualize further

df_park_species = pd.merge(df_parks, df_species, on="Park Name")
#Drop columns I may not use

df_park_species.drop(["Park Code","Species ID","Unnamed: 13"], axis=1, inplace=True)
#Let us look at the diversity in the Category column

dr_Category_Count = pd.DataFrame(df_park_species.groupby("Category").size(), columns=["Count"])

dr_Category_Count = dr_Category_Count.reset_index()

sns.stripplot(y=dr_Category_Count["Category"], x= dr_Category_Count["Count"], jitter=True, size=10, edgecolor="green")



#Conclusion:

#1. Vascular Plant counts are more than 60,000. Plants dominate!

#2. Carb/Lobster/Shrimp, Amphibian, Slug/Snail,Algae, Spider/Scorpion seem less
#Let us understand the type of species ("Order") relative to the Category they belong to. 



#1. The next row will create a Multi-Index

dr_Cat_Order = pd.DataFrame(df_park_species.groupby(["Category","Order"],as_index=False).size(), columns=["Count"])

dr_Cat_Order.reset_index(inplace=True)



#2. We take the top 'N' rows to analyze which are the dominant ones

dr_CO = dr_Cat_Order.nlargest(16,"Count")



#Let us visualize the data of top 16, across categories

#These are the top 16 by numbers, not top 16 per category.

sns.factorplot(x = "Count", y= "Order",  col= "Category", data= dr_CO, kind="bar",size=4, aspect=.7, legend_out=True, palette="Greys" )

#Let us look at the Nativeness of a species to a region

dr_NativeToRegion = pd.DataFrame(df_park_species.groupby(["Region","Category"]).size(),columns=["Count"])

dr_NativeToRegion.reset_index(inplace=True)

dr_NativeToRegion



#Visualize Nativity of the category to a region

sns.factorplot(x= "Count", y="Category",col="Region", kind="bar",  data=dr_NativeToRegion, size=4, aspect=.95,  palette="Greens" )



#Conclusion:

#1. It looks like the "West" region has more native species. Since we observed "West" region to have most Parks, 

#   can we assume the "West" has more native species? Let us plot by region and see 
#Let us look at the Nativeness of a species to a region

dr_NativeToRegion = pd.DataFrame(df_park_species[df_park_species["Nativeness"]=="Native"].groupby(["Region","Category"]).size(),columns=["Count"])

dr_NativeToRegion.reset_index(inplace=True)

dr_NativeToRegion



#Visualize Nativity of the category to a region

sns.factorplot(x= "Count", y="Category",col="Region", kind="bar",  data=dr_NativeToRegion, size=4, aspect=.95,  palette="Greens" )



#Conclusion:

#1. It looks like the "West" region has more native species. Since we observed "West" region to have most Parks, 

#   can we assume the "West" has more native species? We shall look at these,after below plotting
#Let us analyze region wise and see if "West" region has most of the native species



#Northeast

print("Northeast")

sns.barplot(x=dr_NativeToRegion[dr_NativeToRegion.Region=="Northeast"]["Count"], y=dr_NativeToRegion[dr_NativeToRegion.Region=="Northeast"]["Category"], palette="Greys" )

plt.show()



#MidWest

print("MidWest")

sns.barplot(x=dr_NativeToRegion[dr_NativeToRegion.Region=="MidWest"]["Count"], y=dr_NativeToRegion[dr_NativeToRegion.Region=="MidWest"]["Category"], palette="Greys" )

plt.show()



#West

print("West")

sns.barplot(x=dr_NativeToRegion[dr_NativeToRegion.Region=="West"]["Count"], y=dr_NativeToRegion[dr_NativeToRegion.Region=="West"]["Category"], palette="Greys" )

plt.show()



#South

print("South")

sns.barplot(x=dr_NativeToRegion[dr_NativeToRegion.Region=="South"]["Count"], y=dr_NativeToRegion[dr_NativeToRegion.Region=="South"]["Category"], palette="Greys" )

plt.show()



#Conclusion:

#1. Except Northeast, other regions have almost equal number of native species.

#2. Vascular plants seem to comprise most of the native species, across "Regions".
#Now that we have looked at the "Nativeness" per region, let us analyse what Nativeness means, in terms of availability



#Let us look at the Nativeness of a species to a region

dr_NativeEffectiveness = pd.DataFrame(df_park_species[df_park_species["Nativeness"]=="Native"].groupby(["Region","Category","Nativeness","Occurrence","Abundance"]).size(),columns=["Count"])

dr_NativeEffectiveness.reset_index(inplace=True)

dr_NativeEffectiveness



#Visualize how Nativity affects the presence

sns.factorplot(x= "Count", y="Abundance",col="Region", kind="bar",  data=dr_NativeEffectiveness, size=4, aspect=.95,  palette="Greys", errwidth=0 )



#Conclusion:

#1. Though "Native" to a region, there are different 'types' of availability.

#2. Not all species are abundant

#3. For MidWest, South and West regions, "Uncommon" seem to be high. "Native" to a region and classified uncommon??

#4. For South region, "Unknown" is higher
#Let us understand the composition of top 2 groups in "West" region



print("Analysis of Uncommon and Unknown Categories native to \"West\" Region:")

list(dr_NativeEffectiveness[(dr_NativeEffectiveness["Region"]=="West")& (dr_NativeEffectiveness["Abundance"].isin(["Uncommon","Unknown"]))].groupby(["Abundance"])["Category","Occurrence","Abundance", "Count"])



#Conclusion

#1. Vascular Plant and Birds, though native to West region, comprise of top 2 in the 'Uncommon' group

#2. Vascular Plant and Insects, though native to West region, comprise of top 2 in the 'Unknown' group
#What is being done for conservation of native species, in West and South regions?



dr_NativeEffectiveness = pd.DataFrame(df_park_species[(df_park_species["Nativeness"]=="Native") & (df_park_species["Region"].isin(["West","South"]))].groupby(["Region","Category","Nativeness","Occurrence","Abundance","Conservation Status"]).size(),columns=["Count"])

dr_NativeEffectiveness.reset_index(inplace=True)

dr_NativeEffectiveness



#Visualize how Nativity affects the presence

sns.factorplot(x= "Count", y="Abundance",hue="Region", col="Conservation Status", kind="bar", col_wrap=2, data=dr_NativeEffectiveness, size=5, aspect=.9, palette="Greens", legend_out=True, errwidth=0 )



#Conclusion:

#1. We see lot of native species tagged as "Species of Concern", which is worrying

#2. In West region, 'Rare' category tops the endangered list
# For further analysis, we could look at Seasonality and how this affects the data above..

# To summarize what we have analyzed thus far:



# The West region has the most number of Acres under it

# Northeast region has the least. Are we missing data?



# Alaska has the highest number of parks

# There are few states with least number of parks

# There are some parks mapped across states. These are valid??



# Vascular Plant counts are more than 60,000. Plants dominate!

# Carb/Lobster/Shrimp, Amphibian, Slug/Snail,Algae, Spider/Scorpion seem less



# It looks like the West region has more native species, as the West region has most Parks.

# Can we assume the West has more native species? Not by what we saw



# Except Northeast, other regions have almost equal number of native species

# Vascular plants seem to comprise most of the native species, across "Regions"



# Though "Native" to a region, there are different 'types' of availability, i.e., not all species are abundant

# For MidWest, South and West regions, "Uncommon" seem to be high. But being "Native" to a region and classified uncommon, is this right??

# For South region, "Unknown" is higher



# Vascular Plant and Birds, though native to West region, comprise of top 2 in the 'Uncommon' group

# Vascular Plant and Insects, though native to West region, comprise of top 2 in the 'Unknown' group



# We see lot of native species tagged as "Species of Concern", which is worrying

# In West region, 'Rare' category tops the endangered list
