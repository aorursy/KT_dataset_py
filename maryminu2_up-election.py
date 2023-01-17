# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

% matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import  input file

Data = pd.read_csv("../input/up_res.csv")
# Lets see the Data 

Data.head() # this will give us the top 5(by default ) rows of dataset
Data.info() # it will give us the information about the data columns
# Now lets count total  number of votes

print("The numbers of votes voted in UP election result are:",Data["votes"].sum() )
# Now lets count the number of districts in UP

print(" The number of districts in UP is :" ,len(Data['district'].unique()))

# in this the unique will give only unique values in district

# len will count the number of unique values
# Now lets count the number of constituencies in UP 

# same as we did in the the district

print(" The number of constituencies in UP is :" ,len(Data['ac'].unique()))
# there is 403 assembly consistuencies in UP within 75 districts

# Lets Count How much Constituency in a disrict

Data_districts= Data.groupby(["district"])["ac"].nunique()

# In this we grouped the data by districs wise and then we counted the uniques values of constitutncy

Data_districts=Data_districts.reset_index().sort_values(by="ac",ascending=False).reset_index(drop=True)

# Now we sorted values on the basis of count of assembly seats
# let see which districts are at the top

Data_districts.head()
# Allabhad is with the most number of Assembly seats 

# Now look at the bottom who are down in the list

Data_districts.tail(7)

# here tail will give the bottom of Table
# from this we can see that the there are district more than 12 Assembly seats and also with only 2 seats 

# not fair distibution

# lets see the mode of this 

Data_districts.ac.value_counts() # this is to count the number of values
# Let's Count the number of votes per district

Data_Districts_votes = Data.groupby("district")['votes'].sum().reset_index().sort_values(by="votes",ascending= False).reset_index(drop=True)

# group by district and sum the number of votes
Data_Districts_votes.head(10)
# Allahabad is at the top in both case so it may be due to a large number of assembly seats 

# lest calculate the avearage number of votes per seat District wise
Data_districts= pd.merge(Data_districts,Data_Districts_votes,on="district")

# in this we are merging two data set one contain the number of votes and one contain the number of Assembly seats
Data_districts.head()
Data_districts["Average Votes Per assembly"]= (Data_districts['votes']/Data_districts["ac"]).astype(int)

# Making a new coloumn which contains the average number of votes per assembely Seats
Data_districts.sort_values(by="Average Votes Per assembly",ascending=False).reset_index(drop=True)
# Lets Count the number of Candidate per assembly seat

Data_Candidate= Data.groupby("ac")['candidate'].count().reset_index().sort_values(by="candidate",ascending=False).reset_index(drop=True)
Data_Candidate.head(15)
# Let's explore the number of votes per phase 

Votes_Phase= Data.groupby('phase')['votes'].sum().reset_index().sort_values(by="votes",ascending=False).reset_index(drop=True)

Votes_Phase
sns.barplot(x='phase',y='votes',data=Votes_Phase) # to plot the votes in each phase

plt.title("No. Of Votes In each Phase")
Assembly_phase =  Data.groupby("phase")["ac"].count().reset_index().sort_values(by="ac",ascending=False).reset_index(drop=True)

sns.barplot(x='phase',y='ac',data=Assembly_phase)
# Let's see which parties are there 

Data.party.unique() # the parties which are participating in the elections
# Here none of the above means nothing we can convert them into also others 

Data.party.replace("None of the Above","others",inplace=True)
Data.party.unique()
# Vote Distribution of Parties

plt.figure(figsize=(10,8))

sns.pointplot(x='party',y='votes',data=Data)
plt.figure(figsize=(10,8))

sns.boxplot(x='party',y='votes',data=Data)
# Let see the patteren of votes get by parties

Votes_party=Data.groupby("party")['votes'].sum().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True)

Votes_party
# lest plot the barplot of it 

sns.barplot(x='party',y='votes',data=Votes_party)

plt.title("No oF votes got by parties")

plt.xticks(rotation=90)
# Let's see the number of votes get by parties phase wise

No_of_phase= len(Data.phase.unique())

fig=plt.subplots(figsize=(8,10*(No_of_phase+1)))

for i in range(No_of_phase):

    index_values= Data[Data["phase"]==i+1].index.values

    phase_votes= Data.ix[index_values,:] # getting all the value by phase wise

    votes_party_phase= phase_votes.groupby('party')['votes'].sum().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True)

    plt.subplot(No_of_phase+1,1,i+1) # No of Phase +1 is for total no of plots 

    sns.barplot(x='party',y='votes',data=votes_party_phase)

    plt.subplots_adjust(hspace=.3)

    plt.xticks(rotation=90)

    plt.title("Phase {}".format(i+1))
#Lets find the number of Assembly seats won by parties

# This thing I am doing in my way other suggestion will be helpful

# Please comment if you know any other way for it 

Winner= Data.groupby(["ac"])['votes'].max().reset_index()

# This will give us the maximum number of votes for every assembly seat

Winner2=pd.merge(Winner,Data,on=['ac','votes'],how="left",copy=False)



# Now I am merging the this data with the our original data to get all values



winner_party=Winner2.groupby(['party'])['candidate'].count().reset_index() # Now counting the seats won by a party

print(winner_party)

sns.barplot(x='party',y='candidate',data=winner_party)
# We can see these for phase wise

Winner2.groupby(['phase','party'])['candidate'].count()
# Let's do it for who are at the last Postion 

Last_position= Data.groupby(["ac"])['votes'].min().reset_index().sort_values(by='votes').reset_index(drop=True)

Last_position2=pd.merge(Last_position,Data,on="votes",how="left",copy=False)

Last_position2=Last_position2.drop_duplicates('ac_x').reset_index(drop=True) # drop any duplicate if it is there
Last_position2[["ac_x",'candidate','party','votes']]
# Now lets Find who are at the second positions

Second_place=Data.groupby("ac")['votes'].nlargest(2).reset_index() # nlargest(2) will give us the two largest value for each category

Second_place1 = Second_place.groupby('ac')['votes'].min().reset_index().sort_values(by='votes',ascending=False).reset_index(drop=True) # from this we will get the miinimum of those two

#print(second_place1) you can do it for your confirmation

# Now we will merge it with our oringinal data so to get all the fields here



Second_place2=pd.merge(Second_place1,Data,on=['ac','votes'],how="left",copy=False)

#print(Second_place2) you can do it for your confirmation

Second_party=Second_place2.groupby(['party'])['candidate'].count().reset_index() # Now counting the seats won by a party

print(Second_party)

sns.barplot(x='party',y='candidate',data=Second_party)

 
# now I want to see the difference b/w candidate who won the election and the one who finished second

winner= Winner2[['ac','votes']] # here we got the data of winners

second_place= Second_place2[['ac','votes']] # here we got the data of second_place

Winner_comparison= pd.merge(winner,second_place,on='ac')

# Now get the difference b/w the these two position

Winner_comparison["Difference"]=Winner_comparison['votes_x']-Winner_comparison['votes_y']

Winner_comparison.sort_values(by="Difference",ascending=False).reset_index(drop=True)
#lets plot a graph for more information

x=Winner_comparison["Difference"]

sns.distplot(x)
# reduce the xlimit to clear view



plt.figure(figsize=(12,10))

plt.xlim(0,100000)

sns.distplot(x)
# Let's divide UP In four regions

# these list's name repersent the name the region and element repersent distric in them 

# I know it all because I am from neighbouring state of UP

Harit_Pardesh=['Saharanpur',

'Shamli',

'Muzaffarnagar',

'Bijnor',

'Moradabad',

'Sambhal',

'Rampur',

'Amroha',

'Meerut',

'Baghpat',

'Ghaziabad',

'Hapur',

'Gautam Buddha Nagar',

'Bulandshahr',

'Aligarh',

'Hathras',

'Mathura',

'Agra',

'Firozabad',

'Kasganj',

'Etah',

'Mainpuri',

'Budaun',

'Bareilly',

'Pilibhit',

'Shahjahanpur'

]



Avadh_Pardesh=['Lakhimpur Kheri',

'Sitapur',

'Hardoi',

'Unnao',

'Lucknow',

'Raebareli',

'Farrukhabad',

'Kannauj',

'Etawah',

'Auraiya',

'Kanpur Dehat',

'Kanpur Nagar',

'Barabanki'

]



BundelKhand = ['Jalaun',

'Jhansi',

'Lalitpur',

'Hamirpur',

'Mahoba',

'Banda',

'Chitrakoot'

]



Purvanchal= ['Amethi',

'Sultanpur',

'Fatehpur',

'Pratapgarh',

'Kaushambi',

'Allahabad',

'Faizabad',

'Ambedkar Nagar',

'Bahraich',

'Shravasti',

'Balarampur',

'Gonda',

'Siddharthnagar',

'Basti',

'Sant Kabir Nagar',

'Maharajganj',

'Gorakhpur',

'Kushinagar',

'Deoria',

'Azamgarh',

'Mau',

'Ballia',

'Jaunpur',

'Ghazipur',

'Chandauli',

'Varanasi',

'Sant Ravidas Nagar',

'Mirzapur',

'Sonbhadra'

]

print("No of District in Harit Pardesh:",len(Harit_Pardesh))

print("No of District in Purvanchal:",len(Purvanchal))

print("No of District in Avadh Pardesh:",len(Avadh_Pardesh))

print("No of District in BundelKhand:",len(BundelKhand))
mapper={} # now taking a empty dictonary

for i in Harit_Pardesh: # Now iterating through list and adding districts as key and assigning them value Region

    mapper[i]="Harit Pardesh"

for i in Purvanchal: # Same as above

    mapper[i]="Purvanchal"

for i in Avadh_Pardesh:

    mapper[i]="Avadh Pardesh"

for i in BundelKhand:

    mapper[i]="BundelKhand"

    
Data['Region']=Data["district"].map(mapper)  # Now mapping districts to region using mapper dictonary
# Just rechecking the mapping is it correct or not so again counting the number of districts per region

District_Region=Data.groupby("Region")["district"].nunique().reset_index()

District_Region
# Let's Now see vote Per Region

Region_Votes = Data.groupby("Region")["votes"].sum().reset_index().sort_values(by=['votes']).reset_index(drop=True)

Region_Votes
# Lets plot a pie plot of it

plt.figure(figsize=(8,8))

plt.pie(Region_Votes["votes"],labels=Region_Votes["Region"] ,autopct='%1.1f%%',shadow=True,explode=[0.10,0.10,0.10,0.10])

Votes_Region_per_district = pd.merge(Region_Votes,District_Region,on="Region")

Votes_Region_per_district["Votes_Per_District"]=(Votes_Region_per_district["votes"]/Votes_Region_per_district["district"])*100

Votes_Region_per_district.Votes_Per_District=Votes_Region_per_district.Votes_Per_District.astype(int)

Votes_Region_per_district.sort_values(by="Votes_Per_District",ascending=False).reset_index(drop=True)
# Now let's see the number of seat won by party resion wise

Winner= Data.groupby(["ac"])['votes'].max().reset_index()

Winner2=pd.merge(Winner,Data,on=['ac','votes'],how="left",copy=False)

Winner_Region = Winner2.groupby(['Region','party'])['candidate'].count()

Winner_Region