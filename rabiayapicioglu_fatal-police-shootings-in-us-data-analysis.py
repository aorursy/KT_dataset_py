import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Read datas

median_house_hold_in_come = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty_level = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25_completed_highSchool = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

share_race_city = pd.read_csv('../input/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding="windows-1252")
percentage_people_below_poverty_level.head()
percentage_people_below_poverty_level['Geographic Area'].unique()
# Poverty rate of each state

percentage_people_below_poverty_level.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty_level.poverty_rate = percentage_people_below_poverty_level.poverty_rate.astype(float)

area_list = list(percentage_people_below_poverty_level['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty_level[percentage_people_below_poverty_level['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
# Most common 15 Name or Surname of killed people

#separate=kill.name[ kill.name !='TK TK']

#print( type(separate) ) // series



#separate=kill.name[ kill.name !='TK TK' ].str # now they are string objects series

#print( type(separate) )



separate=kill.name[ kill.name !='TK TK' ].str.split()

print( separate.loc[:5] ) #We select the names that are not tk tk and then we convert it 

                          #into the string object series then by using split method we separated name and surname

    

a,b=zip( *separate ) #unzipping the names and surnames

print(a[:5]) #This gives the names tuple

print(b[:5]) #This gives the surname tuple





names_list=a+b #we are adding the second list at the end of the first list // add b to end of a

#print( names_list[0] )

#print( type( names_list )) after concatenation we got another tuple

#print( names_list )

name_count=Counter( names_list ) #it produces a dictionary that indicates the names and frequencies

most_common_names=name_count.most_common(15) # then find the most frequent name or surname

print( most_common_names ) # we see the dictionary at the output

x,y=zip(*most_common_names)# again unzip but this time we are unzippin the names and their frequencies 

#print( type(x)) // this is tuple we'll convert it to list initiallly

x,y=list(x),list(y) # x holds the names and surnames

print(x,y) # y holds the frequencies



plt.figure( figsize=(15,10))

ax=sns.barplot( x=x ,y=y ,palette=sns.cubehelix_palette(len(x)) )

plt.xlabel('Name or Surname')

plt.ylabel('Frequency')

plt.title("Most common 5 names or surnames ")







    
percent_over_25_completed_highSchool.head()
percent_over_25_completed_highSchool.info()
# High school graduation rate of the population that is older than 25 in states

percent_over_25_completed_highSchool.percent_completed_hs.unique()

#how do we know about these invalid characters in the dataset,

#print(percent_over_25_completed_highSchool.percent_completed_hs.value_counts())

percent_over_25_completed_highSchool.percent_completed_hs.replace(['-'],0.0,inplace = True)



percent_over_25_completed_highSchool.percent_completed_hs=percent_over_25_completed_highSchool.percent_completed_hs.astype( float )

area_list=list(percent_over_25_completed_highSchool['Geographic Area'].unique()) # we see the unique states in America



avg_grad_rates=[]



for state in area_list:

    x=percent_over_25_completed_highSchool[ percent_over_25_completed_highSchool['Geographic Area']==state ]

    avg_ratio=sum( x.percent_completed_hs )/len( x )

    avg_grad_rates.append(avg_ratio)



data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':avg_grad_rates})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)

# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 90)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
share_race_city.head()
share_race_city.info()
share_race_city.share_white.unique() # directly convert it into float ,there's no '-'

share_race_city.share_black.unique() # directly convert it into float ,there's no '-'

#... it goes on like that but we know from our previous experiences that we had '-' and '(X)' we change them directly from the dataframe

share_race_city.replace(['-'],0.0,inplace = True)

share_race_city.replace(['(X)'],0.0,inplace = True)



share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = share_race_city.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

#we change the types of indicated column names with all rows coming with them after loc ,astype( float )



area_list=list( share_race_city['Geographic area'].unique() )



share_white=[]

share_black=[]

share_native_american=[]

share_asian=[]

share_hispanic=[]



for each in area_list:

    x=share_race_city[ share_race_city['Geographic area']==each ]

    share_white.append( sum(x.share_white)/len(x) )

    share_black.append( sum(x.share_black)/len(x) )

    share_native_american.append( sum(x.share_native_american)/len(x) )

    share_asian.append( sum(x.share_asian)/len(x) )

    share_hispanic.append( sum(x.share_hispanic)/len(x) )



plt.figure( figsize=(9,15))

sns.barplot( x=share_white ,y=area_list,color='green',alpha=0.5,label='White')

sns.barplot( x=share_black ,y=area_list,color='blue',alpha=0.5,label='African American')

sns.barplot( x=share_native_american ,y=area_list,color='cyan',alpha=0.5,label='Native American')

sns.barplot( x=share_asian ,y=area_list,color='yellow',alpha=0.5,label='Asian')

sns.barplot( x=share_hispanic ,y=area_list,color='red',alpha=0.5,label='Hispanic')

    

ax.legend(   loc='lower right',frameon = True)     # legendlarin gorunurlugu

ax.set(xlabel='Percentage of Races', ylabel='States',title = "Percentage of State's Population According to Races ")

    



#print(sorted_data[:5])

# in this sorted_data we were keeping the areas and area_poverty_ratio

#but these values has a very large scale so to see the values more precise we'll divide the area_poverty_ratio to it's maximum

#and then we get the values between 0 and 1

sorted_data['area_poverty_ratio']=sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio'])

#print(sorted_data[:5])



#print( sorted_data2[:5])

#and sorted_data2 we were keeping the areas and area high school rates 

#and we do the same thing sor sorted_data2

sorted_data2['area_highschool_ratio']=sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])

#print( sorted_data2[:5])



data=pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)

#we concatenate the whole sorted_data and the sorted_data2's area_highschool_ratio column

#print( data[:5] ) # Here we can se the concatenated dataframe but unsorted



data.sort_values('area_poverty_ratio',inplace=True) # sort the given values according to area poverty rate



#print( data[:5] )



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',y='area_poverty_ratio',data=data,color='pink',alpha=0.8)

sns.pointplot(x='area_list',y='area_highschool_ratio',data=data,color='purple',alpha=0.8)

plt.text(40,0.6,'high school graduate ratio',color='pink',fontsize = 17,style = 'italic')

plt.text(40,0.55,'poverty ratio',color='purple',fontsize = 18,style = 'italic')

plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='black')

plt.grid()
data.head()
g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()



#The darker places in the graph indicates the comment of this graph.We see while the area_highschool_ratio increases,

#area poverty rate is decreasing
g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,size=8, ratio=2, color="r")

#while the area-poverty_rate is decreasing ,the area_highchool_rate is increasing in there
# Race rates according in kill data 

kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index



#Here let's have a look at value_counts()

print( type( kill.race.value_counts()) )

print( kill.race.value_counts())

#It is a list like,returns the numbers of each type

#W  1201  //white

#B  618   //black 

#H  423   //hispanic



#And now what is index ?



print( kill.race.value_counts().index )

#Index(['W', 'B', 'H', 'A', 'N', 'O'] // indexes are



colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

#explode : array-like, optional, default: None

#If not None, is a len(x) array which specifies the fraction of the radius with which to offset each wedge.



sizes = kill.race.value_counts().values

#[1201  618  423   39   31   28]



print(sizes)

# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()
sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3,color='purple')

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
#seaborn.cubehelix_palette(n_colors=6, start=0, rot=0.4, gamma=1.0, hue=0.8, light=0.85, dark=0.15, reverse=False, as_cmap=False)

#Make a sequential palette from the cubehelix system.

#This produces a colormap with linearly-decreasing (or increasing) brightness.

#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
print(kill.manner_of_death.unique())

sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()



#the average ages are closed to each other
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()



#if there is a clear separation between the blue points and orange points then we should use this feature it is more understandable
# pair plot

sns.pairplot(data)

plt.show()
# kill properties

# Manner of death

sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)

# kill weapon

armed = kill.armed.value_counts()

#print(armed)

plt.figure(figsize=(10,7))

sns.barplot(x=armed[:7].index,y=armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)
# age of killed people

above25 =['above25' if i >= 25 else 'below25' for i in kill.age]

df = pd.DataFrame({'age':above25}) # we've created a dataframe with one column named as age

#print(type(df.age))

sns.countplot(x=df.age) # one column of a data frame is called series with its column name  as a label

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)
# Race of killed people

sns.countplot(data=kill, x='race')

plt.title('Race of killed people',color = 'blue',fontsize=15)

# Most dangerous cities

city = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)
# most dangerous states

state = kill.state.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(x=state[:20].index,y=state[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]



#print( type(kill.state.value_counts().index))

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)