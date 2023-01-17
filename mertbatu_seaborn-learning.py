import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from collections import Counter



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
percent_over25completed_hs = pd.read_csv('../input/PercentOver25CompletedHighSchool.csv', encoding = 'windows-1252')

kill = pd.read_csv('../input/PoliceKillingsUS.csv', encoding = 'windows-1252')

median_household_income = pd.read_csv('../input/MedianHouseholdIncome2015.csv', encoding = 'windows-1252')

race = pd.read_csv('../input/ShareRaceByCity.csv', encoding = 'windows-1252')

percent_people_povertylevel = pd.read_csv('../input/PercentagePeopleBelowPovertyLevel.csv', encoding = 'windows-1252')

percent_people_povertylevel.head() # Getting dataframe sample for general information about data.
percent_people_povertylevel.info() # Reviewing general information about data. 
percent_people_povertylevel.poverty_rate.value_counts() # Checking data for unclean datas.
percent_people_povertylevel.poverty_rate.replace('-',0.0, inplace = True) # '-' is unknown value. So it was ignored.
percent_people_povertylevel.poverty_rate.value_counts() # Cleaned and smooth data is obtained.
percent_people_povertylevel['Geographic Area'].unique() # Printing unique datas.
len(percent_people_povertylevel['Geographic Area'].unique()) # Number of unique datas.
# Poverty rate of each state. Preparing dataframe for plotting

percent_people_povertylevel.poverty_rate.replace('-', 0.0, inplace = True)

percent_people_povertylevel.poverty_rate = percent_people_povertylevel.poverty_rate.astype(float)

area_list = list(percent_people_povertylevel['Geographic Area'].unique())

area_poverty_ratio = []



for i in area_list:

    x = percent_people_povertylevel[percent_people_povertylevel['Geographic Area'] == i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)



data = pd.DataFrame({'area_list':area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)



# Visualization/plotting



plt.figure(figsize=(15,10))

sns.barplot( x = sorted_data['area_list'], y = sorted_data['area_poverty_ratio'])

plt.xticks( rotation = 45)

plt.xlabel('States')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')

plt.show()
kill.head()
kill.info()
kill.name.value_counts()
#kill.name.replace("TK TK" and "TK Tk",0, inplace = True)
kill.name.value_counts()
separate = kill.name[kill.name != 'TK TK'].str.split()

a,b=zip(*separate)

name_list = a+b

name_count= Counter(name_list) # counter function is imported before.

most_common_names = name_count.most_common(15)

x,y = zip(*most_common_names)

x,y = list(x), list(y)



# Visualization



plt.figure(figsize = (15,10))

sns.barplot( x =x , y =y, palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
percent_over25completed_hs.head()
percent_over25completed_hs.info()
percent_over25completed_hs.percent_completed_hs.value_counts()
# High school graduation rate of the population that is older than 25 in states

percent_over25completed_hs.percent_completed_hs.replace(['-'],0.0, inplace = True)

percent_over25completed_hs.percent_completed_hs = percent_over25completed_hs.percent_completed_hs.astype(float)

area_list = list(percent_over25completed_hs['Geographic Area'].unique())

area_highschool = []



for i in area_list:

    x = percent_over25completed_hs[percent_over25completed_hs["Geographic Area"] == i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

data = pd.DataFrame({'area_list':area_list, 'area_highschool_ratio': area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending = True)).index.values

sorted_data2 = data.reindex(new_index)



# Visualization



plt.figure(figsize = (15,10))

sns.barplot( x = sorted_data2['area_list'], y = sorted_data2['area_highschool_ratio'])

plt.xticks( rotation = 45)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")

plt.show()
race.head()
race.info()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

race.replace(['-'], 0.0, inplace = True)

race.replace(['(X)'], 0.0, inplace = True)

race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(race['Geographic area'].unique())

share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:

    x = race[race['Geographic area'] == i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black)/len(x))

    share_native_american.append(sum(x.share_native_american)/len(x))

    share_asian.append(sum(x.share_asian)/len(x))

    share_hispanic.append(sum(x.share_hispanic)/len(x))

    

# Visualization



f, ax = plt.subplots( figsize = (9,15))

sns.barplot( x = share_white , y = area_list , color = 'green', alpha = 0.5, label = 'White' )

sns.barplot( x = share_black , y = area_list , color = 'blue', alpha = 0.7, label = 'Black' )

sns.barplot( x = share_native_american , y = area_list , color = 'cyan', alpha = 0.6, label = 'Native American' )

sns.barplot( x = share_asian , y = area_list , color = 'yellow', alpha = 0.6, label = 'Asian' )

sns.barplot( x = share_hispanic , y = area_list , color = 'red', alpha = 0.6, label = 'Hispanic' )



ax.legend( loc = 'lower right', frameon = True) # Visibility of legend

ax.set( xlabel = 'Percentage of Races', ylabel = 'States', title = "Percentage of State's Population According to Races")

plt.show()
sorted_data.head()
sorted_data2['area_highschool_ratio']
# high school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max(sorted_data['area_poverty_ratio']) # Normalization

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max(sorted_data2['area_highschool_ratio'])



data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']], axis = 1)

data.sort_values('area_poverty_ratio', inplace = True)



# Visualization

f,ax1 = plt.subplots( figsize = (20,10))

sns.pointplot( x = 'area_list', y = 'area_poverty_ratio' , data = data, color ='lime', alpha = 0.8)

sns.pointplot( x = 'area_list', y = 'area_highschool_ratio', data = data, color = 'red', alpha = 0.8)

plt.text( 40,0.6,'high school graduate ratio', fontsize = 17, color = 'red', style = 'italic')

plt.text(40,0.55,'poverty ratio',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('States', fontsize = 15, color = 'blue')

plt.ylabel('Values', fontsize =15, color = 'blue')

plt.title("High School Graduate  VS  Poverty Rate", fontsize = 20, color = 'blue')

plt.grid()
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# joint kernel density (kde = kernel density estimation)

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }



sns.jointplot( data.area_poverty_ratio, data.area_highschool_ratio, kind = 'kde', size = 7)

plt.savefig('graph.png')

plt.show()
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one



sns.jointplot( data['area_poverty_ratio'], data['area_highschool_ratio'], data = data, size =5, ratio = 3, color = 'r')

plt.show()
kill.race.head(15)
kill.race.value_counts()
# Race rates according in kill data

kill.race.dropna( inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

size = kill.race.value_counts().values



plt.figure(figsize = (7,7))

plt.pie( size, explode = explode, labels = labels, colors = colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# lmplot 

# Show the results of a linear regression within each dataset

plt.figure(figsize = (7,7))

sns.lmplot( x = "area_poverty_ratio", y = "area_highschool_ratio", data = data)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# cubehelix plot

plt.figure(figsize = (7,7))

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade = True, cut = 3)

plt.show
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

plt.figure( figsize = (7,7))

#pal = sns.cubehelix_palette( 2, rot = -.5, dark = .3)

sns.violinplot( data = data, palette = pal, inner = "points")

plt.show()
data.corr()
#correlation map

# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots( figsize = (5,5))

sns.heatmap( data.corr(), annot = True, linewidths = .5, linecolor = 'red', fmt = '.1f', ax = ax)

plt.show()
kill.head()
kill.manner_of_death.unique()
# manner of death : shot, shot and Tasered

# gender

# age

# Plot the orbital period with horizontal boxes

sns.boxplot( x = 'gender', y = 'age' , hue = 'manner_of_death', data = kill, palette = "PRGn")

plt.show()
sns.swarmplot( x = "gender", y = "age", hue = "manner_of_death", data = kill)

plt.show()
sns.pairplot(data)

plt.show()
kill.gender.value_counts()
kill.head()
sns.countplot( kill.gender)

plt.title("gender",color = 'blue',fontsize=15)

plt.show()
# kill weapon

armed = kill.armed.value_counts()

plt.figure( figsize = (10,7))

sns.barplot( x = armed[:7].index, y =armed[:7].values)

plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)

plt.show()
# Age of killed people



above25 = ["above 25" if i >= 25 else "below 25" for i in kill.age]

df = pd.DataFrame({"age":above25})

sns.countplot(x = df.age)

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)

plt.show()
# Race of killed people

sns.countplot( data = kill, x = 'race' )

plt.title('Race of killed people',color = 'blue',fontsize=15)

plt.show()
# Most dangerous cities

cities = kill.city.value_counts()

plt.figure(figsize=(10,7))

sns.barplot( x = cities[:7].index, y = cities[:7].values)

plt.xticks(rotation=45)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)

plt.show()
# most dangerous states

states = kill.state.value_counts()

plt.figure( figsize = (10,7))

sns.barplot( x = states[:20].index, y = states[:20].values)

plt.title('Most dangerous state',color = 'blue',fontsize=15)

plt.show()
# Having mental ilness or not for killed people

sns.countplot(kill.signs_of_mental_illness)

plt.xlabel('Mental illness')

plt.ylabel('Number of Mental illness')

plt.title('Having mental illness or not',color = 'blue', fontsize = 15)

plt.show()
# Threat types

sns.countplot(kill.threat_level)

plt.xlabel('Threat Types')

plt.title('Threat types',color = 'blue', fontsize = 15)

plt.show()
# Flee types

sns.countplot(kill.flee)

plt.xlabel('Flee Types')

plt.title('Flee types',color = 'blue', fontsize = 15)

plt.show()
# Having body cameras or not for police

sns.countplot(kill.body_camera)

plt.xlabel('Having Body Cameras')

plt.title('Having body cameras or not on Police',color = 'blue',fontsize = 15)

plt.show()
# Kill numbers from states in kill data

sta = kill.state.value_counts().index[:10]

sns.barplot(x=sta,y = kill.state.value_counts().values[:10])

plt.title('Kill Numbers from States',color = 'blue',fontsize=15)

plt.show()