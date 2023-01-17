import numpy as np

#NumPy is a python library used for working with arrays.

#It also has functions for working in domain of linear algebra, fourier transform, and matrices.

#We have lists that serve the purpose of arrays, but they are slow.NumPy aims to provide an array object that is up to 50x faster that traditional Python lists.



import pandas as pd 

#Why pandas: you want to explore a dataset stored in a CSV on your computer. Pandas will extract the data from that CSV into a DataFrame — 

#a table, basically — then let you do things like:

#Calculate statistics and answer questions about the data, like: What's the average, median, max, or min of each column?

#Does column A correlate with column B?

#What does the distribution of data in column C look like?

#Clean the data by doing things like removing missing values and filtering rows or columns by some criteria

#Visualize the data with help from Matplotlib. Plot bars, lines, histograms, bubbles, and more.

#Store the cleaned, transformed data back into a CSV, other file or database



import os

#The OS module in python provides functions for interacting with the operating system.

#This module provides a portable way of using operating system dependent functionality.

#The *os* and *os.path* modules include many functions to interact with the file system.



import matplotlib.pyplot as plt

#Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.

plt.style.use("seaborn-whitegrid")

#plt.style.available : To see all the available style in matplotlib library



import seaborn as sns



from collections import Counter



import warnings

warnings.filterwarnings("ignore")



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

#UTF-8 is a variable-width character encoding standard 

#that uses between one and four eight-bit bytes to represent all valid Unicode code points.



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# Read datas

house_income = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding="windows-1252")

percentage_people_below_poverty= pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding="windows-1252")

percent_over_25= pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding="windows-1252")

race = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv', encoding="windows-1252")

kill = pd.read_csv('/kaggle/input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding="windows-1252")
race.head()
percentage_people_below_poverty.head()
percentage_people_below_poverty.info()
percentage_people_below_poverty['Geographic Area'].unique()
percentage_people_below_poverty.poverty_rate.replace(['-'],0.0,inplace = True)

percentage_people_below_poverty.poverty_rate.value_counts()
percentage_people_below_poverty.poverty_rate = percentage_people_below_poverty.poverty_rate.astype(float)

#race - kill - percent_over_25 - percentage_people_below_poverty - house_income
area_list = list(percentage_people_below_poverty['Geographic Area'].unique())

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_poverty[percentage_people_below_poverty['Geographic Area']==i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({'area_list': area_list,'area_poverty_ratio':area_poverty_ratio})

new_index = (data['area_poverty_ratio'].sort_values(ascending=False)).index.values

sorted_data = data.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data['area_list'], y=sorted_data['area_poverty_ratio'])

plt.xticks(rotation= 0)

plt.xlabel('Geographic Area')

plt.ylabel('Poverty Rate')

plt.title('Poverty Rate Given States')
kill.head()
kill.name.value_counts()
# The zip() function returns a zip object, which is an iterator of tuples where the first 

# item in each passed iterator is paired together, and then the second item in each passed iterator are paired together etc.

# If the passed iterators have different lengths, the iterator with the least items decides the length of the new iterator.

# If a single iterable is passed, zip() returns an iterator of tuples with each tuple having only one element.

# If multiple iterables are passed, zip() returns an iterator of tuples with each tuple having elements from all the iterables.

name = [ "Manjeet", "Nikhil", "Shambhavi", "Astha" ] 

roll_no = [ 4, 1, 3, 2 ] 

marks = [ 40, 50, 60, 70 ] 

  

# using zip() to map values 

mapped = zip(name, roll_no, marks) 

mapped = list(mapped) 

print ("The zipped result is : ",end="") 

print (mapped) 

print("\n") 

  

# unzipping values 

namz, roll_noz, marksz = zip(*mapped) 



print ("The unzipped results: \n") 

  

print ("The name list is : ",end="") 

print (namz) 

print ("The roll_no list is : ",end="") 

print (roll_noz) 

print ("The marks list is : ",end="") 

print (marksz) 
# Python Counter is a container that will hold the count of each of the elements present in the container. 

# The counter is a sub-class available inside the dictionary class.

# The Counter holds the data in an unordered collection, just like hashtable objects.

# Arithmetic operations like addition, subtraction, intersection, and union can be easily performed on a Counter.



list1 = ['x','y','z','x','x','x','y', 'z']

print(Counter(list1))

print("*****")

my_str = "Welcome to Guru99 Tutorials!"

print(Counter(my_str))
# Most common 15 Name or Surname of killed people

separate = kill.name[kill.name != 'TK TK'].str.split()  # I don't want to show TK TK. #Convert to string and split the names.

a,b = zip(*separate)                     

name_list = a+b                      

name_count = Counter(name_list)          

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)





plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people')
#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
percent_over_25.head()
percent_over_25.info()
# High school graduation rate of the population that is older than 25 in states

percent_over_25.percent_completed_hs.replace(['-'],0.0,inplace = True)

percent_over_25.percent_completed_hs = percent_over_25.percent_completed_hs.astype(float)



area_list = list(percent_over_25['Geographic Area'].unique())

area_highschool = []



for i in area_list:

    x = percent_over_25[percent_over_25['Geographic Area']==i]

    area_highschool_rate = sum(x.percent_completed_hs)/len(x)

    area_highschool.append(area_highschool_rate)

    

# sorting

data = pd.DataFrame({'area_list': area_list,'area_highschool_ratio':area_highschool})

new_index = (data['area_highschool_ratio'].sort_values(ascending=True)).index.values

sorted_data2 = data.reindex(new_index)



# visualization

plt.figure(figsize=(15,10))

sns.barplot(x=sorted_data2['area_list'], y=sorted_data2['area_highschool_ratio'])

plt.xticks(rotation= 0)

plt.xlabel('States')

plt.ylabel('High School Graduate Rate')

plt.title("Percentage of Given State's Population Above 25 that Has Graduated High School")
race.head()
race.info()
# Percentage of state's population according to races that are black,white,native american, asian and hispanic

race.replace(['-'],0.0,inplace = True)

race.replace(['(X)'],0.0,inplace = True)

race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']] = race.loc[:,['share_white','share_black','share_native_american','share_asian','share_hispanic']].astype(float)

area_list = list(race['Geographic area'].unique())



share_white = []

share_black = []

share_native_american = []

share_asian = []

share_hispanic = []



for i in area_list:      #Find the for each state

    x = race[race['Geographic area']==i]

    share_white.append(sum(x.share_white)/len(x))

    share_black.append(sum(x.share_black) / len(x))

    share_native_american.append(sum(x.share_native_american) / len(x))

    share_asian.append(sum(x.share_asian) / len(x))

    share_hispanic.append(sum(x.share_hispanic) / len(x))

    

# visualization

f,ax = plt.subplots(figsize = (9,15))

sns.barplot(x=share_white,            y=area_list,   color='green',    alpha = 0.5,    label='White' )

sns.barplot(x=share_black,            y=area_list,   color='blue',     alpha = 0.7,    label='African American')

sns.barplot(x=share_native_american,  y=area_list,   color='cyan',     alpha = 0.6,    label='Native American')

sns.barplot(x=share_asian,            y=area_list,   color='yellow',   alpha = 0.6,    label='Asian')

sns.barplot(x=share_hispanic,         y=area_list,   color='red',      alpha = 0.6,    label='Hispanic')



ax.legend(loc='lower right',frameon = True)     # legend opacity

ax.set(xlabel='Percentage of Races', ylabel='States', title = "Percentage of State's Population According to Races ")    

#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
sorted_data.head()
sorted_data2.head()
# High school graduation rate vs Poverty rate of each state

sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])

# ^^^ [1,2,3,4,5] vs  [600,700,800,900,1000] we should have similar scale, we need to normalize the data ^^^

# Divide to max number 0< [1,2,3,4,5]/5 <1  0< [600,700,800,900,1000]/1000 <1



data = pd.concat([sorted_data,sorted_data2['area_highschool_ratio']],axis=1)  #concatenation

data.sort_values('area_poverty_ratio',inplace=True)
data.head()
data.columns
# visualize

f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x='area_list',     y='area_poverty_ratio',      data=data,    color='lime',   alpha=0.8)

sns.pointplot(x='area_list',     y='area_highschool_ratio',   data=data,    color='red',    alpha=0.8)



plt.text(40,0.6,   'high school graduate ratio',  color='red',   fontsize = 17,   style = 'italic')

plt.text(40,0.55,  'poverty ratio',               color='lime',   fontsize = 18,   style = 'italic')



plt.xlabel('States',fontsize = 15,color='blue')

plt.ylabel('Values',fontsize = 15,color='blue')

plt.title('High School Graduate  VS  Poverty Rate',fontsize = 20,color='blue')

plt.grid()
#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code



# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables



#sorted_data['area_poverty_ratio'] = sorted_data['area_poverty_ratio']/max( sorted_data['area_poverty_ratio'])

#sorted_data2['area_highschool_ratio'] = sorted_data2['area_highschool_ratio']/max( sorted_data2['area_highschool_ratio'])



# Show the joint distribution using kernel density estimation 

g = sns.jointplot(data.area_poverty_ratio, data.area_highschool_ratio, kind="kde", size=7)

plt.savefig('graph.png')

plt.show()
# We use the same data, so define data = "data name"

g = sns.jointplot("area_poverty_ratio", "area_highschool_ratio", data=data,    size=5,   ratio=3,    color="r")
#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
kill.head()
kill.race.value_counts()
# Race rates according in kill data 

kill.race.dropna(inplace = True)

labels = kill.race.value_counts().index

colors = ['grey','blue','red','yellow','green','brown']

explode = [0,0,0,0,0,0]

sizes = kill.race.value_counts().values



# visual

plt.figure(figsize = (7,7))

plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')

plt.title('Killed People According to Races',color = 'blue',fontsize = 15)
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

# Show the results of a linear regression within each dataset

sns.lmplot(x="area_poverty_ratio", y="area_highschool_ratio", data=data)

plt.show()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

sns.kdeplot(data.area_poverty_ratio, data.area_highschool_ratio, shade=True, cut=3)

plt.show()
# Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data, palette=pal, inner="points")

plt.show()
#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
data.corr()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))

sns.heatmap(data.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)

plt.show()
kill.head()
kill.manner_of_death.unique()
sns.boxplot(x="gender", y="age", hue="manner_of_death", data=kill, palette="PRGn")

plt.show()
sns.swarmplot(x="gender", y="age",hue="manner_of_death", data=kill)

plt.show()
data.head()
sns.pairplot(data)

plt.show()
#race - kill - percent_over_25 - percentage_people_below_poverty - house_income   - DATA
kill.gender.value_counts()
sns.countplot(kill.gender)

#sns.countplot(kill.manner_of_death)

plt.title("gender",color = 'blue',fontsize=15)
armed = kill.armed.value_counts()

plt.figure(figsize=(10,7))

sns.barplot(   x=armed[:7].index,   y=armed[:7].values  )



plt.ylabel('Number of Weapon')

plt.xlabel('Weapon Types')

plt.title('Kill weapon',color = 'blue',fontsize=15)