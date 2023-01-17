# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # We use this library for professional plots

import matplotlib.pyplot as plt # We use this library for professional plots



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
median_household_income = pd.read_csv("../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv", encoding = "windows-1252")

percent_over_25_completed_highschool = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv", encoding = "windows-1252")

percentage_people_below_povert_level = pd.read_csv("../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv", encoding = "windows-1252")

kill = pd.read_csv("../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv", encoding = "windows-1252") 

share_race_by_city = pd.read_csv("../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv", encoding = "windows-1252")

percentage_people_below_povert_level.head() #Gives us the summary info
percentage_people_below_povert_level.columns #Gives us the column names of data
percentage_people_below_povert_level.info() #Gives us the some informations about data
percentage_people_below_povert_level.poverty_rate.value_counts() #Gives us the poverty_rate column's values. For example there are 1464 zero values.

# But there are 201 - values. We don't know what - means. Normally we can analyze the data and find out what - means. But it is not important for now.

# I decide to change the value from - to 0.
percentage_people_below_povert_level.poverty_rate.replace(["-"], 0.0, inplace = True) # We use inplace command for saving the changings in our variable.
percentage_people_below_povert_level.poverty_rate.value_counts() # As we can see, the - values dropped and 0 values added instead.
# Povert rate's type is object. But we want to change it to float type.

percentage_people_below_povert_level.poverty_rate = percentage_people_below_povert_level.poverty_rate.astype(float)
percentage_people_below_povert_level.info() # As we can see, its type is float now.
#Because we want to find the poverty rate of each state, we must find the unique states in the "Geographic Area" column.

percentage_people_below_povert_level["Geographic Area"].unique()
# To take action on these states, we must save the result as a list in a variable.

area_list = list(percentage_people_below_povert_level["Geographic Area"].unique())
# To be able to sort bars in the barplot, we should run first these codes.

area_poverty_ratio = []

for i in area_list:

    x = percentage_people_below_povert_level[percentage_people_below_povert_level["Geographic Area"] == i]

    area_poverty_rate = sum(x.poverty_rate)/len(x)

    area_poverty_ratio.append(area_poverty_rate)

data = pd.DataFrame({"area_list":area_list, "area_poverty_ratio":area_poverty_ratio})

new_index = (data["area_poverty_ratio"].sort_values(ascending = False)).index.values

sorted_data = data.reindex(new_index)
# Visualizating data

plt.figure(figsize=(15,10)) #Matplotlib olması önemli değil. Aslında bir alt satırda Seaborn kullanıyoruz. Bu satır yeni bir figür açmak için kullanılan bir kod.

# Bu koddaki 15 yatay uzunluğu 10 ise dikey uzunluğu göstermektedir.

sns.barplot(x=sorted_data["area_list"], y=sorted_data["area_poverty_ratio"])

plt.xticks(rotation = 90) # This code is used for rotating the names of states under the graph.

plt.xlabel("States")

plt.ylabel("Poverty Rate")

plt.title("Poverty Rate Given States")

plt.show()
kill.head()
kill.name.value_counts()
# The most important part of the next plot analysis is below. These are preliminaries.

# The most common 15 name or surname of killed people

separate = kill.name[kill.name != 'TK TK'].str.split() # There are 49 TK TK named people. But it is not possible 

# that they are real names. So we don't want to include these names. str.split command separates names and last names.

a,b = zip(*separate)    # This command zip the names and last names again.                 

name_list = a+b # This command add the names and last names as tuple.                      

name_count = Counter(name_list)         

most_common_names = name_count.most_common(15)  

x,y = zip(*most_common_names)

x,y = list(x),list(y)

# We form the plot with the codes below.

plt.figure(figsize=(15,10))

ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))

plt.xlabel('Name or Surname of killed people')

plt.ylabel('Frequency')

plt.title('Most common 15 Name or Surname of killed people') 