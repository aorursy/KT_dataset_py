import pandas as pd
import numpy as np

data  = pd.read_csv("../input/athlete_events.csv")

print(data .info())

#Subseting data for all athletes in 1996 Olympic games
#Grouping by Sex and aggregating by minimum to find the youngest male and female athletes 
data.loc[(data ['Year'] == 1996), ['Sex', 'Age']].groupby(['Sex']).min()

# Find a list of unique sports
sports = data['Sport'].unique()

#Print unique sports to find Gymnasts
print(sorted(sports))

#You can use a regex to find them, but I will only use pandas for this exercise.
#Hardcoding a list with all the gymnastic sports
gym_sports = ['Gymnastics', 'Rhythmic Gymnastics']

#Subseting data to find all unique male gymnasts in 2000's Olympic games
gym_male_ath_2000 = np.unique(data.loc[(data['Sport'].isin(gym_sports)) & (data['Sex'] == 'M') & (data['Year'] == 2000), 
                                       ['Name']].values).size

#Subseting data to find all uniqeue male athletes in 2000's Olympic games
all_male_2000 = np.unique(data.loc[(data['Sex'] == 'M') & (data['Year'] == 2000), 
                                   ['Name']].values).size

#Finding the proportion
round(gym_male_ath_2000 * 100 / all_male_2000, 1) 

#Subset data to find all female basketball players in 2000's Olympic games 
bktbll_fem_ath_2000 = pd.DataFrame(data.loc[(data['Sport'] == 'Basketball') & (data['Sex'] == 'F') & (data['Year'] == 2000), 
                                            ['Name', 'Height']].values).drop_duplicates(keep = "first")
#Rename columns
bktbll_fem_ath_2000.columns = ['Name','Height']

#Find mean and std dev of heigth
answer = [round(np.mean(bktbll_fem_ath_2000['Height']),1), round(np.std(bktbll_fem_ath_2000['Height']),1)]
answer
#Subset data to find athletes in 2002's Olympic games 
heaviest_ath_2002 = pd.DataFrame(data.loc[(data['Year'] == 2002) & (data['Weight'].notnull()),
                                          ['Name', 'Sport','Weight']].values).drop_duplicates(keep = "first")
#Rename columns
heaviest_ath_2002.columns = ['Name', 'Sport', 'Weight']

#Finding heaviest by sport
heaviest_ath_2002_by_sport = heaviest_ath_2002.groupby('Sport')['Weight'].max()
heaviest_ath_2002_by_sport[heaviest_ath_2002_by_sport == max(heaviest_ath_2002_by_sport)]
 
#heaviest_ath_2002[data.loc]
#Rename columns
#bktbll_fem_ath_2000.columns = ['Name','Height']

#Subset data to find Pawe Abratkiewicz's games 
p_a_olympics = pd.DataFrame(data.loc[(data['Name'] == 'Pawe Abratkiewicz'), 
                                     ['Name', 'Year']].values).drop_duplicates(keep = "first")
p_a_olympics
#Find all country codes
countries = data['NOC'].unique()

#Print unique sports to find Gymnasts
print(sorted(countries))

#Subset 2000 tennis silver medals for austrlia
pd.DataFrame(data.loc[(data['Year'] == 2000) & (data['NOC'] == 'AUS') & (data['Sport'] == 'Tennis') & (data['Medal'] == 'Silver'), 
         ["Name",'NOC', 'Medal']].values).drop_duplicates(keep = "first")
# Serbia and Switzerland vector
nocs_names = ['SUI', 'SRB']

#Subset 2016 olympic medals for selected countries
medals_ser_switz = data.loc[(data['Year'] == 2016) & (data['NOC'].isin(nocs_names)) & data['Medal'].notnull(),
                            ['NOC', 'Name','Medal']].drop_duplicates(keep = "first")


medals_ser_switz.groupby('NOC').size()
# Set age ranges
age_breaks = [0,15,25,35,45,55]

# Subset ages in 2014 olympic 
ages_2014 = data.loc[(data['Year'] == 2014) & (data['Age'].notnull()), ['Name', 'Age']].drop_duplicates(keep = "first")


# Find counts by age_gruop
age_count = ages_2014.groupby(pd.cut(ages_2014.Age, age_breaks)).count()
print(age_count.sort_values(by = 'Age', ascending = False))

# Fin max and min
print(age_count.loc[age_count['Age'] == age_count['Age'].min(),].index.get_values())
print(age_count.loc[age_count['Age'] == age_count['Age'].max(),].index.get_values())

# Set city vector
cities = ['Sankt Moritz', 'Lake Placid']

# Subset all seasons by city and season 
olympyc_city_season = data.loc[(data['City'].isin(cities)), ['City', 'Season']].drop_duplicates(keep = 'first')
olympyc_city_season
#Selecting unique sports in each year

sports_2016 = data.loc[data['Year'] == 2016, 'Sport'].drop_duplicates(keep = "first").count()
ports_1995 = data.loc[data['Year'] == 1995, 'Sport'].drop_duplicates(keep = "first").count()

abs(sports_2016 - ports_1995)
