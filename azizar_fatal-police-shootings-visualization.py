import numpy             as np

import pandas            as pd

import seaborn           as sns

import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')

 

import warnings

warnings.filterwarnings('ignore')
med_househ_inc     = pd.read_csv('../input/fatal-police-shootings-in-the-us/MedianHouseholdIncome2015.csv', encoding='ISO-8859-1')

perc_completed_hs  = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentOver25CompletedHighSchool.csv', encoding='ISO-8859-1') 

share_race         = pd.read_csv('../input/fatal-police-shootings-in-the-us/ShareRaceByCity.csv')    

police_killings    = pd.read_csv('../input/fatal-police-shootings-in-the-us/PoliceKillingsUS.csv', encoding='ISO-8859-1')

perc_poverty       = pd.read_csv('../input/fatal-police-shootings-in-the-us/PercentagePeopleBelowPovertyLevel.csv', encoding='ISO-8859-1')
perc_poverty.head()
perc_poverty.info()
perc_poverty.poverty_rate.replace('-', np.nan, inplace = True)

perc_poverty.poverty_rate = perc_poverty.poverty_rate.astype(float)

grouped_data = perc_poverty.groupby('Geographic Area', as_index = False)['poverty_rate'].mean().sort_values(by = 'poverty_rate', ascending = False)

 

plt.figure(figsize = (10,9))

sns.barplot(grouped_data['Geographic Area'],grouped_data['poverty_rate'],palette = 'rainbow')

plt.xticks(rotation=45)

plt.title('Poverty Rate Given States')

plt.ylabel('Poverty Ratio')

plt.xlabel('States')

plt.show()
police_killings.head()
names = ' '.join(police_killings.name).split()

common_names = pd.Series(names).value_counts()[1:16]

 

plt.figure(figsize=(11,10))

sns.barplot( common_names.index, common_names, palette = sns.color_palette('coolwarm',15))

plt.xticks(rotation=75)

plt.title('Most common 15 names or surnames')

plt.show()
perc_completed_hs.head()
perc_completed_hs.percent_completed_hs.replace('-', np.nan, inplace = True)

perc_completed_hs.percent_completed_hs = perc_completed_hs.percent_completed_hs.astype(float)

grouped_data2 = perc_completed_hs.groupby('Geographic Area', as_index = False)['percent_completed_hs'].mean()

 

plt.figure(figsize=(10,9))

sns.barplot(grouped_data2.iloc[:,0], grouped_data2.iloc[:,1], palette = sns.color_palette('cool', 51))

plt.ylabel('Graduate rate')

plt.title('Percentage of population that has graduated high school')

plt.xlabel('States')

plt.xticks(rotation=70)

plt.show()
share_race.head()
share_race.info()
share_race.replace('(X)', np.nan, inplace=True)

share_race.replace('-', np.nan, inplace=True)

 

for i in share_race.columns[2:]:

    share_race[i] = share_race[i].astype(float)
share_race.dtypes
grouped_data3 = share_race.groupby('Geographic area', as_index = False)[share_race.columns[2:]].mean()
f, ax = plt.subplots(figsize = (9,15))

 

sns.barplot(grouped_data3.iloc[:,1],grouped_data3.iloc[:,0], color = 'g',alpha = 0.7, label = 'White')

sns.barplot(grouped_data3.iloc[:,2],grouped_data3.iloc[:,0], color = 'b', label = 'African American')

sns.barplot(grouped_data3.iloc[:,3],grouped_data3.iloc[:,0], color = 'c', label = 'Native American')

sns.barplot(grouped_data3.iloc[:,4],grouped_data3.iloc[:,0], color = 'y', label = 'Asian')

sns.barplot(grouped_data3.iloc[:,5],grouped_data3.iloc[:,0], color = 'r', label = 'Hispanic')

 

ax.legend(frameon = True)

ax.set(xlabel = 'Percentage of races', ylabel = 'States', title = 'Percentage of Sate\'s population according to races')

plt.show()
grouped_data.poverty_rate /= max(grouped_data.poverty_rate)

grouped_data2.percent_completed_hs /= max(grouped_data2.percent_completed_hs)

data = pd.concat([grouped_data, grouped_data2.percent_completed_hs], axis = 1).sort_values(by = 'poverty_rate')

data.head()
plt.figure(figsize = (11,8))

sns.pointplot('Geographic Area', 'poverty_rate', data = data, color = 'lime')

sns.pointplot('Geographic Area', 'percent_completed_hs', data = data, color = 'red' )

plt.text(35, 0.6, 'high school graduate ratio', color = 'r', fontsize = 16, style = 'italic')

plt.text(35, 0.55, 'poverty ratio ', color = 'lime', fontsize = 16, style = 'italic')

plt.xlabel('States', fontsize = 16, color = 'b')

plt.ylabel('Values', fontsize = 16, color = 'b')

plt.title('High school graduate vs poverty rate', fontsize = 20, color = 'blue')

plt.xticks(rotation = 90)

plt.grid(True)

plt.show()
data.head()
# Visualization of high school graduation rate vs Poverty rate of each state with different style of seaborn code

#joint kernel density

# pearsonr= if it is 1, there is positive correlation and if it is, -1 there is negative correlation.

# If it is zero, there is no correlation between variables

# Show the joint distribution using kernel density estimation 

 

g = sns.jointplot(data.poverty_rate, data.percent_completed_hs, kind = 'kde', size = 8)

plt.show() 
# you can change parameters of joint plot

# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }

# Different usage of parameters but same plot with previous one

sns.jointplot('poverty_rate', 'percent_completed_hs',data = data, size = 8, ratio = 5, color = 'magenta')

plt.show()
police_killings.head()
police_killings.race.value_counts()
 #Race rates according in kill data

 

labels  = police_killings.race.dropna().unique()

co

explode = [0,0,0,0,0,0]  

colors = ['#acf8ff','#ffb6ff','#bdb2ff','#ffadad','#ffd6a5','#caffbf'] 

 

sizes   = police_killings.race.value_counts().values

  

# Visualization 

plt.figure(figsize = (10,10))

plt.pie(sizes, explode = explode, colors = colors,  labels = labels, autopct = '%1.1f%%')

plt.title('Killed people according to races', color = 'blue', fontsize = 18) 

plt.show()
data.head()
# Show the results of a linear regression within each dataset



sns.lmplot('poverty_rate', 'percent_completed_hs', size = 8, data = data, scatter_kws ={'color': '#00bbf9'}, line_kws={'color': '#00bbf9'})

plt.show()
# kernel density estimation

 

plt.figure(figsize = (10,10))

sns.kdeplot(data.poverty_rate, data.percent_completed_hs, color = '#7400b8', shade = True, cut = 2)

plt.show()
 # Show each distribution with both violins and points

# Use cubehelix to get a custom sequential palette

plt.figure(figsize=(10,8))

colors = ['#0366c8', '#780aca'] 

sns.violinplot(data = data, inner = 'points', size = 10, palette = sns.color_palette(colors) )  

plt.show()
data.corr()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot = True, cmap = 'Greens',linecolor = 'red', fmt='.2f')

plt.show() 
police_killings.head()
police_killings.manner_of_death.unique()
plt.figure(figsize=(10,8))

sns.boxplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings, palette = sns.color_palette(colors))

plt.show() 
# swarm plot

plt.figure(figsize=(10,8))

sns.swarmplot(x = 'gender', y = 'age', hue = 'manner_of_death', data = police_killings)

plt.show()
sns.pairplot(data, palette = 'Greens',  height = 4)

plt.show()
police_killings.gender.value_counts()
# kill properties

# manner of death

plt.figure(figsize=(10,8))

sns.countplot(police_killings.gender, palette = sns.color_palette(colors))  

plt.show()
armed = police_killings.armed.value_counts()

 

plt.figure(figsize = (10,8))

sns.barplot(x = armed[:7].index, y = armed[:7].values, palette = sns.color_palette('Greens_r', 7 )) 

plt.xlabel('Number of weapon')

plt.ylabel('Weapon types')

plt.title('Kill weapon')

plt.show()
# Age of killed people

plt.figure(figsize=(10,8))

above25 = ['above25' if i > 25 else 'below25' for i in police_killings.age]

df = pd.DataFrame({'age':above25})

sns.countplot(x=df.age, palette = sns.color_palette('rainbow',  7)) 

plt.ylabel('Number of Killed People')

plt.title('Age of killed people',color = 'blue',fontsize=15)

plt.show()
plt.figure(figsize=(10,8))

sns.countplot(data=police_killings, x='race', palette = sns.color_palette('rainbow_r',7)) 

plt.title('Race of killed people',color = 'blue',fontsize=15)

plt.show()
 # Most dangerous cities

city = police_killings.city.value_counts()

plt.figure(figsize=(10,9))

sns.barplot(x=city[:12].index,y=city[:12].values)

plt.xticks(rotation=60)

plt.title('Most dangerous cities',color = 'blue',fontsize=15)

plt.show()