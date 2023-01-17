# importing modules which are going to use during EDA



import pandas as pd

import numpy as np

import seaborn as sns

import scipy.stats as stats

from scipy.stats.mstats import winsorize

from statsmodels.stats.weightstats import ttest_ind

import warnings

%matplotlib inline 

import matplotlib as mpl

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
# the first look through the data 



Terror_World = pd.read_csv("../input/gtd/globalterrorismdb_0718dist.csv", encoding = "ISO-8859-1")

Terror_World.head()
# data types and numbers of variables



Terror_World.info()
# we need to see 135 columns to understand what we have in the data



for i in Terror_World.columns:

    print(i)
# we need to see rows and columns together to grasp the data



for i in range (0, 136, 10):

    display (Terror_World.iloc[0:3, i:i+10])
# rename the columns which we need to use during our EDA



Terror_World.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country', 'region_txt':'Region',

                             'city':'City', 'latitude':'Lat.', 'longitude':'Long.', 'attacktype1_txt':'AttackType',

                             'target1':'Target', 'nkill':'Killed', 'nwound':'Wounded', 'gname':'Group',

                             'targtype1_txt':'Target_type', 'weaptype1_txt':'Weapon_type'},inplace=True)
# new Terror data set which we are going to work on



Terror_World=Terror_World[['Year','Month','Day','Country','Region', 'City', 'Lat.','Long.', 'AttackType','Killed','Wounded',

                           'Target', 'Group','Target_type','Weapon_type']]



Terror_World
# new data types and numbers of variables which we are going to work on



Terror_World.info()



# There seems no object feature should be transformed to numeric (float or int) features.
# info of NaN in our data set as percentage



def show_missing (df):

    """This function returns percentage and total number of missing values"""

    percent = df.isnull().sum()*100/df.shape[0]

    total = df.isnull().sum()

    missing = pd.concat([percent, total], axis=1, keys=['percent', 'total'])

    return missing[missing.total>0].sort_values('total', ascending=False)
show_missing(Terror_World)
Terror=Terror_World.copy()
# dropping the NaNs from 'City' and 'Target'

# Because of the low percentage of "City"s and "Target"s NANs, we need to drop them.



Terror=Terror.dropna(subset=['City', 'Target'])
# filling the NaNs of "Killed", "Wounded" with median and creating a variable named "Casualties" as a sum of "Killed" and "Wounded"



Terror['Killed'] = Terror['Killed'].fillna(Terror['Killed'].median())

Terror['Wounded'] = Terror['Wounded'].fillna(Terror['Killed'].median())
Terror.describe()



# we can see that per an attack, the mean of "Killed" is nearly 2. 

# On the other side because the median is zero, at least half of the attacks there are no killed/wounded people. 
# we can even see the number of our analysis above when we look like to the numbers as below.



len(Terror[Terror.Killed==0])
print('There are no killed people in {} of the total {} attacks.'.format (len(Terror[Terror.Killed==0]), len(Terror)))
Terror.Killed.hist(bins=100)
Terror[Terror.Killed<50].Killed.hist(bins=100)

plt.show()
plt.boxplot(Terror.Killed)

plt.show()
from scipy.stats.mstats import winsorize



Terror['winsorize_Killed'] = winsorize(Terror["Killed"], (0, 0.01))



max(Terror.winsorize_Killed)
plt.boxplot(Terror.winsorize_Killed)

plt.title("Boxplot of Killed People during Attacks")

plt.show()
# Number of terrorist activities by year (The data of 1993 is missing at GTD dataset)



plt.subplots(figsize=(14,6))

sns.countplot('Year',data=Terror,palette='Spectral')

plt.xticks(rotation=70)

plt.title('Number Of Terrorist Activities By Year', color='red')

plt.show()
# to see the Country and the regions with highest terrorist attacks and also the max. killed attack



print('Regions with Highest Terrorist Attacks:',Terror['Region'].value_counts().index[0])

print('Country with Highest Terrorist Attacks:',Terror['Country'].value_counts().index[0])

print('Maximum people killed in an attack are:',Terror['Killed'].max(),'that happened in',Terror.loc[Terror['Killed'].idxmax()].Country)
# Number of terrorist attacks by region on chart



plt.subplots(figsize=(14,6))

sns.countplot('Region',data=Terror,palette='Spectral',order=Terror['Region'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities By Region', color='red')

plt.show()
# The first 18 countries by the most killed ones in terrorist attacks



KilledByTerror=Terror.groupby('Country').sum().sort_values('Killed', ascending=False).iloc[:18].Killed
# Number of terrorist attacks and number of killed by country on charts



plt.subplots(figsize=(12,6))

plt.subplot(1,2,1)

sns.countplot('Country',data=Terror,palette='Spectral', 

              order=Terror.Country.value_counts().iloc[:15].index)

plt.xticks(rotation=70)

plt.title('Number Of Terrorist Attacks by Country', fontsize=16, color='red')



plt.subplot(1,2,2)

plt.bar(KilledByTerror.index, KilledByTerror)

plt.xticks(rotation=70)

plt.title('Number Of Killed by Country', fontsize=16, color='red')



plt.show()
# Attacking methods by terrorists



plt.subplots(figsize=(14,6))

sns.countplot('AttackType', data=Terror,palette='Spectral',order=Terror['AttackType'].value_counts().index)

plt.xticks(rotation=70)

plt.title('Attacking Methods by Terrorists', fontsize=16, color='red')

plt.show()
# Favorite targets of attacks



plt.figure(figsize=(15,5))

sns.countplot(Terror['Target_type'],palette='Spectral',order=Terror['Target_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Favorite Targets', fontsize=16, color='darkred')

plt.show()
# Terrorist groups by highest terror attacks



sns.barplot(Terror['Group'].value_counts()[1:15],Terror['Group'].value_counts()[1:15].index,palette=('Spectral'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Terrorist Groups by Highest Terror Attacks', fontsize=16, color='red')

plt.show()
TR_Terror=Terror[Terror['Country']=='Turkey']

TR_Terror
# filling the NaNs of 'Lat.' with 39 and the NaNs of 'Long.' with 35.15 which are transiting in the middle of Turkey.

# because we are going to analyze Turkey's terror attacks below.



TR_Terror['Lat.'] = TR_Terror['Lat.'].fillna(39)

TR_Terror['Long.'] = TR_Terror['Long.'].fillna(35.15)
TR_Terror.info()
show_missing (TR_Terror)
# Number of attacks by year



TR_Terror.Year.plot(kind = 'hist', color = 'b', bins=range(1970, 2018), figsize = (15,7), alpha=0.5, grid=True)

plt.xticks(range(1970, 2018), rotation=90, fontsize=14)

plt.yticks(fontsize=14)

plt.xlabel("Year", fontsize=15)

plt.ylabel("Number of Attacks", fontsize=15)

plt.xticks(rotation=70)

plt.title("Number of Attacks By Year", fontsize=16, color = 'r')

plt.show()
# when we look at the groups claiming responsibility of the attacks



World_Unknown=len(Terror[Terror.Group=='Unknown'])

TR_Unknown=len(TR_Terror[TR_Terror.Group=='Unknown'])

print('Through the world, there is no group claiming the responsibility of {} attacks of the total {} and the percentage is   %{:.2f}'

      .format (World_Unknown, len(Terror), World_Unknown/len(Terror)*100))

print('In Turkey, there is no group claiming the responsibility of {} attacks of the total {} and the percentage is %{:.2f}'

      .format (TR_Unknown, len(TR_Terror), TR_Unknown/len(TR_Terror)*100))
# when we analyze the parts of Turkey which are fronted with terrorist attacks



TR_Terror["part_long"] = np.where((TR_Terror['Long.']>35.15),'East', 'West')

TR_Terror["part_lat"] = np.where((TR_Terror['Lat.']>39),'North', 'South')

TR_Terror["part"] = TR_Terror["part_lat"] + TR_Terror["part_long"]
TR_Terror.head()
# Most targeted part



TR_Terror.part.value_counts().plot.bar(figsize=[12,6], grid=True, alpha=0.8)

sns.countplot(TR_Terror.part)

plt.yticks(fontsize=10)

plt.xticks(fontsize=10)

plt.xlabel("Parts", fontsize=15)

plt.ylabel("Number of Attacks", fontsize=15)

plt.xticks(rotation=70)

plt.title("Most Targeted Part", fontsize=16, color = 'r')

plt.show()

TR_Terror.groupby('part').sum()
# parts with total Killed people



TR_Terror.groupby('part').sum()['Killed'].plot.bar()
# Most targeted cities



TR_Terror.City.value_counts().drop('Unknown').head(10).plot.bar(figsize=[12,6], grid=True, alpha=0.8)

plt.yticks(fontsize=10)

plt.xticks(fontsize=10)

plt.xlabel("Cities", fontsize=15)

plt.ylabel("Number of Attacks", fontsize=15)

plt.xticks(rotation=70)

plt.title("Most Targeted Cities", fontsize=16, color = 'r')

plt.show()
from wordcloud import WordCloud

df = TR_Terror[TR_Terror.City != 'Unknown']

wordcloud = WordCloud(max_font_size=80, max_words=100, background_color="yellow").generate(" ".join(df.City))

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.savefig("graph.png")

plt.show()
# Attacking methods by terrorists in Turkey



plt.subplots(figsize=(14,6))

sns.countplot('AttackType', data=TR_Terror,palette='Spectral',order=TR_Terror['AttackType'].value_counts().index)

plt.xticks(rotation=70)

plt.title('Attacking Methods by Terrorists', fontsize=16, color='red')

plt.show()
# Favorite targets of terrorists in Turkey



plt.subplots(figsize=(14,6))

sns.countplot(Terror['Target_type'],palette='Spectral',order=Terror['Target_type'].value_counts().index)

plt.xticks(rotation=90)

plt.title('Favorite Targets of Terrorists', fontsize=16, color='red')

plt.show()
sns.barplot(TR_Terror['Group'].value_counts()[0:12],TR_Terror['Group'].value_counts()[0:12].index,palette=('Spectral'))

plt.xticks(rotation=90)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.title('Terrorist Groups by Highest Terror Attacks', fontsize=16, color='red')

plt.show()
TR_Terror.corr()
f,ax = plt.subplots(figsize=(13, 13))

sns.heatmap(TR_Terror.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
TR_Terror.groupby('part').mean().Killed
ttest = ttest_ind(TR_Terror[TR_Terror["part"]=='SouthEast'].Killed, 

                    TR_Terror[TR_Terror["part"]=='SouthWest'].Killed)

print(ttest)
parts = TR_Terror["part"].unique()

pd.options.display.float_format = '{:.15f}'.format

for var in ["Killed"]:

    Comparison = pd.DataFrame(columns=['Group_1', 'Group_2', 'Statistics','p_value'])

    print("Comparison for {}".format(var),end='')

    for i in range(0, len(parts)):

        for j in range(i+1, len(parts)):

            ttest = stats.ttest_ind(TR_Terror[TR_Terror["part"]==parts[i]][var], 

                                TR_Terror[TR_Terror["part"]==parts[j]][var])

            Group_1 = parts[i]

            Group_2 = parts[j]

            Statistics = ttest[0]

            p_value = ttest[1]

            

            Comparison = Comparison.append({"Group_1" : Group_1 ,

                                                  "Group_2" : Group_2 ,

                                                "Statistics" : Statistics,

                                                  "p_value" : p_value}, ignore_index=True)

    display(Comparison)