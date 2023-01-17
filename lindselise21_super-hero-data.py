# Import all the required packages

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import csv as csv

import re





#import the data sets

hero_information = pd.read_csv('../input/heroes_information.csv', header = 0, na_values = ['-'] )

hero_powers = pd.read_csv('../input/super_hero_powers.csv', header = 0, na_values = ['_'])

#I want to join the data sets together, but since the primary keys don't match up all I have to join the tables on are the names.

#Unfortunately there are a lot of duplicate names. Lets get rid of those. 

hero_information=hero_information.drop_duplicates(subset='name')

hero_powers = hero_powers.drop_duplicates(subset='hero_names')
#Combining the tables

super_info = hero_information.set_index('name').join(hero_powers.set_index('hero_names'))

#Gets rid of any null values in the Alignment column, which is where our heros are listed as good or bad

super_info=super_info.dropna(subset = ['Alignment'])

#Drops columns we won't need

supers=super_info.drop(columns = ['Unnamed: 0', 'Gender', 'Eye color', 'Race', 'Hair color', 'Height', 'Publisher', 'Skin color', 'Weight'], axis = 1)

#Changes true/false values to 1/0 so we can work with the super power data a little easier

supers=supers*1

#Drops any rows that don't have superpowers or alignment listed

supers=supers.dropna(subset=['Agility', 'Alignment'])

#Creates a new column with a sum of superpowers for each super

supers.loc[:, 'Total_Powers'] = supers.iloc[:, 1:].sum(axis=1)
#Ok, finally ready to make some comparisons. First lets look at the means of total superpowers for bothheros and villains

villain = supers[(supers['Alignment']=='bad')]

hero = supers[(supers['Alignment']== 'good')]

neutral = supers[(supers['Alignment']== 'neutral')]

vil_power_mean= villain['Total_Powers'].mean()

her_power_mean=hero['Total_Powers'].mean()

neu_power_mean=neutral['Total_Powers'].mean()

print("Mean of Villain Powers: " + str(vil_power_mean))

print("Mean of Hero Powers: "+ str(her_power_mean))

print("Mean of Neutral Super Powers: " + str(neu_power_mean))
# Interesting, the mean puts villians with an average of one more superpower than the heros, but the neutral supers really pack a punch.

#Lets look at it on a boxplot though to get a better picture.



sns.boxplot(x='Alignment', y='Total_Powers', data=supers)
#There are a lot of outlier heros who have a ton of powers, which would pull the mean up. 

#Based on the interquartile range it looks like the majority of heros really do have a fewer number of superpowers than our villians.

#Ok, apparently super heros powers aren't what gives them such an advantage, maybe it's height or weight?
#Let's take a look:

super_info = super_info[(super_info['Height'] >= 0) | (super_info['Height'].isnull())]

sns.boxplot(x='Alignment', y='Height', data=super_info)
#Hmmm. not too much height variation. What about weight?

super_info = super_info[(super_info['Weight'] >= 0) | (super_info['Weight'].isnull())]

sns.boxplot(x='Alignment', y='Weight', data=super_info)







#Good guys seem to be a little lighter, what if we compare their BMI?

super_info['BMI'] = (super_info['Weight']/(super_info['Height']/100)**2)

sns.boxplot(x='BMI', y='Alignment', data=super_info, showfliers=False)



#Look at that! Assuming that a healthy BMI range of 18.5-24.9 actually applies to super-people (don't think too much about it),

#there seems to be a correlation between being a good guy and being a healthy weight. 

#Maybe it's time to hit the gym!