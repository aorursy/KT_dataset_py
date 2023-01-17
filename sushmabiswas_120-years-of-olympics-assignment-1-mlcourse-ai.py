import pandas as pd

import numpy as np



# enable multiple outputs

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = 'all'
data = pd.read_csv('../input/120_years_of_olympics_data.csv')

data.head()
# Q1. How old were the youngest male and female participants of the 1992 Olympics?



lowest_age_female, lowest_age_male = data[data.Year==1992].groupby(by='Sex').Age.min()

print('In 1992 Olympics, age of youngest male was {0} and age of youngest female was {1}'.format(lowest_age_male, lowest_age_female))
# Q2: What was the percentage of male basketball players among all the male participants of the 2012 Olympics? Round the answer to the first decimal

total_male_players = data[(data.Year==2012) & (data.Sex=='M')].Name.nunique()

male_basketball_players = data[(data.Year==2012) & (data.Sex=='M') & (data.Sport=='Basketball')].Name.nunique()



print('Percentage of male basketball players among all male participants in 2012 was: {:.1f}%'.format(male_basketball_players*100/total_male_players))



# Reminder: Well, one person can play multiple sports and due to that, the total no. of men is NOT 7105. Instead, it is 5858!
# Q3: What are the mean and standard deviation of height for female tennis players who participated in the 2000 Olympics? Round the answer to the first decimal

mean_height, std_height = data[(data.Sex=='F') & (data.Year==2000) & (data.Sport == 'Tennis')].Height.describe()[1:3]

print('Mean height: {:.1f}\nMean standard deviation: {:.1f}'.format(mean_height,std_height))
# Q4: Find a sportsman who participated in the 2006 Olympics, with the highest weight among other participants of the same Olympics. 

# What sport did he or she do?



data[data.Year==2006].groupby(by='Sport').Weight.max()
# Q5: How many times did John Aalberg participate in the Olympics held in different years?

len(data[data.Name == 'John Aalberg'].Year.unique())
# Q6: How many gold medals in tennis did sportspeople from the Switzerland team win at the 2008 Olympics? 

# Count every medal from every sportsperson.



data[(data.Team == 'Switzerland') & (data.Year == 2008) & (data.Medal == 'Gold') & (data.Sport == 'Tennis')].Medal.count()
# Q7: Is it true that Spain won fewer medals than Italy at the 2016 Olympics? Do not consider NaN values in Medal column.

(data[(data.Team == 'Spain') & (data.Year == 2016) & ~(data.Medal.isna())].Medal.count()) < (data[(data.Team == 'Italy') & (data.Year == 2016) & ~(data.Medal.isna())].Medal.count())
# Q8: What age category did the fewest and the most participants of the 2008 Olympics belong to? 

data_2008 = data[data.Year == 2008].drop_duplicates(subset='Name')



# # we are binning the age by using the pd.cut method that creates 5 buckets using the range function. Really nifty!

# data_2008.groupby(pd.cut(data_2008["Age"], range(5, 56, 10))).Age.count()



# a different way - also, changing the range to fit the buckets given in the question

def age_category(age):

    '''Maps age to four categories'''



    if 15 <= age < 25:

        return '[15, 25)'

    elif 25 <= age < 35:

        return '[25, 35)'

    elif 35 <= age < 45:

        return '[35, 45)'

    return '[45, 55]'





# map() applies age_category() function to every value in data.Age

data['age_category'] = data.Age.map(age_category)

(data[data.Year == 2008]

.drop_duplicates(subset='Name')  

.groupby('age_category')

.size())
# Q9: part 1: Is it true that there were Summer Olympics held in Atlanta?

data[(data.City == 'Atlanta') & (data.Season == 'Summer')].ID.any()



# in fact, Atlanta ONLY had summer olympics!

data[data.City == 'Atlanta'].Season.unique()



# part 2: Is it true that there were Winter Olympics held in Squaw Valley?

data[(data.City == 'Squaw Valley') & (data.Season == 'Winter')].ID.any()

data[data.City == 'Squaw Valley'].Season.unique()
# Q10: What is the absolute difference between the number of unique sports at the 1986 Olympics and 2002 Olympics?

sport_count_1986 = data[data.Year == 1986].Sport.nunique() # no event in 1986!

sport_count_2002 = data[data.Year == 2002].Sport.nunique()



print('The absolute difference between the number of unique sports at the 1986 Olympics and 2002 Olympics: ', abs(sport_count_1986 - sport_count_2002))