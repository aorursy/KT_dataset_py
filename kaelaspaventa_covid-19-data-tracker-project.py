project_name = "COVID Tracker Project" 
#Covid-19 Data Tracker Project: Death Rates Among 25-34 Year Olds With Underlying Diseases by Mikaela Spaventa



#This project gauges the impact of COVID-19 death rates among different racial and ethnic groups in the United States.



#It is known that minorities are more likely to be affected by COVID-19 relative to other racial and ethnic classes.



#Therefore, this project will highlight these effects.



#I thought the 25-34 year old age group would be interesting to follow since most Americans in this age group are working,

#which can shed light on how different working environments may impact some groups more than others.



#Also, it would be interesting to see how COVID is impacting the younger population since there seems to be a misconception 

#that only older people are succumbing to the virus.



#Additionally, I chose to evaluate the Covid deaths experienced by those with underlying health issues.  



#This may show disparities in the U.S. heath system as well.



#I'm using a dataset that from the U.S. government that records diseases and illnesses Americans die from

#which is cited on the bottom of the page.



#I plan to filter and clean this data to just get the number of COVID death rates for males and females from 25-34 year olds

#from the United States since the beginning of the pandemic.



#Once I acquire and clean the data, I plan to evaluate the data from a series of graphics and visualizations.



#For this project, I'm using Python3 to acquire, clean, and evaluate the data, 

#along with the use of numpy, pandas, matplotlib, and seaborn libraries.
pwd
#Data Collection- Here I'm importing a CSV file from a government website

import pandas as pd

import numpy as np

from urllib.request import urlretrieve

#urlretrieve('https://data.cdc.gov/api/views/65mz-jvh5/rows.csv?accessType=DOWNLOAD')

df = pd.read_csv('../input/deaths-in-us/Monthly_provisional_counts_of_deaths_by_age_group__sex__and_race_ethnicity_for_select_causes_of_death (3).csv')

df
df.shape

#Using the numpy .shape function, I have found that there are 2400 columns and

#37 columns in this initial dataset before I start the cleaning process
df.info()

#Also, just some more basic info about the dataset
#Data Cleaning

covid_deaths = df[['Date Of Death Year', 'Date Of Death Month', 'Sex','Race/Ethnicity', 

                   'AgeGroup',  

                   'COVID-19 (U071, Underlying Cause of Death)']]

covid_deaths
covid_deaths.rename(columns = {'Date Of Death Year':'DeathYear'}, inplace = True)

covid_deaths.rename(columns = {'Date Of Death Month':'DeathMonth'}, inplace = True)

covid_deaths.rename(columns = {'Race/Ethnicity':'RaceEthnicity'}, inplace = True)

covid_deaths.rename(columns = {'COVID-19 (U071, Underlying Cause of Death)':'COVIDUnderlyingCause'}, inplace = True)

covid_deaths
covid_deaths = covid_deaths[covid_deaths.COVIDUnderlyingCause != 0.0]

covid_deaths = covid_deaths[covid_deaths.RaceEthnicity != 'Other']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '65-74 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '45-54 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '85 years and over']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '75-84 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '0-4 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '5-14 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '55-64 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '15-24 years']

covid_deaths = covid_deaths[covid_deaths.AgeGroup != '35-44 years']

covid_deaths 

##I'm focusing on the working population between the ages 25-34 so I'm filtering out all of the other age groups
covid_deaths.dropna(how='any')
covid_deaths = covid_deaths[['DeathMonth', 'Sex', 'RaceEthnicity', 

                   'AgeGroup',  

                   'COVIDUnderlyingCause']]

covid_deaths

newcovid=covid_deaths.dropna(how='any')

newcovid

## Removed DeathYear since we know all COVID deaths took place during 2020 in the United States
newcovid.RaceEthnicity.unique()
newcovid.RaceEthnicity.replace(to_replace='Non-Hispanic Asian', 

                                   value='Asian', inplace=True)

newcovid.RaceEthnicity.replace(to_replace='Non-Hispanic Black', 

                                   value='Black', inplace=True)

newcovid.RaceEthnicity.replace(to_replace='Non-Hispanic White', 

                                   value='White', inplace=True)

newcovid.RaceEthnicity.replace(to_replace='Non-Hispanic American Indian or Alaska Native', 

                                   value='American Indian', inplace=True)

newcovid.RaceEthnicity.unique()

##Taking Non-Hispanic out infront of every other race name just to make each less clustered and easier to read
newcovid
newcovid.COVIDUnderlyingCause.sum()
femalecovid = newcovid[newcovid.Sex != 'Male (M)']

femalecovid
malecovid = newcovid[newcovid.Sex != 'Female (F)']

malecovid
#In this next section, I'm going to analyze the certain sections of the data

#and make comparisons

#First, I'm going to compare the rates of deaths among American men and women in 

#this age group

femalecovid.COVIDUnderlyingCause.describe()
#Finding the mean for deaths all female groups (in another way)

meanfemalecovid = femalecovid['COVIDUnderlyingCause'].mean()

print('Mean:', meanfemalecovid)
#Finding the sum for covid deaths in female groups (in another way)

sumfemalecovid = femalecovid['COVIDUnderlyingCause'].sum()

print('Sum:', sumfemalecovid)
#From the describe method, we know the range for covid deaths in 

#female groups is between 48 and 11
#All other interesting statistics can be seen in from the results from

#the describe method, such as the standard deviation, quartile ranges, and 

#variance, which can simply be calculated by taking the square root of the standard

#deviation (11.59)^1/2=3.4
malecovid.COVIDUnderlyingCause.describe()

#As we can see here, males between 25-34 that died from COVID due to

#underlying causes are dying at higher rates compared to females



#Why might this be?

#It's possible that men of this age group having higher exposure to COVID-19 due to the 

#type of work they take on

#It's also possible that men of this age group are more willing to take risks than women,

#as they risk to be in environments that may make them more prone to COVID

#Let's move on to see the visualizations of this data
#Finding the mean for deaths all male groups (in another way)

meanmalecovid = malecovid['COVIDUnderlyingCause'].mean()

print('Mean:', meanmalecovid)
#Finding the sum for covid deaths in male groups (in another way)

summalecovid = malecovid['COVIDUnderlyingCause'].sum()

print('Sum:', summalecovid)
#From the describe method, we know the range for covid deaths in 

#male groups is between 10 and 114 in a given month
#All other interesting statistics can be seen in from the results from

#the describe method, such as the standard deviation, quartile ranges, and 

#variance, which can simply be calculated by taking the square root of the standard

#deviation (32.2)^1/2=5.67
import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

%matplotlib inline
plt.title('COVID Deaths Among American Women Between the Ages of 25-34 with Underlying Medical Issues')

sns.barplot('DeathMonth', 'COVIDUnderlyingCause', hue='RaceEthnicity',

            data=femalecovid);

#Now we're going to look at the differences in COVID deaths between men and women.

#We're also going to look at the differences of deaths in experienced by racial and ethnic groups.



#Here we can see that Black and Hispanic females have faced the highest rates of COVID deaths due to 

#other underlying medical conditions.

#This is significant because our data shows that more COVID deaths have been experienced in both

#Black and Hispanic females between the ages 25-34, despite both groups making up a racial minority. 

#Black and Hispanic Americans only make up approximately 13 and 15 percent of the population repectively.

#White Americans make up the major of the population (approximately 63 percent) but white female deaths in this group

#barely make up half of the COVID deaths experienced by Black and Hispanic females in almost every month.
#I also want to look at other types of graphs with this data

#Here is a scatterplot

plt.title('COVID Deaths Among American Women Between the Ages of 25-34 with Underlying Medical Issues')

sns.scatterplot(femalecovid.COVIDUnderlyingCause, 

                femalecovid.DeathMonth, hue=femalecovid.RaceEthnicity, s=100)
sns.pairplot(femalecovid, hue='RaceEthnicity')
plt.title('COVID Deaths Among American Men Between the Ages of 25-34 with Underlying Medical Issues')

sns.barplot('DeathMonth', 'COVIDUnderlyingCause', hue='RaceEthnicity',

            data=malecovid);

#Here with the male Covid data, we see very similar trends with Hispanic and Black Americans with underlying health issues

#making up a exceedingly larger proportion of COVID-19 deaths compared to that of other groups.

#It's worth to note that male covid-deaths in general are higher than female deaths to a great extent

#In the month of April (denoted by 4), the number of death are more than doubled in every racial group compared to the

#the number of female Covid deaths.
#Also going to use different graphs

plt.title('COVID Deaths Among American Men Between the Ages of 25-34 with Underlying Medical Issues')

sns.scatterplot(malecovid.COVIDUnderlyingCause, 

                malecovid.DeathMonth, hue=malecovid.RaceEthnicity, s=100)
sns.pairplot(femalecovid, hue='RaceEthnicity')
##At this point, I'm going to create different datasets to make it easier to compare to eachother

blackdeaths = femalecovid[femalecovid.RaceEthnicity != 'Hispanic']

blackdeaths= blackdeaths[femalecovid.RaceEthnicity != 'White']

blackdeaths = blackdeaths[femalecovid.RaceEthnicity != 'Asian']

blackdeaths = blackdeaths[femalecovid.RaceEthnicity != 'American Indian']

blackdeaths
blackdeaths.COVIDUnderlyingCause.describe()
blackdeaths2 = malecovid[malecovid.RaceEthnicity != 'Hispanic']

blackdeaths2= blackdeaths2[malecovid.RaceEthnicity != 'White']

blackdeaths2 = blackdeaths2[malecovid.RaceEthnicity != 'Asian']

blackdeaths2 = blackdeaths2[malecovid.RaceEthnicity != 'American Indian']

blackdeaths2
blackdeaths2.COVIDUnderlyingCause.describe()
hispanicdeaths = femalecovid[femalecovid.RaceEthnicity != 'Black']

hispanicdeaths= hispanicdeaths[femalecovid.RaceEthnicity != 'White']

hispanicdeaths = hispanicdeaths[femalecovid.RaceEthnicity != 'Asian']

hispanicdeaths = hispanicdeaths[femalecovid.RaceEthnicity != 'American Indian']

hispanicdeaths
hispanicdeaths.COVIDUnderlyingCause.describe()
hispanicdeaths2 = malecovid[malecovid.RaceEthnicity != 'Black']

hispanicdeaths2= hispanicdeaths2[malecovid.RaceEthnicity != 'White']

hispanicdeaths2 = hispanicdeaths2[malecovid.RaceEthnicity != 'Asian']

hispanicdeaths2 = hispanicdeaths2[malecovid.RaceEthnicity != 'American Indian']

hispanicdeaths2
hispanicdeaths2.COVIDUnderlyingCause.describe()
whitedeaths = femalecovid[femalecovid.RaceEthnicity != 'Black']

whitedeaths= whitedeaths[femalecovid.RaceEthnicity != 'Hispanic']

whitedeaths = whitedeaths[femalecovid.RaceEthnicity != 'Asian']

whitedeaths = whitedeaths[femalecovid.RaceEthnicity != 'American Indian']

whitedeaths
whitedeaths.COVIDUnderlyingCause.describe()
whitedeaths2 = malecovid[malecovid.RaceEthnicity != 'Black']

whitedeaths2= whitedeaths2[malecovid.RaceEthnicity != 'Hispanic']

whitedeaths2 = whitedeaths2[malecovid.RaceEthnicity != 'Asian']

whitedeaths2 = whitedeaths2[malecovid.RaceEthnicity != 'American Indian']

whitedeaths2
whitedeaths2.COVIDUnderlyingCause.describe()
asiandeaths = femalecovid[femalecovid.RaceEthnicity != 'Black']

asiandeaths= asiandeaths[femalecovid.RaceEthnicity != 'Hispanic']

asiandeaths = asiandeaths[femalecovid.RaceEthnicity != 'White']

asiandeaths = asiandeaths[femalecovid.RaceEthnicity != 'American Indian']

asiandeaths

#There aren't any Asian female covid deaths from this data set recorded
asiandeaths.COVIDUnderlyingCause.describe()
asiandeaths2 = malecovid[malecovid.RaceEthnicity != 'Black']

asiandeaths2= asiandeaths2[malecovid.RaceEthnicity != 'Hispanic']

asiandeaths2 = asiandeaths2[malecovid.RaceEthnicity != 'White']

asiandeaths2 = asiandeaths2[malecovid.RaceEthnicity != 'American Indian']

asiandeaths2
asiandeaths2.COVIDUnderlyingCause.describe()
aideaths = femalecovid[femalecovid.RaceEthnicity != 'Black']

aideaths= aideaths[femalecovid.RaceEthnicity != 'Hispanic']

aideaths = aideaths[femalecovid.RaceEthnicity != 'White']

aideaths = aideaths[femalecovid.RaceEthnicity != 'Asian']

aideaths
aideaths.COVIDUnderlyingCause.describe()
aideaths2 = malecovid[malecovid.RaceEthnicity != 'Black']

aideaths2= aideaths2[malecovid.RaceEthnicity != 'Hispanic']

aideaths2 = aideaths2[malecovid.RaceEthnicity != 'White']

aideaths2 = aideaths2[malecovid.RaceEthnicity != 'Asian']

aideaths2
aideaths2.COVIDUnderlyingCause.describe()
#Questions about the data

#1. Which group has had the highest COVID death average since March?

#2. How many Americans between the ages 25-34 with underlying health issues

#   have passed away from COVID since March?

#3. What is the highest number of deaths experienced in one month by one

#   group and what was the group that experienced this?

#4. How does the highest number of deaths experienced by females in one month 

#   compare to the highest number of deaths experienced by males in one month?

#5. Have death rates inclined and declined equally across all groups during 

#   the pandemic?
#Question 1

#I can find the averages for each group using the .mean() function, which finds the mean of a specified column in a dataframe

#I left Asian women out the calculation because they experienced 0 deaths under this category

#This function is from Pandas.

print('Mean deaths for Black women: {}'.format(blackdeaths['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Black men: {}'.format(blackdeaths2['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Hispanic women: {}'.format(hispanicdeaths['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Hispanic men: {}'.format(hispanicdeaths2['COVIDUnderlyingCause'].mean()))

print('Mean deaths for White women: {}'.format(whitedeaths['COVIDUnderlyingCause'].mean()))

print('Mean deaths for White men: {}'.format(whitedeaths2['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Asian men: {}'.format(asiandeaths2['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Native american women: {}'.format(aideaths['COVIDUnderlyingCause'].mean()))

print('Mean deaths for Native american men: {}'.format(aideaths2['COVIDUnderlyingCause'].mean()))



##As a result, we can see Hispanic men face the highest COVID death average.

##An average of 65 Hispanic men in the US between 25-34 years old with underlying heath issues have died every month since the 

## beginning of the pandemic.
#Question 2

#I can find the total number of Americans in this dataframe that have passed away by using the .sum() function.

#This function is also from Pandas

print('Sum: {}'.format(newcovid.COVIDUnderlyingCause.sum()))
#Question 3

#In order to figure out what the highest number of deaths was in one month and which group, we can use the .max() function.

#This function is from Pandas.

print('Highest deaths in one month for Black women: {}'.format(blackdeaths['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Black men: {}'.format(blackdeaths2['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Hispanic women: {}'.format(hispanicdeaths['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Hispanic men: {}'.format(hispanicdeaths2['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for White women: {}'.format(whitedeaths['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for White men: {}'.format(whitedeaths2['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Asian men: {}'.format(asiandeaths2['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Native American women: {}'.format(aideaths['COVIDUnderlyingCause'].max()))

print('Highest deaths in one month for Native American men: {}'.format(aideaths2['COVIDUnderlyingCause'].max()))



#As we can see here, the highest number of deaths experienced in one month is 114, which is from Hispanic men.

#We can also see that Black men were a close second with 95 deaths.
#Question 4

#This next question can be answered using the output above.

#We can see here that Black women experienced the highest number of deaths at 48 deaths, followed by Hispanic women at 42.

# a close second. Black women experienced this high number of deaths in April 2020 and Hispanic women during August of 2020.

#This can be seen on the graph displayed for question 5.

#Comparatively Hispanic men faced the highest number of deaths at 114 deaths in April 2020, followed by Black men at 95 during 

#April 2020.

#What's interesting about this is that the highest number of deaths experienced in one month for men of this age group

# is double that of women in this age group.
#Question 5

#We can look at how the deaths have inclined and declined over time by looking at the barplots

#I can create the bar graphs by using the sns.barplot, which is a Seaborn function that plots tabular data, which in this 

# case is the covid dataset that was condensed to find female Covid cases

plt.title('COVID Deaths Among American Women Between the Ages of 25-34 with Underlying Medical Issues')

sns.barplot('DeathMonth', 'COVIDUnderlyingCause', hue='RaceEthnicity',

            data=femalecovid);



#It can be seen here that although the proportions of females dying according to their race is already unproportional to 

# their representation in the population, the rates at which each increased and decreased haven't been proportional either

#In the month of March, only Black females have succumbed to COVID.  

#After April the number of Black female deaths has declined over time, but Hispanic females experienced about

# as many deaths in July as they've had in April.  

#Also it can be seen that Native American women have did not experience any COVID deaths until July.

#The number of deaths for White women hasn't exactly followed a specific pattern.
plt.title('COVID Deaths Among American Men Between the Ages of 25-34 with Underlying Medical Issues')

sns.barplot('DeathMonth', 'COVIDUnderlyingCause', hue='RaceEthnicity',

            data=malecovid);

## We can see here that even more promenantly than females, the men has experienced unequal increases and decreases in 

# COVID death rates since the beginning of the pandemic

# We can see that in every month Hispanic men in this age group have succumbed to COVID most frequently since the virus's 

#arrival

# Also, it can be seen that the death rates of Hispanic men have remained high in certain months, such as May and July,

# where death rates have dramatically dropped for all other groups

# This could be due to working conditions, living conditions, health care disparities, or COVID caution levels in this group.

# (likely a combination of the three)
#Conclusion

# As a result of this project, I was able to practice and showcase my skills in Python, Numpy, Pandas, Matplotlib, and

# Seaborn by downloading a CSV file from a government site (cited at the end of the page) of American deaths over the past 

# year. I'm very grateful to have completed this project.  Being a younger American myself, I've wondered for months now 

# how many died many younger Americans died.  I chose this age group because I thought it would be more interesting to compare

# since most Americans in this age group are working.  I also heard previously that death rates were higher for minority groups

# and this was something I wanted to investigate myself.  I also chose to look at those who had underlying heath issues to 

# identity health disparities in the nation. 



# By the end of my data project, I learned that, as I expected, COVID-19 death rates are even across all groups.  In fact, the 

# disparities are even more visual than I previously thought.  I knew the COVID death rates were generally higher for Blacks 

# and Hispanics in America, but not at the alarming rates I have seen in my data visualizations.  Hispanic men especially 

# are passing away from COVID in such high numbers, doubling and even tripling the total number of some other groups (shown in the graph below.  



plt.title('COVID Deaths Among American Men Between the Ages of 25-34 with Underlying Medical Issues')

sns.barplot('DeathMonth', 'COVIDUnderlyingCause', hue='RaceEthnicity',

            data=malecovid);



#I believe if more people say my data, including businesses, hospitals, and communities, more could be done to fight against 

# these disparities.  In the future, someone else could conduct a similar project by acquiring this dataset along with data

# on health issues across different ethnic groups to further show the inequalities minorities face in America.

# Another idea is to compare these death rates in America to those of another diverse country, such as Canada or the United

# Kingdom and make a comparison.  Are unequal share of deaths in certain groups an American problem or an everywhere problem?

# I guess another big question that we still have to ask if more Hispanic and Black American between 25 and 34 have more health 

# issues that exacerbate their COVID conditions, have higher exposure to COVID due to workplace and home environments, 

#or both? And maybe is it more than that?

# All of these questions can be investigated in the future



# In the end, I hope everyone enjoyed reading through my project.  I tried my best to explore a topic that was both relevant

# and fairly uncovered for the most part.  



# Some articles that contributed to my background knowledge on this topic are included below:

# https://time.com/5848557/black-owned-business-coronavirus-aid/

# www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-race-ethnicity.html

# www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html

# www.brookings.edu/blog/up-front/2020/06/16/race-gaps-in-covid-19-deaths-are-even-bigger-than-they-appear/

##Data collected from:

##https://catalog.data.gov/dataset/monthly-provisional-counts-of-deaths-by-age-group-sex-and-race-ethnicity-for-select-causes-6a8fa/resource/d8834e56-4b28-40f9-b044-48ebdbfc44df



#Additional References that contributed to my background knowledge on this topic

## https://time.com/5848557/black-owned-business-coronavirus-aid/

## www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-race-ethnicity.html

## www.cdc.gov/coronavirus/2019-ncov/covid-data/investigations-discovery/hospitalization-death-by-age.html

## www.brookings.edu/blog/up-front/2020/06/16/race-gaps-in-covid-19-deaths-are-even-bigger-than-they-appear/