#importing librareis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

from geopy.exc import GeocoderTimedOut 

from geopy.geocoders import Nominatim
#importing the dataset

df = pd.read_csv(r'../input/data-police-shootings/fatal-police-shootings-data.csv')

df.drop('name', axis=1,inplace=True)

df
#Lets find the deaths in time 

x_months = [0]

y_months = [0]

x_year = [0]

y_year = [0]

temp = 0

for i in df.date: 

    #finding the deaths per month

    temp = i[:7]

    if x_months[-1] == temp:

        y_months[-1] += 1

    else:

        x_months.append(temp)

        y_months.append(1)

    #finding the deaths per year

    temp = i[:4]

    if x_year[-1] == temp:

        y_year[-1] += 1

    else:

        x_year.append(temp)

        y_year.append(1)

        

x_months.pop(0)

y_months.pop(0)

x_year.pop(0)

y_year.pop(0)

f, ax = plt.subplots(1,2,figsize=(20,8))

ax[0].plot(x_months,y_months)

ax[0].axhline(y=np.mean(y_months), color = 'orange', label = 'Average')

ax[0].legend(fontsize=12)

ax[0].set_ylabel('Number of Deaths', fontsize=15)

ax[0].set_xlabel('Months', fontsize=15)

ax[0].set_title('Number of Police Shooting per Month', fontsize=20)

ax[0].xaxis.set_major_locator(ticker.MaxNLocator(10))

ax[1].bar(x_year,y_year)

ax[1].set_xlabel('Years', fontsize=15)

ax[1].set_ylabel('Number of Deaths', fontsize=15)

ax[1].set_title('Number of Police Shooting per Year', fontsize=20)

print('Average Deaths per month: {}\nStandard deviation Deaths per month: {}\nAverage Deaths per Year (not including 2020): {}\nStandard deviation Deaths per year: {}'.format(np.mean(y_months),np.var(y_months)**(1/2),np.mean(y_year),np.var(y_year)**(1/2)))
# lets see about gender, race and age factors

males = len(df.loc[df.gender == 'M'])

wom = len(df.loc[df.gender == 'F'])

blacks = len(df.loc[df.race == 'B'])

white = len(df.loc[df.race == 'W'])

ages = df.age

f, ax = plt.subplots(1,2,figsize=(15,7))

ax[0].set_title('Gender & Race Factor')

ax[0].bar([1,2], [males,wom])

ax[0].bar([4,5], [white,blacks])

ax[0].set_xticks([1,2,4,5])

ax[0].set_xticklabels(['males','females','white', 'blacks'], fontsize = 12)

ax[1].hist(ages,bins=np.arange(0,100,5))

ax[1].set_title('Police Shooting by age')

print('Number of males: {}\nNumber of females: {}\nNumber of White: {}\nNumber of Blacks: {}\nAverage age: {}\nStandar Deviation of age: {}'.format(males,wom,white, blacks, np.mean(ages), np.var(ages)**(1/2)))
#check the scenes of crime

shot = len(df.loc[df.manner_of_death == df.manner_of_death.unique()[0]])

taser = len(df.loc[df.manner_of_death == df.manner_of_death.unique()[1]])

gun = len(df[df.armed == 'gun'])

knife = len(df[df.armed == 'knife'])

unarmed = len(df[df.armed == 'unarmed'])

not_fleeing = len(df[df.flee == 'Not fleeing'])

car = len(df[df.flee == 'Car'])

foot = len(df[df.flee == 'Foot'])

#plotting

f, ax = plt.subplots(1,1,figsize=(15,8))

ax.bar([1,2], [shot,taser], label = 'Manner of Death')

ax.bar([4,5,6], [gun,knife,unarmed], label = 'armed')

ax.bar([8,9,10], [not_fleeing,car,foot], label = 'flee')

ax.set_xticks([1,2,4,5,6,8,9,10])

ax.set_xticklabels(['shot','taser & shot','gun','knife','unarmed','not_fleeing','car','foot'])

ax.legend(fontsize=12)

print('Manner of Death: Shot({}) ,taser & shot({})\nArmed: gun({}), knife({}), unarmed({})\nFlee: not fleeing({}), car({}), foot({})'.format(shot,taser,gun,knife,unarmed,not_fleeing,car,foot))
#Deeper Relationships with gender

u_males = len(df.loc[(df.armed == 'unarmed') & (df.gender == 'M')])

threat_males = len(df.loc[(df.threat_level == 'attack') & (df.gender == 'M')])

flee_males = len(df.loc[(df.flee != 'Not fleeing') & (df.gender == 'M')])

u_females = len(df.loc[(df.armed == 'unarmed') & (df.gender == 'F')])

threat_females = len(df.loc[(df.threat_level == 'attack') & (df.gender == 'F')])

flee_females = len(df.loc[(df.flee != 'Not fleeing') & (df.gender == 'F')])

pltlist = [u_males, u_females,threat_males,threat_females,flee_males,flee_females]

f, ax = plt.subplots(2,1,figsize=(15,16))

for i in range(3):

    ax[0].bar(3*i+1, pltlist[2*i+1], color = ['red'], label='female')

    ax[0].bar(3*i, pltlist[2*i], color = ['blue'],label='male')

ax[0].set_xticks([0.5,3.5,6.5]) 

ax[0].set_xticklabels(['unarmed', 'attack', 'flee'])

ax[0].legend(['males','females'])

ax[0].set_title('Compare Actual Number of male and female victims in different cases', fontsize=15)

for i in range(3):

    ax[1].bar(3*i+1, pltlist[2*i+1]/len(df.loc[(df.gender == 'F')]) *100, color = ['red'], label='female')

    ax[1].bar(3*i, pltlist[2*i]/len(df.loc[(df.gender == 'M')]) *100, color = ['blue'],label='male')

ax[1].set_xticks([0.5,3.5,6.5]) 

ax[1].set_xticklabels(['unarmed', 'attack', 'flee'])

ax[1].legend(['males','females'])

ax[1].set_title('Compare Percentages of male and female victims in different cases', fontsize=15)

ax[1].set_ylabel('100%')
youngs = df.loc[(df.age<18)]

num_of_youngs = len(youngs)

lowest_age = min(youngs.age.unique())

unarmed = len(youngs.loc[youngs.armed == 'unarmed'])

women = len(youngs.loc[youngs.gender == 'F'])

print('Number of under aged children: {}\nLowest age: {}'.format(num_of_youngs, lowest_age))

youngs.loc[youngs.age == 6]

f, ax = plt.subplots(1,2, figsize=(15,6))

ax[0].bar([1,2],[unarmed, num_of_youngs-unarmed])

ax[0].bar([4,5],[women, num_of_youngs-women])

ax[0].set_xticks([1,2,4,5])

ax[0].set_xticklabels(['unarmed','armed','Females','Males'])

ax[0].set_title('Under aged Victims',fontsize=15)

ax[1].hist(youngs.age, bins = range(1,18))

ax[1].set_title('Age Distribution for Under aged Victims',fontsize=15)
illenss = len(df.loc[df.signs_of_mental_illness == True])

noillenss = len(df.loc[df.signs_of_mental_illness == False])

illenss_armed = len(df.loc[(df.signs_of_mental_illness == True) & (df.armed != 'unarmed')])

illenss_unarmed = len(df.loc[(df.signs_of_mental_illness == True) & (df.armed == 'unarmed')])

plt.bar(['Mental Illness', 'No Mental illness'], [illenss,noillenss], width= 0.5)

plt.title('Mental Illness vs No mental Illness')

plt.show()

plt.bar(['armed', 'unarmed'], [illenss_armed,illenss_unarmed], width= 0.5)

plt.title('Mental Illness armed vs unarmed')

print('Mental Illness: {} ({}%)\nNo Mental illness: {} ({}%)'.format(illenss,illenss/(illenss+noillenss)*100,noillenss,noillenss/(illenss+noillenss)*100))