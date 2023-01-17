import pandas as pd



data = pd.read_csv('../input/CompleteDataset.csv')

data.head(5)
pd.set_option("display.max_columns",50)



print(pd.get_option("display.max_columns"))

data.head(5)
#Which country produces the most players in FIFA?

print(data.shape)

country_count = data['Nationality'].value_counts()

#Plot top 20 countries with most players

country_count[0:20].plot.bar()

#What proportion of Fifa is the country of?

def country_proportions(country):

    length = len(data[data['Nationality'] == country])

    return str(round(length/data.shape[0],5)*100) + "%"

print("England's proportion in game is " + country_proportions('England'))
import re

import matplotlib.pyplot as plt



#Convert wages to int with RE

int_wage = []

for i in data.Wage:

    i = re.search('[0-9]+',i)

    int_wage.append(i.group(0))



#Add int_wage to df

data['int_wage'] = int_wage



#scatter plot overall vs int_wage

plt.scatter(data.Overall,data.int_wage)

plt.show()
#Are some people really paid €0?

data[data['Wage']== "€0"]

#Yes...I wonder if I can find out why?
plt.scatter(data.Potential,data.Overall)

plt.show()

#This means you can't ever rate higher than your potential
#Which club has the highest average of players?

#Top 30 teams with highest average of players

club_average = data.groupby('Club').mean().sort_values(by="Overall",ascending=False)

club_average
#Are teams almost maxing out their potential?



club_average['Potential Index'] = club_average.Potential - club_average.Overall

club_average.sort_values(by='Potential Index')



#Does this mean that older teams have less potential? 

plt.scatter(club_average['Age'],club_average['Potential Index'])

plt.xlabel('Average Age Per Club')

plt.ylabel('Potential Index')

plt.show()



#Makes sense, younger clubs are further away from meeting their potential
#What age 