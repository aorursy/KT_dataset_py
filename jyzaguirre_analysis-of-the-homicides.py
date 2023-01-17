# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

#read database

data = pd.read_csv('../input/database.csv', na_values=['NA'], dtype='unicode')

years = pd.DataFrame(data, columns = ['Year']) 

count_years = years.stack().value_counts()

homicides = count_years.sort_index(axis=0, ascending=False)

#plot the total of homicides

print(homicides.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(12, 10), title='Homicides in EEUU between 1980 and 2014'))
## Rate of crime's solved



solved = pd.DataFrame(data, columns = ['Crime Solved']) 

resolution = solved.stack().value_counts()

ax = resolution.plot(kind = 'pie',

                              title = 'Rates of crimes solved between 1980 & 2014 (in %)',

                              startangle = 10,

                              autopct='%.2f')

ax.set_ylabel('')
##Sex of the victims

sex = pd.DataFrame(data, columns = ['Victim Sex']) 

count_sex = sex.stack().value_counts()

ax = count_sex.plot(kind = 'pie',

                              title = 'Sex of the victims',

                              startangle = 10,

                              autopct='%.2f')

ax.set_ylabel('')
#Race of Victims

race = pd.DataFrame(data, columns = ['Victim Race']) 

count_race = race.stack().value_counts()

ax = count_race.plot(kind = 'pie',

                              title = 'Race of the victims',

                              startangle = 10,

                              autopct='%.2f',

                              explode=(0, 0, 0.7, 1, 1.3))

ax.set_ylabel('')
#Victims under 21



data['Victim Age'] = data['Victim Age'].astype("int")

mask = (data['Victim Age'] < 21)

young_victims =  pd.DataFrame(data.loc[mask], columns = ['Year']) 

count_years = young_victims.stack().value_counts()

homicides_young = count_years.sort_index(axis=0, ascending=False)

mask2 = (data['Victim Age'] > 21)

adult_victims =  pd.DataFrame(data.loc[mask2], columns = ['Year']) 

count_years = adult_victims.stack().value_counts()

homicides_adult = count_years.sort_index(axis=0, ascending=False)

print(homicides_young.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(12, 10), title='Victims under 21 years old'))
## Comparation between victims by age

homicides_adult.to_frame()

homicides_young.to_frame()

homicides = pd.DataFrame({'Adult': homicides_adult,'Young':homicides_young})

homicides.sort_index(inplace=True)

pos = list(range(len(homicides['Adult'])))

width = 0.15



# Plotting the bars

fig, ax = plt.subplots(figsize=(25,15))



# in position pos,

plt.bar(pos,

        #using homicides['Adult'] data,

        homicides['Adult'],

        # of width

        width,

        # with alpha 0.5

        alpha=0.5,

        # with color

        color='#EE3224',

        # with label the first value in year

        label=homicides.index[0])



# Create a bar with young data,

# in position pos + some width buffer,

plt.bar([p + width for p in pos],

        #using homicides['Young'] data,

        homicides['Young'],

        # of width

        width,

        # with alpha 0.5

        alpha=0.5,

        # with color

        color='#F78F1E',

        # with label the second value in year

        label=homicides.index[1])







# Set the y axis label

ax.set_ylabel('Adult / Young')



# Set the chart's title

ax.set_title('Comparation between victims by age')



# Set the position of the x ticks

ax.set_xticks([p + 1.5 * width for p in pos])



# Set the labels for the x ticks

ax.set_xticklabels(homicides.index)



# Setting the x-axis and y-axis limits

plt.xlim(min(pos)-width, max(pos)+width*4)

plt.ylim([0, max(homicides['Adult'] + homicides['Young'])] )



# Adding the legend and showing the plot

plt.legend(['Adult', 'Young'], loc='upper left')



plt.grid()

plt.show()

# Sex of the perpetrators



perpetrator_sex = pd.DataFrame(data, columns = ['Perpetrator Sex']) 

count_perpetrator_sex = perpetrator_sex.stack().value_counts()

ax = count_perpetrator_sex.plot(kind = 'pie',

                              title = 'Sex of the perpetrators',

                              startangle = 10,

                              autopct='%.2f')

ax.set_ylabel('')
# Race of the perpetrators

perpetrator_race = pd.DataFrame(data, columns = ['Perpetrator Race']) 

count_perpetrator_race = perpetrator_race.stack().value_counts()

ax = count_perpetrator_race.plot(kind = 'pie',

                              title = 'Race of the perpetrators',

                              startangle = 15,

                              autopct='%.2f',

                              explode=(0, 0, 0, 1, 1.3))

ax.set_ylabel('')
#Crime types

crime_types = pd.DataFrame(data, columns = ['Crime Type']) 

count_types = crime_types.stack().value_counts()

count_crime_types = count_types.sort_index(axis=0, ascending=False)

#plot the total of homicides



ax = count_crime_types.plot(kind = 'pie',

                              title = 'Crime Types',

                              startangle = 25,

                              autopct='%.2f')

ax.set_ylabel('')
#Crimes by State

state = pd.DataFrame(data, columns = ['State']) 

count_states = state.stack().value_counts()

states = count_states.sort_index(axis=0, ascending=False)

#plot the total of homicides

print(states.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(10, 10), title='Homicides in EEUU by State between 1980 and 2014'))