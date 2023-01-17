# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



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

homicides.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(12, 10), title='Homicides in EEUU between 1980 and 2014');
## Rate of crimes solved

solved = pd.DataFrame(data, columns = ['Crime Solved']) 

resolution = solved.stack().value_counts()

ax = resolution.plot(kind = 'pie',

                              title = 'Crimes solved between 1980 & 2014 (in %)',

                              startangle = 10,

                              autopct='%.2f')

ax.set_ylabel('')
#Gender of victims

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
## Comparation between victims by age // ToDo adjust plot

homicides_adult.to_frame()

homicides_young.to_frame()

homicides = pd.DataFrame({'Adult': homicides_adult,'Young':homicides_young})

homicides.sort_index(inplace=True)

pos = list(range(len(homicides['Adult'])))

width = 0.25



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

print(states.plot(kind='barh', fontsize=10,  width=0.5,  figsize=(12, 10), title='Homicides in EEUU by State between 1980 and 2014'))
#Geographical plot of homicides 

#Requires basemap toolkit



import matplotlib.pyplot as plt

import matplotlib.cm

from mpl_toolkits.basemap import Basemap

from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize



states_eeuu = pd.DataFrame({'homicides':states, 'state':states.index})

states_name = states_eeuu.index

#Set the resolution of the plot 

fig, ax = plt.subplots(figsize=(20,10))



#set type of map (projection) and lat&lon parameters

m = Basemap(resolution='h', # c, l, i, h, f or None

            projection='lcc',

            lat_1=33,lat_2=45,lon_0=-95,

            llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49)



#Set shapefile to print states

m.readshapefile('../input/st99_d00', 'states')



#Set colors of map

m.drawmapboundary(fill_color='#46bcec')

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

m.drawcoastlines()





#Merge info of shapes with homicides stats

geo = pd.DataFrame({

        'shapes': [Polygon(np.array(shape), True) for shape in m.states],

        'state': [state['NAME'] for state in m.states_info]

    })

geo = geo.merge(states_eeuu, on='state', how='left')



#Colour the map

cmap = plt.get_cmap('Oranges')   

pc = PatchCollection(geo.shapes, zorder=2)

norm = Normalize()

 

pc.set_facecolor(cmap(norm(geo['homicides'].fillna(0).values)))

ax.add_collection(pc)



#Add bar

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)



mapper.set_array(geo['homicides'])

plt.colorbar(mapper, shrink=0.4)

plt.title("Geographic homicide distribution")
# Rate of homicides resolution by state



df = pd.DataFrame(data, columns = ['State','Crime Solved']) 

homicides_state = df['State'].value_counts()



#Get homicides solved by state

mask = (df['Crime Solved'] == 'Yes')

homicides_solved = pd.DataFrame(data.loc[mask], columns = ['State', 'Crime Solved']) 

homicides_solved = homicides_solved['State'].value_counts()



#Get homicides unsolved by state

mask2 = (df['Crime Solved'] == 'No')

homicides_unsolved = pd.DataFrame(data.loc[mask2], columns = ['State', 'Crime Solved'])

homicides_unsolved = homicides_unsolved['State'].value_counts()

homicides = pd.DataFrame({'Solved':homicides_solved, 'Unsolved':homicides_unsolved})

homicides['Resolution Rate'] = (homicides['Solved'] *100 ) / (homicides['Solved'] + homicides['Unsolved'])



#Plot Results

state = pd.DataFrame(data, columns = ['State']) 

resolution_rate = pd.DataFrame({'state':states.index, 'resolution rate':homicides['Resolution Rate']})



#Set the resolution of the plot 

fig, ax = plt.subplots(figsize=(20,10))



#set type of map (projection) and lat&lon parameters

m = Basemap(resolution='h', # c, l, i, h, f or None

            projection='lcc',

            lat_1=33,lat_2=45,lon_0=-95,

            llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49)



#Set shapefile to print states

m.readshapefile('../input/st99_d00', 'states')



#Set colors of map

m.drawmapboundary(fill_color='#46bcec')

m.fillcontinents(color='#f2f2f2',lake_color='#46bcec')

m.drawcoastlines()





#Merge info of shapes with homicides stats

geo = pd.DataFrame({

        'shapes': [Polygon(np.array(shape), True) for shape in m.states],

        'state': [state['NAME'] for state in m.states_info]

    })

geo = geo.merge(resolution_rate, on='state', how='left')



#Colour the map

cmap = plt.get_cmap('Greens')   

pc = PatchCollection(geo.shapes, zorder=2)

norm = Normalize()

 

pc.set_facecolor(cmap(norm(geo['resolution rate'].fillna(0).values)))

ax.add_collection(pc)



#Add bar

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)



mapper.set_array(geo['resolution rate'])

plt.colorbar(mapper, shrink=0.4)

plt.title("Resolution rate (in %) of homicides by state")