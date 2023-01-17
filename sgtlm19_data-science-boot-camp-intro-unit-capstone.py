import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from PIL import Image

from scipy.stats import ttest_ind
baltimore = pd.read_csv('../input/BPD_Part_1_Victim_Based_Crime_Data.csv')
baltimore.info()
baltimore.isnull().sum()
baltimore.head()
baltimore.tail()
baltimore.iloc[3]['CrimeCode']
img = Image.open('4c.png')

img
baltimore['Description'].unique()
baltimore['Inside/Outside'].value_counts()
#Create new column

baltimore['InsideOutside'] = baltimore['Inside/Outside']



#Get Inside instances

inside1 = int(baltimore[baltimore['InsideOutside'] == 'Inside'].InsideOutside.value_counts())

inside2 = int(baltimore[baltimore['InsideOutside'] == 'I'].InsideOutside.value_counts())



#Get Outside instances

outside1 = int(baltimore[baltimore['InsideOutside'] == 'Outside'].InsideOutside.value_counts())

outside2 = int(baltimore[baltimore['InsideOutside'] == 'O'].InsideOutside.value_counts())



print("There are {} inside instances".format(inside1 + inside2))

print("There are {} outside instances".format(outside1 + outside2))
baltimore['Weapon'].unique()
print(baltimore['Post'].unique())

print(len(baltimore['Post'].unique()))
baltimore.iloc[3]['Post']
img1 = Image.open('934.png')

img1
baltimore['District'].unique()
img2 = Image.open('Baltimore District.png')

img2
print(len(baltimore['Neighborhood'].unique()))
img3 = Image.open('Baltimore Neighborhoods.png')

img3
baltimore['Premise'].unique()
print(len(baltimore['Premise'].unique()))
baltimore['Total Incidents'].unique()
# Create a new column that has the year of which the crime occurred



for n in range(0, 276528):

    x = baltimore.loc[n, 'CrimeDate']

    baltimore.loc[n,'CrimeYear'] = int(x[6:])
# Create a dataframe that has the number of crime occurrences by year from 2012-2017



crime_year = baltimore.CrimeYear.value_counts()

crime_yearindex = crime_year.sort_index(axis=0, ascending=True)

print(crime_yearindex)

# Line plot of crime data from 2012-2017



fig = plt.figure(figsize=(20, 20))

f, ax = plt.subplots(1)

xdata = crime_yearindex.index

ydata = crime_yearindex

ax.plot(xdata, ydata)

ax.set_ylim(ymin=0, ymax=60000)

plt.xlabel('Year')

plt.ylabel('Number of Crimes')

plt.title('Baltimore Crimes from 2012-2017')

plt.show(f)
# Create a new column that has the hour at which the crime occurred



for n in range(0, 276528):

    x = baltimore.loc[n, 'CrimeTime']

    baltimore.loc[n,'CrimeHour'] = int(x[:2])
# Create a dataframe with the crime occurrences by hour 



crime_hour = baltimore.CrimeHour.value_counts()

crime_hourindex = crime_hour.sort_index(axis=0, ascending=True)

print(crime_hourindex)
# There is only one occurrence of one crime at hour 24. For 24-hour format, midnight can be described as either 24:00

# or 00:00, so we will change the observation from 24 to 0



print(baltimore[baltimore['CrimeHour'] == 24])

baltimore.at[239894, 'CrimeHour'] = 0

print(baltimore.loc[239894])
# Incorporate the change of the observation



crime_hour = baltimore.CrimeHour.value_counts()

crime_hourindex = crime_hour.sort_index(axis=0, ascending=True)
# Create line plot that shows the crime occurrence by hour



fig = plt.figure(figsize=(20, 20))

f, ax = plt.subplots(1)

xdata = crime_hourindex.index

ydata = crime_hourindex

ax.plot(xdata, ydata)

ax.set_ylim(ymin=0, ymax=17000)

ax.set_xlim(xmin=0, xmax=24)

plt.xlabel('Hour')

plt.ylabel('Number of Crimes')

plt.title('Number of Crimes by Hour')

plt.show(f)
# Create a pivot table to identify if hours had more occurence of specific crime categories



baltimore.pivot_table(index='Description',

               columns='CrimeHour',

               values='CrimeTime',

               aggfunc= 'count')
#Create a dataframe that has the number of crime occurences by district



districtcount = baltimore.District.value_counts()

baltimore.District.value_counts()
#Create bar graph of number of crimes by district



my_colors = 'rgbkymc'

districtcount.plot(kind='bar',

                color=my_colors,

                title='Number of Crimes Committed by District')
#Create a dataframe that has the occurrence of crimes by category



crimecount = baltimore.Description.value_counts()

baltimore.Description.value_counts()
#Create bar graph of number of crimes by category



my_colors = 'rgbkymc'

crimecount.plot(kind='bar',

                color=my_colors,

                title='Crimes Committed by Category')
# Create a list of unique neighborhoods in Baltimore and then create an empty list which will be appended with larceny

# counts by neighborhood.



neighborhood_list = baltimore['Neighborhood'].unique()

larceny_list = []



# Iterate through unique neighborhood list and then append the count of larceny for that neighborhood to the empty 

# larceny list.



for neighborhood in neighborhood_list:

    x = baltimore[(baltimore['Neighborhood'] == neighborhood) & (baltimore['Description'] == 'LARCENY')]

    larceny_list.append(len(x))



# Create a pandas dataframe with the Larceny counts and sort the values with the highest count at the top 

    

neighborhood_larceny = np.array(larceny_list)

neighborhood_larceny = pd.DataFrame(neighborhood_larceny)

neighborhood_larceny.columns = ['Larceny Counts']

neighborhood_larceny.index = neighborhood_list

neighborhood_larceny = neighborhood_larceny.sort_values(['Larceny Counts'], ascending = False)

print(neighborhood_larceny)
# Plot a histogram of the larceny counts 



#plt.figure()

#neighborhood_larceny.plot.hist(bins=50)



#Plot a histogram for larceny counts.

plt.hist(neighborhood_larceny['Larceny Counts'], bins=20, color='c')



# Add a vertical line at the mean.

plt.axvline(neighborhood_larceny['Larceny Counts'].mean(), color='b', linestyle='solid', linewidth=2)



# Add a vertical line at one standard deviation above the mean.

plt.axvline(neighborhood_larceny['Larceny Counts'].mean() + neighborhood_larceny['Larceny Counts'].std(), color='b', linestyle='dashed', linewidth=2)



# Add a vertical line at one standard deviation below the mean.

plt.axvline(neighborhood_larceny['Larceny Counts'].mean() - neighborhood_larceny['Larceny Counts'].std(), color='b', linestyle='dashed', linewidth=2) 



plt.title('Histogram of Larceny Counts by Neighborhood')



# Print the histogram.

plt.show()
neighborhood_larceny['Larceny Counts'].median()
neighborhood_larceny['Larceny Counts'].mean()
neighborhood_larceny['Larceny Counts'].std()
# Get unique list of neighborhoods in the Northeastern district and create an empty list to hold larceny counts



ne_neighborhoods = baltimore[baltimore['District'] == 'NORTHEASTERN']

ne_neighborhoodlist = ne_neighborhoods['Neighborhood'].unique()



larceny_count1 = []



# Iterate through Northeastern neighborhood list and append the larceny counts to list



for neighborhood in ne_neighborhoodlist:

    x = ne_neighborhoods[(ne_neighborhoods['Neighborhood'] == neighborhood) & (baltimore['Description'] == 'LARCENY')]

    larceny_count1.append(len(x))



# Create a pandas dataframe with the Larceny counts and sort the values with the highest count at the top 

    

ne_larceny = np.array(larceny_count1)

ne_larceny = pd.DataFrame(ne_larceny)

ne_larceny.columns = ['Larceny Counts']

ne_larceny.index = ne_neighborhoodlist

ne_larceny = ne_larceny.sort_values(['Larceny Counts'], ascending = False)

print(ne_larceny)



# Remove the NaN indexed observation



ne_larceny = ne_larceny.loc[ne_larceny.index.dropna()]

print(ne_larceny)
# Get unique list of neighborhoods in the Western district and create an empty list to hold larceny counts



w_neighborhoods = baltimore[baltimore['District'] == 'WESTERN']

w_neighborhoodlist = w_neighborhoods['Neighborhood'].unique()



larceny_count2 = []



# Iterate through Western neighborhood list and append the larceny counts to list



for neighborhood in w_neighborhoodlist:

    x = w_neighborhoods[(w_neighborhoods['Neighborhood'] == neighborhood) & (baltimore['Description'] == 'LARCENY')]

    larceny_count2.append(len(x))



# Create a pandas dataframe with the Larceny counts and sort the values with the highest count at the top 

    

w_larceny = np.array(larceny_count2)

w_larceny = pd.DataFrame(w_larceny)

w_larceny.columns = ['Larceny Counts']

w_larceny.index = w_neighborhoodlist

w_larceny = w_larceny.sort_values(['Larceny Counts'], ascending = False)

print(w_larceny)
#Drop the NaN indexed observation



w_larceny = w_larceny.loc[w_larceny.index.dropna()]

print(w_larceny)
# T-test to determine whether there is a difference between the means of the Northeastern and Western neighborhood counts of

# larceny



diff=ne_larceny['Larceny Counts'].mean( ) - w_larceny['Larceny Counts'].mean()

size = np.array([len(ne_larceny), len(w_larceny)])

sd = np.array([ne_larceny['Larceny Counts'].std(), w_larceny['Larceny Counts'].std()])

diff_se = (sum(sd ** 2 / size)) ** 0.5

t_val = diff/diff_se



print(ttest_ind(ne_larceny['Larceny Counts'], w_larceny['Larceny Counts'], equal_var=False))