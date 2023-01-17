# Import the libraries we need

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # for graphing



df = pd.read_csv('../input/ca_offenses_by_city.csv', thousands=',')
# To start, I will look at the overall crime picture. I'll add them up first, then come back

# and look at the categories more closely

df = df[['City', 'Population', 'Violent crime', 'Property crime', 'Arson']]



# Let's take a look at a row

df.ix[2]
# This bit of code gets the crime columns into int data types, then sums them up

col_list= list(df)

col_list.remove('City')

col_list.remove('Population')



df[['Violent crime','Property crime', 'Arson']] = df[['Violent crime','Property crime', 'Arson']].astype(int)

df['All crimes'] = df[col_list].sum(axis=1)



# Let's look at a summary of the total crime numbers

df['All crimes'].describe()
# And population while we're at it

#df['Population'].describe()

#huge = df['Population'] > 100000

#df[huge].sort_values(by='Population')

df.ix[233]
# Let's create a new dataframe without Los Angeles

dfx = df.drop([233])

dfx['Population'].describe()
# Let's create a normalized "batting average" to compare cities by

# We'll call it 'Crime ratio'

df['Crime ratio'] = df['All crimes'] / df['Population']

df['Crime ratio'].describe()
# Let's graph population vs. crimes

# We should see a positive correlation

x = df.ix[:,'Population']

y = df.ix[:, 'All crimes']



plt.title("California Cities: Population vs. Total Crimes")

plt.xlabel("Population")

plt.ylabel("Total Crimes")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
# Now let's graph population vs. crimes without Los Angeles (the big outlier)

x = dfx.ix[:,'Population']

y = dfx.ix[:, 'All crimes']



plt.title("California Cities: Population vs. Total Crimes (without LA)")

plt.xlabel("Population")

plt.ylabel("Total Crimes")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
wildWest = dfx['All crimes'] > 20000

dfx[wildWest].sort_values(by='Population', ascending=False)
low = df['Crime ratio'] < .01

df[low].sort_values(by='Crime ratio')
high = df['Crime ratio'] > 0.05

df[high].sort_values(by='Crime ratio', ascending=False)
# Let's graph population vs. violent crimes

# We should see a positive correlation

x = df.ix[:,'Population']

y = df.ix[:, 'Violent crime']



plt.title("California Cities: Population vs. Violent Crime")

plt.xlabel("Population")

plt.ylabel("Violent Crime")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
# Let's graph population vs. violent crimes without LA

# We should see a positive correlation

x = dfx.ix[:,'Population']

y = dfx.ix[:, 'Violent crime']



plt.title("California Cities: Population vs. Violent Crime (without LA)")

plt.xlabel("Population")

plt.ylabel("Violent Crime")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
# Violent crime ratio

df['Violent crime ratio'] = df['Violent crime'] / df['All crimes']

df['Violent crime ratio'].describe()
low = df['Violent crime ratio'] < 0.025

df[low].sort_values(by='Violent crime ratio')
high = df['Violent crime ratio'] > 0.19

df[high].sort_values(by='Violent crime ratio', ascending=False)
# Let's graph population vs.property crime without LA

x = dfx.ix[:,'Population']

y = dfx.ix[:, 'Property crime']



plt.title("California Cities: Population vs. Property Crime")

plt.xlabel("Population")

plt.ylabel("Property Crime")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
# Let's graph population vs. Arson without LA

x = dfx.ix[:,'Population']

y = dfx.ix[:, 'Arson']



plt.title("California Cities: Population vs. Arson (without LA)")

plt.xlabel("Population")

plt.ylabel("Arson")

plt.scatter(x, y)

# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
high = dfx['Arson'] > 300

dfx[high].sort_values(by='Arson', ascending=False)
dfv = pd.read_csv('../input/ca_offenses_by_city.csv', thousands=',')

dfv.describe()
# Shorten column names

dfv = dfv[['City','Population','Violent crime', 'Murder and nonnegligent manslaughter', 'Rape (revised definition)','Robbery','Aggravated assault']]

dfv.describe()

dfv.columns = ['City','Population','Violent crime','Murder','Rape','Robbery','Assault']
plt.figure(0)

plt.subplot(2,2,1)

x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Murder']

#plt.title('California Cities: Pop. vs. Murder')

#plt.xlabel('Population')

plt.ylabel('Murder')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))



plt.subplot(2,2,2)

#x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Rape']

#plt.title('California Cities: Pop. vs. Rape')

#plt.xlabel('Population')

plt.ylabel('Rape')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))



plt.subplot(2,2,3)

#x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Robbery']

#plt.title('California Cities: Pop. vs. Murder')

#plt.xlabel('Population')

plt.ylabel('Robbery')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))



plt.subplot(2,2,4)

#x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Assault']

#plt.title('California Cities: Pop. vs. Rape')

#plt.xlabel('Population')

plt.ylabel('Assault')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))



plt.show()
dfv = dfv.drop([233])
x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Murder']

plt.title('California Cities: Population vs. Murder (w/o LA)')

plt.xlabel('Population')

plt.ylabel('Murder')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
high = dfv['Murder'] > 40

dfv[high].sort_values(by='Murder', ascending=False)
# Let's look at the murder ratio:

dfv['Murder ratio'] = dfv['Murder'] / dfv['Population']

dfv['Murder ratio'].describe()
high = dfv['Murder ratio'] > 0.0002

dfv[high].sort_values(by='Murder ratio', ascending=False)
x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Rape']

plt.title('California Cities: Population vs. Rape (w/o LA)')

plt.xlabel('Population')

plt.ylabel('Rape')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
# Rape ratio

dfv['Rape ratio'] = dfv['Rape'] / dfv['Population']

dfv['Rape ratio'].describe()



high = dfv['Rape ratio'] > 0.0008

dfv[high].sort_values(by='Rape ratio', ascending=False)
x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Robbery']

plt.title('California Cities: Population vs. Robbery (w/o LA)')

plt.xlabel('Population')

plt.ylabel('Robbery')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
high = dfv['Robbery'] > 1000

dfv[high].sort_values(by='Robbery', ascending=False)
x = dfv.ix[:,'Population']

y = dfv.ix[:, 'Assault']

plt.title('California Cities: Population vs. Assault (w/o LA)')

plt.xlabel('Population')

plt.ylabel('Assault')

plt.scatter(x, y)

plt.grid()



# This next bit of code places a best-fit line on the plot

plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))

plt.show()
high = dfv['Assault'] > 1000

dfv[high].sort_values(by='Assault', ascending=False)