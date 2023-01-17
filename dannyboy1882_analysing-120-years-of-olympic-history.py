# Import relevant modules.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import os



# Load the main dataset.

dataAll  = pd.read_csv('../input/athlete_events.csv')

dataAll.head(3)
# Examine the details of team names.

print(dataAll['Team'].nunique())

teamNames = dataAll['Team'].unique()

teamNames.sort()

print(teamNames[0:3])
# Load the updated regions dataset.

nocNames = pd.read_csv('../input/noc_regions.csv')

nocNames.head(3)
# Based on NOC name, merged the datasets.

dataAll = pd.merge(dataAll, nocNames, on='NOC', how='left')

dataAll.head(3)
# Create new dataframes for just the Summer or Winter Games.

summerGames = dataAll[(dataAll.Season == 'Summer')]

winterGames = dataAll[(dataAll.Season == 'Winter')]



# Count the number of unique events after grouping by year.

numSummerEvents = summerGames.groupby(["Year"])["Event"].nunique().reset_index()

numWinterEvents = winterGames.groupby(["Year"])["Event"].nunique().reset_index()



# Plot the results.

plt.figure(figsize=(15,8))

plt.plot(numSummerEvents.Year, numSummerEvents.Event, '-o')

plt.plot(numWinterEvents.Year, numWinterEvents.Event, '-o')

plt.xlabel("Year", fontsize=22)

plt.ylabel("Number of Events", fontsize=22)

plt.legend(["Summer", "Winter"])

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Take a look at some of the art-based events.

earlyEvents = summerGames[(summerGames.Sport == 'Art Competitions')]['Event'].unique()

print(earlyEvents[0:11])
# Create new dataframes for male and female athletes at the Summer games.

summerGamesMale   = summerGames[(summerGames.Sex == 'M')]

summerGamesFemale = summerGames[(summerGames.Sex == 'F')]



# First group by year, then based on athlete ID, count the number of individuals.

numSummerMale   = summerGamesMale.groupby(["Year"])["ID"].nunique().reset_index()

numSummerFemale = summerGamesFemale.groupby(["Year"])["ID"].nunique().reset_index()



# Plot the results.

plt.figure(figsize=(15,8))

plt.plot(numSummerMale.Year, numSummerMale.ID, '-o')

plt.plot(numSummerFemale.Year, numSummerFemale.ID, '-o')

plt.xlabel("Year", fontsize=22)

plt.ylabel("Number of Athletes", fontsize=22)

plt.legend(["Male", "Female"])

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Calculate the ratio of male/female participants.

# I remove the first year as no women participated.

maleRatios   = numSummerMale[1::].ID/(numSummerMale[1::].ID + numSummerFemale.ID)

femaleRatios = numSummerFemale.ID/(numSummerMale[1::].ID + numSummerFemale.ID)



# Plot the results as a stacked bar chart.

plt.figure(figsize=(15,8))

width = 2 

p1 = plt.bar(numSummerMale.Year, maleRatios, width)

p2 = plt.bar(numSummerMale.Year, femaleRatios, width, bottom=maleRatios)

plt.xlabel("Year", fontsize=22)

plt.ylabel("Gender Ratio", fontsize=22)

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# After first grouping by year, count the number of regions competing.

numSummerCountries = summerGames.groupby(["Year"])["region"].nunique().reset_index()

numWinterCountries = winterGames.groupby(["Year"])["region"].nunique().reset_index()



# Plot the results.

plt.figure(figsize=(15,8))

plt.plot(numSummerCountries.Year, numSummerCountries.region, '-o')

plt.plot(numWinterCountries.Year, numWinterCountries.region, '-o')

plt.xlabel("Year", fontsize=22)

plt.ylabel("Number of Participating Regions", fontsize=22)

plt.legend(["Summer", "Winter"])

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Use drop_duplicates to remove multiple entries by the same athlete in a single year.

maleAges = summerGamesMale.drop_duplicates(['ID', 'Year'])['Age']

# Remove NaN entries.

maleAges = maleAges[np.isfinite(maleAges)]



femaleAges = summerGamesFemale.drop_duplicates(['ID', 'Year'])['Age']

femaleAges = femaleAges[np.isfinite(femaleAges)]



plt.figure(figsize=(15,8))

# Calculate and plot histograms.

n1, bins1, patches1 = plt.hist(maleAges, range(0, 101), alpha=0.75, density='true')

n2, bins2, patches2 = plt.hist(femaleAges, range(0, 101), alpha=0.75, density='true')

# Add a step plot overlay to make boundaries clear.

plt.step(range(0, 100), n1, 'b', where='post')

plt.step(range(0, 100), n2, 'r', where='post')

plt.xlabel("Age", fontsize=22)

plt.ylabel("Normalised Count", fontsize=22)

plt.legend(["Male", "Female"])

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Calculate the mean and standard deviation of athlete ages.

print('Average Male Athlete:', 

      format(maleAges.mean(), '.1f'), 

      '+/-', 

      format(maleAges.std(), '.1f'),

      'years old.')



print('Average Female Athlete:', 

      format(femaleAges.mean(), '.1f'), 

      '+/-', 

      format(femaleAges.std(), '.1f'),

      'years old.')
# Find the details of the youngest athlete.

dataAll[(dataAll.Age == dataAll['Age'].min())]
# Find the details of the oldest athlete.

dataAll[(dataAll.Age == dataAll['Age'].max())]
# Choose four high profile sports.

femaleSports = ['Athletics',

                'Equestrianism',

                'Gymnastics',

                'Swimming']



# Create a figure and loop over each sport.

plt.figure(figsize=(15,8))

for ii, sportName in enumerate(femaleSports):

    

    # Choose only female athletes at the Summer Games for the current sport.

    currentSport = summerGamesFemale[(summerGamesFemale.Sport == sportName)]

    # Only consider medal winners.

    currentSport = currentSport[currentSport.Medal.notnull()]

    # Remove NaN entries.

    currentSport = currentSport[(np.isfinite(currentSport.Age))]

    # After first grouping by year, calculate mean ages.

    currentSport = currentSport.groupby(["Year"])["Age"].mean().reset_index()

    # Plot results for current sport.

    plt.plot(currentSport.Year, currentSport.Age, '-o')

    

# Tidy up the plot.

plt.xlabel("Year", fontsize=22)

plt.ylabel("Average Age of Medal Winners", fontsize=22)

plt.legend(femaleSports)

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Select only medal winners at Summer Games.

summerMedalWinners = summerGames[summerGames.Medal.notnull()]

# Remove entries for unrecognised 1906 Games.

summerMedalWinners = summerMedalWinners[summerMedalWinners.Year != 1906]



# Store the names of all winning regions.

winningTeams = summerMedalWinners['region'].unique()

# Create a list of zeros to store results.

medalCount   = [0]*len(winningTeams)



# Loop over each region that has ever won a medal.

for ii, teamName in enumerate(winningTeams):

    

    # Choose only medal winning athletes for the current team.

    currentTeam = summerMedalWinners[(summerMedalWinners.region == teamName)]

    # Need to be careful to remove multiple medals won by teams (like football).

    currentTeam = currentTeam.drop_duplicates(['Year', 'Event', 'Medal'])

    # Store the total medal count for the current region.

    medalCount[ii] = currentTeam["ID"].count()

    

# Create a medal table dataframe and view the top 10.

data = {'Team':winningTeams, 'Total Medal Count':medalCount}

medalTable = pd.DataFrame(data)

medalTable = medalTable.sort_values('Total Medal Count', ascending=False).reset_index(drop=True)

medalTable.head(10)
# Choose the eight most recent hosts.

recentHostCountries = ['South Korea',

                       'Spain',

                       'USA',

                       'Australia',

                       'Greece',

                       'China',

                       'UK',

                       'Brazil']



# The years they hosted and a list of zeros to store results.

hostYears = [1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016]

hostCount   = np.empty([len(recentHostCountries), len(hostYears)])



# Loop over each year.

for ii in range(0, len(hostYears)):

    

    # Choose all medal winners from the current year.

    currentYear = summerMedalWinners[(summerMedalWinners.Year == hostYears[ii])]

    # Need to be careful to remove multiple medals won by teams.

    currentYear = currentYear.drop_duplicates(['Year', 'Event', 'Medal'])

    # Count all the available medals in the current year.

    totalMedals = currentYear["ID"].count()

    

    # Loop over each host country.

    for jj in range(0, len(recentHostCountries)):

        

        # Choose the current host country and calculate the percentage of 

        # total medals they won in the current year.

        currentTeam = currentYear[(currentYear.region == recentHostCountries[jj])]

        hostCount[ii,jj] = (currentTeam["ID"].count()/totalMedals)*100



# Create some empty arrays to store the results.

meanResult = np.empty([len(recentHostCountries)])

bestResult = np.empty([len(recentHostCountries)])

hostResult = np.empty([len(recentHostCountries)])



# Plot each host nations medal results and store mean, best and host year medal results.

plt.figure(figsize=(15,8))

for nn in range(0, len(recentHostCountries)):

    

    plt.plot(hostYears, hostCount[:,nn], '-o')

    meanResult[nn] = hostCount[:,nn].mean()

    bestResult[nn] = max(hostCount[:,nn])

    hostResult[nn] = hostCount[nn,nn]

    

# Tidy the figure.

plt.xlabel("Year", fontsize=22)

plt.ylabel("Percentage of Available Medals Won", fontsize=22)

plt.legend(recentHostCountries, bbox_to_anchor=(1.04,1), loc="upper left")

plt.grid(True)

plt.rc('xtick', labelsize=16)

plt.rc('ytick', labelsize=16)
# Create a medal table dataframe.

data = {'Team':recentHostCountries, 'Hosting Year':hostYears, 'Average (%)':meanResult, 'Best (%)':bestResult, 'Host Year (%)':hostResult}

hostTable = pd.DataFrame(data)

hostTable.head(8)