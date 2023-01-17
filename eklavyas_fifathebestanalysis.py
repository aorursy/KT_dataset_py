import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('../input/fifathebest/FIFATheBest.csv')
data.head()
data.describe()
surnames = {

    'Messi Lionel': 'Messi',

    'De Jong Frenkie': 'De Jong',

    'Cristiano Ronaldo': 'Ronaldo',

    'Salah Mohamed': 'Salah',

    'Van Dijk Virgil': 'Van Dijk',

    'Kane Harry': 'Kane',

    'De Ligt Matthijs': 'De Ligt',

    'Hazard Eden': 'Hazard',

    'Mané Sadio': 'Mané',

    'Mbappé Kylian': 'Mbappé',

}
clubCols = {

    'Messi Lionel': 'MediumBlue',

    'De Jong Frenkie': 'MediumBlue',

    'Cristiano Ronaldo': 'Black',

    'Salah Mohamed': 'Firebrick',

    'Van Dijk Virgil': 'Firebrick',

    'Kane Harry': 'Turquoise',

    'De Ligt Matthijs': 'Black',

    'Hazard Eden': 'Lightgrey',

    'Mané Sadio': 'Firebrick',

    'Mbappé Kylian': 'Midnightblue',

}
topPlayers = set(data['First'])
topPlayersCounts = {}



for topPlayer in topPlayers:

    topPlayersCounts[topPlayer] = 0

    

topPlayersCounts
for topPlayer in topPlayers:

    firstCount  = data[data['First']==topPlayer]['First'].value_counts().values[0]

    secondCount = data[data['Second']==topPlayer]['Second'].value_counts().values[0]

    thirdCount  = data[data['Third']==topPlayer]['Third'].value_counts().values[0]

    topPlayersCounts[topPlayer] = firstCount + secondCount + thirdCount
sortedTopPlayers = sorted(topPlayersCounts.items(), key=lambda item: item[1])

names = []

pts = []

cols = []



for k,v in sortedTopPlayers[::-1]:

    names.append(k)

    pts.append(v)

    cols.append(clubCols[k])

    

pcpts = pts/sum(pts)

pcpts = np.round(pcpts*100, 2)
namesTotal = [surnames[name] for name in names]

valsTotal = pts

colsTotal = [clubCols[name] for name in names]
plt.figure(figsize=(8,3), dpi=200)

plt.title('Total Votes')

plt.ylabel('Counts')

plt.bar(namesTotal, valsTotal, color=colsTotal, zorder=3)

plt.tight_layout()

plt.savefig('TotalCount.png', dpi=200)

plt.show()
first = data['First'].value_counts()

second = data['Second'].value_counts()

third = data['Third'].value_counts()



namesFirst = [surnames[names] for names in first.index.to_numpy()]

valsFirst = first.values

colsFirst = [clubCols[names] for names in first.index.to_numpy()]



namesSecond = [surnames[names] for names in second.index.to_numpy()]

valsSecond = second.values

colsSecond = [clubCols[names] for names in second.index.to_numpy()]



namesThird = [surnames[names] for names in third.index.to_numpy()]

valsThird = third.values

colsThird = [clubCols[names] for names in third.index.to_numpy()]
# Config

fig, axes = plt.subplots(3,1, figsize=(10, 8), dpi=200)

fig.suptitle('      Most Total Votes per Position')

axes[0].set_title('First Place')

axes[1].set_title('Second Place')

axes[2].set_title('Third Place')

axes[1].set_ylabel('Counts')



# Plot

axes[0].bar(namesFirst, valsFirst, color=colsFirst, zorder=3)

axes[1].bar(namesSecond, valsSecond, color=colsSecond, zorder=3)

axes[2].bar(namesThird, valsThird, color=colsThird, zorder=3)



# Markup

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('MostTotalVotesPerPosition.png', dpi=200)

plt.show()
first = data[data['Role']=='Media']['First'].value_counts()

second = data[data['Role']=='Media']['Second'].value_counts()

third = data[data['Role']=='Media']['Third'].value_counts()



namesFirst = [surnames[names] for names in first.index.to_numpy()]

valsFirst = first.values

colsFirst = [clubCols[names] for names in first.index.to_numpy()]



namesSecond = [surnames[names] for names in second.index.to_numpy()]

valsSecond = second.values

colsSecond = [clubCols[names] for names in second.index.to_numpy()]



namesThird = [surnames[names] for names in third.index.to_numpy()]

valsThird = third.values

colsThird = [clubCols[names] for names in third.index.to_numpy()]
# Config

fig, axes = plt.subplots(3,1, figsize=(10, 8), dpi=200)

fig.suptitle('      Most Media Votes per Position')



axes[0].set_title('First Place')

axes[1].set_title('Second Place')

axes[2].set_title('Third Place')

axes[1].set_ylabel('Counts')



# Plot

axes[0].bar(namesFirst, valsFirst, color=colsFirst, zorder=3)

axes[1].bar(namesSecond, valsSecond, color=colsSecond, zorder=3)

axes[2].bar(namesThird, valsThird, color=colsThird, zorder=3)



# Markup

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('MostMediaVotesPerPosition.png', dpi=200)

plt.show()
first = data[data['Role']=='Captain']['First'].value_counts()

second = data[data['Role']=='Captain']['Second'].value_counts()

third = data[data['Role']=='Captain']['Third'].value_counts()



namesFirst = [surnames[names] for names in first.index.to_numpy()]

valsFirst = first.values

colsFirst = [clubCols[names] for names in first.index.to_numpy()]



namesSecond = [surnames[names] for names in second.index.to_numpy()]

valsSecond = second.values

colsSecond = [clubCols[names] for names in second.index.to_numpy()]



namesThird = [surnames[names] for names in third.index.to_numpy()]

valsThird = third.values

colsThird = [clubCols[names] for names in third.index.to_numpy()]
# Config

fig, axes = plt.subplots(3,1, figsize=(10, 8), dpi=200)

fig.suptitle('      Most Captain Votes per Position')



axes[0].set_title('First Place')

axes[1].set_title('Second Place')

axes[2].set_title('Third Place')

axes[1].set_ylabel('Counts')



# Plot

axes[0].bar(namesFirst, valsFirst, color=colsFirst, zorder=3)

axes[1].bar(namesSecond, valsSecond, color=colsSecond, zorder=3)

axes[2].bar(namesThird, valsThird, color=colsThird, zorder=3)



# Markup

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('MostCaptainVotesPerPosition.png', dpi=200)

plt.show()
first = data[data['Role']=='Coach']['First'].value_counts()

second = data[data['Role']=='Coach']['Second'].value_counts()

third = data[data['Role']=='Coach']['Third'].value_counts()



namesFirst = [surnames[names] for names in first.index.to_numpy()]

valsFirst = first.values

colsFirst = [clubCols[names] for names in first.index.to_numpy()]



namesSecond = [surnames[names] for names in second.index.to_numpy()]

valsSecond = second.values

colsSecond = [clubCols[names] for names in second.index.to_numpy()]



namesThird = [surnames[names] for names in third.index.to_numpy()]

valsThird = third.values

colsThird = [clubCols[names] for names in third.index.to_numpy()]
# Config

fig, axes = plt.subplots(3,1, figsize=(10, 8), dpi=200)

fig.suptitle('      Most Coach Votes per Position')



axes[0].set_title('First Place')

axes[1].set_title('Second Place')

axes[2].set_title('Third Place')

axes[1].set_ylabel('Counts')



# Plot

axes[0].bar(namesFirst, valsFirst, color=colsFirst, zorder=3)

axes[1].bar(namesSecond, valsSecond, color=colsSecond, zorder=3)

axes[2].bar(namesThird, valsThird, color=colsThird, zorder=3)



# Markup

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('MostCoachVotesPerPosition.png', dpi=200)

plt.show()