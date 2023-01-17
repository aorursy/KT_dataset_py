import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

listdir = os.listdir("../input")

csv = []

#for x in listdir:

 #   csv.append(pd.read_csv("../input/"+x))

print(listdir)

# Any results you write to the current directory are saved as output.
listdir = os.listdir("../input")



#frame1 = pd.DataFrame((pd.read_csv("../input/"+listdir[1]))['matchid'])

#frame2 = pd.DataFrame((pd.read_csv("../input/"+listdir[1]))['ss2'])



df = (pd.read_csv("../input/"+listdir[4]))

#graph = df[df ['win'] !=0]

graph = [[0,0,0,0]]

for index, row in df.iterrows():

    if row['win'] == row['firstblood'] == 0:

        graph[0][0] += 1

    elif row['win'] == row['firstblood'] == 1:

        graph[0][1] += 1

    elif row['win'] == 1 and row['firstblood'] == 0:

        graph[0][2] += 1

    else:

        graph[0][3] +=1

graph = pd.DataFrame(graph, columns = ['wFb0', 'wFb1', 'w1Fb0', 'w0Fb1'])

graph.plot.bar()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.



import os

listdir = os.listdir("../League Matches")

df = pd.DataFrame(pd.read_csv("../League Matches/"+listdir[4], dtype =str))

loseperkills = df[df['win'] == '0']

loseperkills['kills'].value_counts().plot.bar()

plt.show()
listdir = os.listdir("../input")

df = pd.DataFrame(pd.read_csv("../input/"+listdir[4], dtype =str))

winperkills = df[df['win'] == '1']

winperkills['kills'].value_counts().plot.bar()

plt.show()
# Input data files are available in the "../input/" directory.

listdir = os.listdir("../input")

#stats1df = pd.DataFrame(pd.read_csv("../input/"+listdir[4], dtype =str))

#stats2df = pd.DataFrame(pd.read_csv("../input/"+listdir[6], dtype =str))

champs = (pd.DataFrame(pd.read_csv("../input/"+listdir[0], dtype =str)))

particpantsdf = pd.DataFrame(pd.read_csv("../input/"+listdir[1], dtype =str))

bansdf = pd.DataFrame(pd.read_csv("../input/"+listdir[2], dtype =str))

champsDict = {}

for index in champs['id']:

        champsDict[index] = 0 

for index, rows in particpantsdf.iterrows():

	champsDict[rows['championid']] += 1

print(champsDict)

pd.DataFrame(champsDict, columns = champs['name']).plot.bar()

plt.show()
#listdir = os.listdir('../DataScience/League Matches/')

#print(listdir)

stats1df = pd.read_csv("../DataScience/League Matches/stats1.csv", dtype ='int64')

stats2df = pd.read_csv("../DataScience/League Matches/stats2.csv", dtype ='int64')

print (stats1df[0:10], stats2df[0:10])

statsdf = pd.concat([stats1df, stats2df])

#champs = (pd.DataFrame(pd.read_csv("../DataScience/League Matches/"+listdir[0], dtype =str)))

particpantsdf = pd.DataFrame(pd.read_csv("../DataScience/League Matches/"+listdir[2], dtype =str))

damagePerteam = []

killsPerteam = []

visionPerteam = []

# [[{role:dmg, role: damage...},{role:dmg, role: damage...}]]

tempdmg = []

tempkda = []

tempvision =[]

for index, rows in particpantsdf.iterrows():

	role = ''

	if rows['role'] == 'SOLO':

		role = rows['position']

		tempdmg[role] = int(statsdf[index]['totdmgtochamps'])

		tempkda[role] = (statsdf[index]['kills'] + statsdf[index]['assists']) / double(statsdf[index]['deaths'])

		tempvision[role] = int(statsdf[index]['visionscore'])

	else:

		tempdmg[rows['role']] = int(statsdf[index]['totdmgtochamps'])

		tempkda[rows['role']] = (statsdf[index]['kills'] + statsdf[index]['assists']) / double(statsdf[index]['deaths'])

		tempvision[rows['role']] = int(statsdf[index]['visionscore'])

	if index != 0 and index % 5 == 0:

		damagePerteam.append(tempdmg)

		killsPerteam.append(tempkda)

		visionPerteam.append(tempvision)

		tempdmg.clear()

		tempkda.clear()

		tempvision.clear()

# List of temp variables for the calculations

sumSupport1 = 0;

sumSupport2 = 0;

sumSupport3 = 0;

sumAdc1 = 0;

sumAdc2 = 0;

sumAdc3 = 0;

sumTop1 = 0;

sumTop2 = 0;

sumTop3 = 0;

sumMid1 = 0;

sumMid2 = 0;

sumMid3 = 0;

sumJungle1 = 0;

sumJungle2 = 0;

sumJungle3 = 0;

# Sum up the values of each role and the cooresponding data researched

for index in range(len(damagePerteam)):

	sumSupport1 += damagePerteam[x]['DUO_SUPPORT']

	sumSupport2 += killsPerteam[x]['DUO_SUPPORT']

	sumSupport3 += visionPerteam[x]['DUO_SUPPORT']

	

	sumAdc1 += damagePerteam[x]['DUO_CARRY']

	sumAdc2 += killsPerteam[x]['DUO_CARRY']

	sumAdc3 += visionPerteam[x]['DUO_CARRY']

	

	sumTop1 += damagePerteam[x]['TOP']

	sumTop2 += killsPerteam[x]['TOP']

	sumTop3 += visionPerteam[x]['TOP']

	

	sumMid1 += damagePerteam[x]['MID']

	sumMid2 += killsPerteam[x]['MID']

	sumMid3 += visionPerteam[x]['MID']

	

	sumJungle1 += damagePerteam[x]['NONE']

	sumJungle2 += killsPerteam[x]['NONE']

	sumJungle3 += visionPerteam[x]['NONE']

#Create a dataframe for the objects to be graphed

damage = pd.DataFrame([sumSupport1, sumAdc1, sumTop1, sumJungle1, sumMid1], columns = ['Support', 'ADC', 'TOP', 'Jungle', 'Mid'])

kda = pd.DataFrame([sumSupport2, sumAdc2, sumTop2, sumJungle2, sumMid2],  columns = ['Support', 'ADC', 'TOP', 'Jungle', 'Mid'])

vision = pd.DataFrame([sumSupport3, sumAdc3, sumTop3, sumJungle3, sumMid3],  columns = ['Support', 'ADC', 'TOP', 'Jungle', 'Mid'])

#Damage diagram

plt.figure(1)

plt.subplot(211)

damage.plot.bar(title = 'Damage')

#KDA diagram

plt.subplot(212)

kda.plot.bar(title = 'KDA')

#Vision diagram

plt.subplot(213)

vision.plot.bar(title = ' Vision')



plt.show()

#for index, rows in particpantsdf.iterrows():

	#champsDict[rows['championid']] += 1

#pd.DataFrame([list(champsDict.values())], columns = champs['name']).plot.bar()

#plt.show()



#How often bot lane won the game for the team 

#I.e. kills/asist/damage per game per win

#I.e. average amount of damage per role per win/loss