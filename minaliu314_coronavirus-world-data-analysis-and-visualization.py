# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Import Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
# read files

totalCases = pd.read_csv("../input/covid19-dataset/total_cases.csv")

totalDeath = pd.read_csv("../input/covid19-dataset/total_deaths.csv")

newDeath = pd.read_csv("../input/covid19-dataset/new_deaths.csv")

newCases = pd.read_csv("../input/covid19-dataset/new_cases.csv")
world_confirmed_cases = totalCases['World'].iloc[-1]

china_confirmed = totalCases['China'].iloc[-1]

outside_china_confirmed = totalCases['World'].iloc[-1] - totalCases['China'].iloc[-1]
endDate = totalCases['date'].max()

print(f'{endDate}')

print()

print('Outside China:   {} cases'.format(outside_china_confirmed))

print('China:\t\t {} cases'.format(china_confirmed))

print('Total:\t\t {} cases'.format(world_confirmed_cases))
names = ['China','Outside China']

values = [china_confirmed, outside_china_confirmed]

plt.figure(figsize=(16, 9))

plt.barh(names,values,color='darkred')

plt.title('Coronavirus Confirmed Cases', size=20)

plt.xticks(size=20)

plt.yticks(size=20)

plt.grid()

plt.show()
df1 = totalCases.tail(1)

df1
df2 = df1.drop(df1.columns[[0,1]], axis = 1)

df2
df2.fillna(0)

np.count_nonzero(df2)
df2.idxmax(axis = 1)
df = pd.read_csv("../input/covid19-dataset/full_data.csv",

                 usecols=['date','location','total_cases'])

df.drop(df[df['location'] == 'World' ].index, inplace=True)

end = df['date'].max()

dff = (df[df['date'].eq(end)]

       .sort_values(by='total_cases', ascending=False).head(10))

dff
plt.figure(figsize = (14,14))

labels = dff['location'].tolist()

values = dff['total_cases'].tolist()

#Explode = [0, 0, 0.1, 0, 0, 0, 0 , 0 , 0] 

colors=['lightsalmon','pink','lightskyblue',

        'lightyellow','grey','plum','peachpuff',

        'lightgreen','aliceblue','slateblue']

plt.title('COVID-19 Confirmed Cases per Country', size=20)

plt.pie(values,labels = labels, autopct="%.1f%%",colors=colors)#, explode = Explode)

plt.show()
print('Top 10 countries have the most confirmed cases: {}'. format(labels))
plt.figure(figsize=(25, 10))

plt.plot(totalCases['date'], totalCases['United States'],color='red')

plt.plot(totalCases['date'], totalCases['Brazil'],color='blue')

plt.plot(totalCases['date'], totalCases['Russia'],color='orange')

plt.plot(totalCases['date'], totalCases['United Kingdom'],color='cyan')

plt.plot(totalCases['date'], totalCases['Italy'],color='slategray')

plt.plot(totalCases['date'], totalCases['Germany'],color='black')

plt.plot(totalCases['date'], totalCases['Turkey'],color='brown')

plt.plot(totalCases['date'], totalCases['India'],color='purple')

plt.plot(totalCases['date'], totalCases['France'],color='green')

plt.plot(totalCases['date'], totalCases['Iran'],color='navy')

plt.title('Total Coronavirus Cases', size=30)

plt.xlabel('Days Since 12/31/2020', size=30)

plt.ylabel('Amount', size=30)

plt.legend(['United States', 'Spain', 'Italy',

            'Germany', 'United Kingdom',

            'France', 'Turkey', 'Iran',

            'China', 'Russia'], prop={'size': 20})

plt.xticks(rotation = 90,size=9)

plt.yticks(size=20)

plt.grid()

plt.show()
totalCases[totalCases['date'] == '2020-03-16'].index.values
dff = totalCases.iloc[76:,:]
plt.figure(figsize=(25, 10))

plt.plot(dff['date'], dff['United States'],color='red')

plt.plot(dff['date'], dff['Brazil'],color='blue')

plt.plot(dff['date'], dff['Russia'],color='orange')

plt.plot(dff['date'], dff['United Kingdom'],color='cyan')

plt.plot(dff['date'], dff['Italy'],color='slategray')

plt.plot(dff['date'], dff['Germany'],color='black')

plt.plot(dff['date'], dff['Turkey'],color='brown')

plt.plot(dff['date'], dff['India'],color='purple')

plt.plot(dff['date'], dff['France'],color='green')

plt.plot(dff['date'], dff['Iran'],color='navy')

plt.title('Total Coronavirus Cases', size=30)

plt.xlabel('Days Since 03/16/2020', size=30)

plt.ylabel('Amount', size=30)

plt.legend(['United States', 'Spain', 'Italy',

            'Germany', 'United Kingdom',

            'France', 'Turkey', 'Iran',

            'China', 'Russia'], prop={'size': 20})

plt.xticks(rotation = 90,size=13)

plt.yticks(size=20)

plt.grid()

plt.show()
dff_death = totalDeath.iloc[76:,:]
plt.figure(figsize=(25, 10))

plt.plot(dff_death['date'], dff_death['United States'],color='red')

plt.plot(dff_death['date'], dff_death['Brazil'],color='blue')

plt.plot(dff_death['date'], dff_death['Russia'],color='orange')

plt.plot(dff_death['date'], dff_death['United Kingdom'],color='cyan')

plt.plot(dff_death['date'], dff_death['Italy'],color='slategray')

plt.plot(dff_death['date'], dff_death['Germany'],color='black')

plt.plot(dff_death['date'], dff_death['Turkey'],color='brown')

plt.plot(dff_death['date'], dff_death['India'],color='purple')

plt.plot(dff_death['date'], dff_death['France'],color='green')

plt.plot(dff_death['date'], dff_death['Iran'],color='navy')

plt.title('Total Coronavirus Death Cases', size=30)

plt.xlabel('Days Since 03/16/2020', size=30)

plt.ylabel('Amount', size=30)

plt.legend(['United States', 'Spain', 'Italy',

            'Germany', 'United Kingdom',

            'France', 'Turkey', 'Iran',

            'China', 'Russia'], prop={'size': 20})

plt.xticks(rotation = 90,size=13)

plt.yticks(size=20)

plt.grid()

plt.show()
us20 = totalCases.tail(20)
plt.figure(figsize=(10, 7))

plt.bar("date", "United States", data = us20, color = "darkred") 

plt.plot(us20['date'], us20['United States'],color = 'black')

plt.xlabel("Dates", size=20) 

plt.ylabel("Amount", size=20)

plt.title("Last 20 Days Confirmed Cases in US", size=20)

plt.xticks(rotation = 70,size=13)

plt.yticks(size=13)

plt.grid()

plt.show()
usDeath20 = totalDeath.tail(20)

plt.figure(figsize=(10, 7))

plt.bar("date", "United States", data = usDeath20, color = "darkred") 

plt.plot(usDeath20['date'], usDeath20['United States'],color = 'black')

plt.xlabel("Dates", size=20) 

plt.ylabel("Amount", size=20)

plt.title("Last 20 Days Death Cases in US", size=20)

plt.xticks(rotation = 70,size=13)

plt.yticks(size=13)

plt.grid()

plt.show()
mortalityRate = (totalDeath['United States'].iloc[-1]) / (totalCases['United States'].iloc[-1])

mortalityRate
newCases20 = newCases.tail(20)

newDeath20 = newDeath.tail(20)
plt.figure(figsize=(16, 9))

plt.plot(usDeath20['date'], usDeath20['United States'],color='red')

plt.plot(us20['date'], us20['United States'],color='green')

plt.plot(newDeath20['date'],newDeath20['United States'])

plt.plot(newCases20['date'],newCases20['United States'])

plt.legend(['Total Death', 'Total Confirmed','New Death','New Confirmed'], prop={'size': 15})

plt.xlabel("Dates", size=20) 

plt.ylabel("Amount", size=20)

plt.title("Late 20 Days Coronavirus Cases in US", size=20)

plt.xticks(rotation = 70,size=13)

plt.yticks(size=13)

plt.grid()

plt.show()
plt.figure(figsize=(16, 9))



newCases10 = newCases.tail(10)

newDeath10 = newDeath.tail(10)



plt.plot(newDeath10['date'],newDeath10['United States'],color = 'r', marker = 'o')

plt.plot(newCases10['date'],newCases10['United States'],color = 'g', marker = 'o')

plt.legend(['New Death','New Confirmed'], prop={'size': 15})

plt.title('New Confirmed VS. New Death in US (last 10 days)',size=25)

plt.ylabel('Amount', size=20)

plt.xlabel('Dates', size=20)

plt.grid()

plt.show()
plt.figure(figsize=(16, 9))



newCases10 = newCases.tail(10)

newDeath10 = newDeath.tail(10)

width = 0.5



p1 = plt.bar(newCases10['date'], newCases10['United States'], width)

p2 = plt.bar(newDeath10['date'], newDeath10['United States'], width)



plt.title('New Confirmed VS. New Death in US (last 10 days)',size=25)

plt.ylabel('Amount', size=20)

plt.xlabel('Dates', size=20)

plt.legend((p1[0], p2[0]), ('New Confirmed', 'New Death'))

plt.grid()

plt.show()