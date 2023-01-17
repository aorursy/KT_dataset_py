import numpy as np 

import pandas as pd

import requests 



from bs4 import BeautifulSoup
r = requests.get('https://en.wikipedia.org/wiki/List_of_cities_by_average_temperature')

soup = BeautifulSoup(r.text,'html.parser')
r.text[0:500]
continent = soup.find_all('span', attrs={'class':'mw-headline'})

continent
results = soup.find_all('td')

len(results)
temperature = list()

count = 0

c = 0

temp = list()

for idx in range(len(results)):

    try:

        if count == 0:

            country = results[idx].find('a')['title']

            count += 1

            temp.append(country)

        else: 

            if idx == 3457: # There is dirty data in here -_- monaco didn't have a city -_-

                city = results[idx].contents[0] 

                count = 0

                temp.append(city)

                

            else:

                city = results[idx].find('a')['title']

                count = 0

                temp.append(city)

    except:

        temp.append(results[idx].contents[0])

        c+=1

    if c == 14:

        temperature.append(tuple(temp))

        temp = list()

        c = 0
columns = ['Country','City','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Avg_Year', 'other']
world_temp_2020 = pd.DataFrame(temperature,columns = columns)

world_temp_2020.drop(columns = 'other',inplace=True)
world_temp_2020.tail()
num_col = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul',

       'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Avg_Year']
Month = pd.DataFrame()

for col in num_col:

    month = list()

    for i in range(len(world_temp_2020)):

        try:

            month.append(float(world_temp_2020.iloc[i,:][col]))

        except:

            month.append(float(world_temp_2020.iloc[i,:][col][1:])*-1)

    Month[col] = month

for col in num_col:

    world_temp_2020[col] = Month[col]
world_temp_2020.head()
Continent = list()
for i in range(len(world_temp_2020)):

    if i <= 108:

        Continent.append('Africa')

    elif i > 108 and i<= 183:

        Continent.append('Asia')

    elif i > 183 and i <= 245:

        Continent.append('Europe')

    elif i > 245 and i <= 343:

        Continent.append('North America')

    elif i > 343 and i<=369:

        Continent.append('Oceania')

    elif i > 369:

        Continent.append('South America')
world_temp_2020['Continent'] = Continent
world_temp_2020.head()
world_temp_2020.tail()
world_temp_2020.info()
world_temp_2020.to_csv('Avg_World_Temp_2020.csv')