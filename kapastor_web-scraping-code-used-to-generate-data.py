import pandas as pd 

import urllib.request

from bs4 import BeautifulSoup

import datetime



city_name = 'toronto'

month = [m for m in range(1,13)]

year = [y for y in range(2010,2021)]

dataset = []

for y in year:

    for m in month:

        url = 'https://www.timeanddate.com/sun/canada/'+city_name+'?month='+str(m)+'&year='+str(y)

        with urllib.request.urlopen(url) as response:

            html = response.read()

            soup = BeautifulSoup(html)

            day = soup.findAll("tr", {"title": "Click to expand for more details"})

            for d in day:

                # Format the month and day to have leading zeros:

                month_date = '{:02}'.format(m)

                day_date = '{:02}'.format(int(d['data-day']))

                final_date = str(y)+'-'+str(month_date)+'-'+str(day_date)

                

                ele = d.findAll("td")

                sundown = ele[1].findAll(text=True, recursive=False)[0]

                sundown_hour = '{:02}'.format(int(sundown.split(':')[0]));sundown_hour = int(sundown_hour)+12

                sundown_min = '{:02}'.format(int(sundown.split(':')[1].split()[0]))

                sundown_time = str(sundown_hour)+':'+str(sundown_min)

                

                sunup = ele[0].findAll(text=True, recursive=False)[0]

                sunup_hour = '{:02}'.format(int(sunup.split(':')[0])) 

                sunup_min = '{:02}'.format(int(sunup.split(':')[1].split()[0]))

                sunup_time = str(sunup_hour)+':'+str(sunup_min)

                

                dataset.append([city_name,final_date,sunup_time,sundown_time])

                

df = pd.DataFrame(dataset,columns=['City','Date','Sunup','Sundown'])

df.to_csv('Sunup_Sundown_Data.csv',index=False)            

            

            

            
