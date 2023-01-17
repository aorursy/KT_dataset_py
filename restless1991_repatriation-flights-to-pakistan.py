import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
import numpy as np
from datetime import datetime
website = 'http://covid.gov.pk/intl_travellers/flight_info'
cxn = sqlite3.connect("flights.db")

page = requests.get(website)
if page.status_code == 200:
    soup = soup = BeautifulSoup(page.content, 'html.parser')
else:
    print("Error Page status:", page.status_code)
scheduled_flights = soup.find_all('table')[1]
complted_flights = soup.find_all('table')[2]

data = []
for idx,tr in enumerate(scheduled_flights.find('tbody').find_all('tr')):
    if idx == 0:
        continue
    row = [td.text.replace('\n','').replace('\t','').replace('\r','').replace(' ','') for td in tr.find_all('td')]
    row.append('open' if 'open' in row[1] else 'close' if 'close' in row[1] else np.nan)
    data.append(row)

cols = ['sr_no', 'from_place', 'departure_airport', 'to_place', 'arrival_date', 'passengers', 'airline', 'status']
schedule = pd.DataFrame(data, columns=cols)
# date format setting for the database
schedule['arrival_date'] = pd.to_datetime(schedule['arrival_date']).dt.date
schedule.to_sql('schedule', cxn, if_exists='replace', index=False)

data = []
for idx,tr in enumerate(complted_flights.find('tbody').find_all('tr')):
    if idx == 0:
        continue
    row = [td.text.replace('\n','').replace('\t','').replace('\r','').replace(' ','') for td in tr.find_all('td')]
    data.append(row)

cols = ['sr_no', 'from_place', 'to_place', '_date', 'passengers', 'airline',]
completed = pd.DataFrame(data, columns=cols)
# date format setting for the database
completed['_date'] = pd.to_datetime(completed['_date']).dt.date
completed.to_sql('completed', cxn, if_exists='replace', index=False)


pak_loc = 'https://en.wikipedia.org/wiki/List_of_cities_in_Pakistan'
page = requests.get(pak_loc)
if page.status_code == 200:
    soup = soup = BeautifulSoup(page.content, 'html.parser')
else:
    print("Error Page status:", page.status_code)
data = []
for x in soup.find_all('table', {'class': 'wikitable'}):
    provice = ''
    _type = ''
    if 'Balochistan' in x.find('th').text:
        provice = 'Balochistan'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Khyber' in x.find('th').text:
        provice = 'Khyber'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Punjab' in x.find('th').text:
        provice = 'Punjab'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Sindh' in x.find('th').text:
        provice = 'Sindh'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Kashmir' in x.find('th').text:
        provice = 'Kashmir'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    elif 'Capital' in x.find('th').text:
        provice = 'Punjab'
        _type = 'municipalities'
    elif 'Gilgit' in x.find('th').text:
        provice = 'Gilgit'
        if 'municipalities' in x.find('th').text:
            _type = 'municipalities'
        elif 'districts' in x.find('th').text:
            _type = 'districts'
    
    population = x.find('tbody').find_all('tr')[3]
    names = x.find('tbody').find_all('tr')[2]
    for name, pop in zip(names.find_all('td'), population.find_all('td')):
        data.append({
            'provice': provice,
            '_type': _type,
            'city': name.text[:-6],
            'population': pop.text.replace('\n','').replace(',','')
        })

pak = pd.DataFrame(data)
pak.to_sql('pakistan', cxn, if_exists='replace', index=False)
def sql_fetch(con):

    cursorObj = con.cursor()

    cursorObj.execute('SELECT name from sqlite_master where type= "table"')

    return cursorObj.fetchall()

for x in sql_fetch(cxn):
    filename = f"{x[0]}.csv"
    df = pd.read_sql(f"select * from {x[0]}", cxn)
    df.to_csv(filename, index=False)
    del df
    print(filename, 'created')