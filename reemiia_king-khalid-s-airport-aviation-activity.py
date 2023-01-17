import requests

from bs4 import BeautifulSoup

import pandas as pd
res = []

page = 0

r = requests.get("https://uk.flightaware.com/live/airport/OERK/arrivals?;offset={};order=actualarrivaltime;sort=DESC".format(page))

html = r.text

soup = BeautifulSoup(html, "html.parser")

table = soup.find('table', attrs={"class": "prettyTable"})

table_rows = table.find_all('tr')





while len(table_rows) > 3:

    for tr in table_rows:

        td = tr.find_all('td')

        row = [tr.text.strip() for tr in td if tr.text.strip()]

        if row:

            res.append(row)

    page += 20



df_arrivals = pd.DataFrame(res, columns=['Ident', 'Type', 'Origin', 'Departure', 'Arrival'])

df_arrivals
r = requests.get("https://uk.flightaware.com/live/airport/OERK/departures?;offset={};order=actualdeparturetime;sort=DESC".format(page))

html = r.text

soup = BeautifulSoup(html, "html.parser")

table = soup.find('table', attrs={"class": "prettyTable"})

table_rows = table.find_all('tr')



res = []

page = 0

while len(table_rows) > 3:   

    for tr in table_rows:

        td = tr.find_all('td')

        row = [tr.text.strip() for tr in td if tr.text.strip()]

        if row:

            res.append(row)

    page += 20



df_departures = pd.DataFrame(res, columns=['Ident', 'Type', 'Destination','Departure', 'Estimated Arrival Time', 'Arrival'])

df_departures