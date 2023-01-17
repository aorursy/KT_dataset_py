from urllib.request import urlopen as uReq

from bs4 import BeautifulSoup as soup

import re



filename = 'Cities.csv'

headers  = 'City,Latitude,Longitude\n'

f = open(filename, "w")

f.write(headers)

for j in range(1,9):    

    page_url  = f'https://www.latlong.net/category/cities-102-15-{j}.html'

    uClient   = uReq(page_url)

    page_soup = soup(uClient.read(), "html.parser")

    uClient.close()

    rows = page_soup.findAll('tr')

    rows = rows[1:]

    for row in rows:

        cell      = row.findAll('td')

        City      = cell[0].text

        Latitude  = cell[1].text

        Longitude = cell[2].text

        f.write(re.sub(r',\s*', '|',City) + ',' + Latitude + ',' + Longitude + '\n')

f.close()
!cat Cities.csv
!curl https://www.latlong.net/category/cities-102-15-1.html