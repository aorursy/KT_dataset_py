# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#lets import required libraries



import numpy as np #for data preperation to make dataset

import pandas as pd  #for dataset making



#for webscraping

from urllib.request import urlopen  #to handle url

import urllib

from bs4 import BeautifulSoup  #popular webscraping lib
def get_webpage(link):

    page=urlopen(link)  #opening url

    soup=BeautifulSoup(page, 'html.parser') #collecting whole webpage as soup content

    return soup
#call the function with link

content=get_webpage('https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population')



#extract tables from content

tables=content.find_all('table')



for table in tables:

    #lets print what looks like our data using pretify lib

    print(table.prettify())

#lets extract the table with class "wikitable sortable"

#table=content.find('table', {'class': 'wikitable sortable plainrowheaders jquery-tablesorter'})

#extract rows from the table

#rows=table.find_all('tr') #tr means table row..its html tag for tables



table = content.find('table', {'class': 'wikitable sortable plainrowheaders jquery-tablesorter'})

rows = table.find_all('tr')



# List of all links

for row in rows:

    cells = row.find_all('td')

    if len(cells) > 1:

        country_link = cells[1].find('a', href=True)

        #print(country_link.get('href'))
def getallinfo(url):

    try:

        country_page = get_webpage('https://en.wikipedia.org' + url)

        table = country_page.find('table', {'class': 'infobox geography vcard'})

        additional_details = []

        read_content = False

        for tr in table.find_all('tr'):

            if (tr.get('class') == ['mergedtoprow'] and not read_content):

                link = tr.find('a')

                if (link and (link.get_text().strip() == 'Area' or

                   (link.get_text().strip() == 'GDP' and tr.find('span').get_text().strip() == '(nominal)'))):

                    read_content = True

                if (link and (link.get_text().strip() == 'Population')):

                    read_content = False

            elif ((tr.get('class') == ['mergedrow'] or tr.get('class') == ['mergedbottomrow']) and read_content):

                additional_details.append(tr.find('td').get_text().strip('\n')) 

                if (tr.find('div').get_text().strip() != '???\xa0Total area' and

                   tr.find('div').get_text().strip() != '???\xa0Total'):

                    read_content = False

        return additional_details

    except Exception as error:

        print('Error occured: {}'.format(error))

        return []
data_content = []

for row in rows:

    cells = row.find_all('td')

    if len(cells) > 1:

        print(cells[1].get_text())

        country_link = cells[1].find('a')

        country_info = [cell.text.strip('\n') for cell in cells]

        additional_details = getallinfo(country_link)

        if (len(additional_details) == 4):

            country_info += additional_details

            data_content.append(country_info)



dataset = pd.DataFrame(data_content)
#give name to your columns

headers = rows[0].find_all('th')

headers = [header.get_text().strip('\n') for header in headers]

headers += ['Total Area', 'Percentage Water', 'Total Nominal GDP', 'Per Capita GDP']

dataset.columns = headers



drop_columns = ['Rank', 'Date', 'Source']

dataset.drop(drop_columns, axis = 1, inplace = True)

dataset.sample(3)



dataset.to_csv("Dataset.csv", index = False)