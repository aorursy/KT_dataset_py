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
os.chdir('/kaggle/input/predicting-energy-rating-from-raw-data')

train_data = pd.read_csv('train_rating_eu.csv')

test_data = pd.read_csv('test_rating_eu.csv')



train_data = train_data.drop(['building_id', 'site_id', 'Unnamed: 0'], axis=1)

test_data = test_data.drop(['building_id', 'site_id', 'Unnamed: 0'], axis=1)
import requests



URL1 = 'https://platform.carbonculture.net/communities/ucl/30/apps/assets/list/place/'

URL2 = 'https://platform.carbonculture.net/places/119-torrington-place/1155/'

URL3 = 'https://platform.carbonculture.net/about/'

page1 = requests.get(URL1)

page2 = requests.get(URL2)

page3 = requests.get(URL3)
f = open('/kaggle/working/output1', 'wb')

f.write(page1.content)

f = open('/kaggle/working/output2', 'wb')

f.write(page2.content)

f = open('/kaggle/working/output3', 'wb')

f.write(page3.content)
from bs4 import BeautifulSoup



soup1 = BeautifulSoup(page1.content, 'html.parser')

soup2 = BeautifulSoup(page2.content, 'html.parser')

soup3 = BeautifulSoup(page3.content, 'html.parser')
places_elems = soup3.find_all('a', href=True)

places_elems
i = 0

places = []



for place in places_elems:

    if i > 10 and i < 23:

        if place.text: 

            places.append(place['href'])

    i = i + 1

    

places
urls=[]

base = 'https://platform.carbonculture.net'

end = 'apps/assets/list/place/'



for p in places:

    urls.append((base + p + end))

    

urls
soups=[]



for u in urls:

    p = requests.get(u)

    soups.append(BeautifulSoup(p.content, 'html.parser'))
titles = []

titles_whole = []



for soup in soups:

    url_elems = soup.find_all(href=True)

    for elem in url_elems:

        if elem.text: 

            if elem['href'].find('places') == 1:

                titles.append(elem['href'])

                

for t in titles:

    titles_whole.append(base + t)
years = []

floors = []

heating = []

occupants = []

i = 0



for w in titles_whole:

    test = requests.get(w)

    soup_test = BeautifulSoup(test.content, 'html.parser')

    test_elems = soup_test.find_all('li', class_='assets-meta__list-item')

    if len(test_elems) == 0:

        print(w)

        years.append('-')

        floors.append(-1)

        heating.append('-')

        occupants.append(-1)

    for elem in test_elems:

        a = str(elem.find('span'))[6:-7]

        if i == 0:

            years.append(a)

        elif i == 1:

            floors.append(a)

        elif i == 3:

            heating.append(a)

        elif i == 4:

            occupants.append(a)

            i = -1

        i = i + 1
i = 0

j = 0

title = []

consumption = []

floor_area = []

consumption_area = []

rating = []



for soup in soups:

    url = titles_whole[i]

    page = requests.get(url)

    soup_title = BeautifulSoup(page.content, 'html.parser')

    table_elems = soup.find_all('td')

    for elem in table_elems:

        if i != 5 and i != 6 and i != 7:

            elem = str(elem)

            elem = elem[4:-5]

            if i == 0:

                elem = elem[-11:-8]

                if elem != 'N/A':

                    elem = elem[2:]

                rating.append(elem)

            if i == 1:

                title.append(elem)

            elif i == 2:

                elem = elem[:-9]

                if len(elem) == 1:

                    elem = '-1'

                elem = elem.replace(',', '')

                consumption.append(elem)

            elif i == 3:

                elem = elem[:-9]

                if len(elem) == 1:

                    elem = '-1'

                elem = elem.replace(',', '')

                floor_area.append(elem)

            elif i == 4:

                elem = elem[:-9]

                if len(elem) == 1:

                    elem = '-1'

                elem = elem.replace(',', '')

                consumption_area.append(elem)

            j = j + 1

        if i == 7:

            i = -1

        i = i + 1

i = 0

null = occupants[320]



for o in occupants:

    if str(o) == null:

        occupants[i] = -1

    i = i + 1



i = 0

null = floors[33]

for f in floors:

    if str(f) == null:

        print('here!')

        floors[i] = -1

    elif len(str(f)) > 3:

        seq_type= type(f)

        f = seq_type().join(filter(seq_type.isdigit, f))

        floors[i] = f

    i = i + 1
consumption = list(map(float, consumption))

consumption_area = list(map(float, consumption_area))

floor_area = list(map(float, floor_area))

occupants = list(map(float, occupants))

floors = list(map(float, floors))
columns = ['sqm', 'building', 'energy consumption', 'energy consumption per area', 'year built', 'floors', 'no. occupants', 'main heating type', 'rating2']

df = pd.DataFrame(columns=columns)
df['sqm'] = floor_area

df['building'] = title

df['energy consumption'] = consumption

df['energy consumption per area'] = consumption_area

df['year built'] = years

df['floors'] = floors

df['no. occupants'] = occupants

df['main heating type'] = heating

df['rating2'] = rating
df['sqm'] = df['sqm'].replace(-1, np.nan)

df['energy consumption'] = df['energy consumption'].replace(-1, np.nan)

df['energy consumption per area'] = df['energy consumption per area'].replace(-1, np.nan)

df['rating2'] = df['rating2'].replace('N/A', np.nan)

df['floors'] = df['floors'].replace(-1, np.nan)

df['no. occupants'] = df['no. occupants'].replace(-1, np.nan)

df['year built'] = df['year built'].replace('-', np.nan)

df['main heating type'] = df['main heating type'].replace('-', np.nan)

df
df.info()
train1 = df[df['rating2'].notna()]

train1.info()
test1 = df[df['rating2'].isnull()]

test1 = test1.drop('rating2', axis=1)

test1.info()
merge1 = pd.merge(left=train_data, right=train1, how='outer', left_on='sqm', right_on='sqm')

merge1.info()
merge2 = pd.merge(left=test_data, right=test1, how='outer', left_on='sqm', right_on='sqm')

merge2.info()
sqm1 = train_data['sqm']

sqm2 = train1['sqm']

year1 = train_data['yearbuilt']

year2 = train1['year built']



train_intersection = list(set(sqm1) & set(sqm2))

year_intersection = list(set(year1) & set(year2))





print("There are ", len(train_intersection), " sqm matches!")

print("There are ", len(year_intersection), " year matches!")
sqm1 = test_data['sqm']

sqm2 = test1['sqm']

year1 = test_data['yearbuilt']

year2 = test1['year built']



train_intersection = list(set(sqm1) & set(sqm2))

year_intersection = list(set(year1) & set(year2))





print("There are ", len(train_intersection), " sqm matches!")

print("There are ", len(year_intersection), " year matches!")