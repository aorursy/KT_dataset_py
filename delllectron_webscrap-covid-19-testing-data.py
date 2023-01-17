import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from bs4 import BeautifulSoup

import requests

import json

from tqdm import tqdm

from datetime import datetime

import os

import string

import random
#set the target url

base_url = 'https://en.wikipedia.org/wiki/COVID-19_testing'
res = requests.get(base_url)

soup = BeautifulSoup(res.text, 'html.parser')

#find all table

all_table = soup.find_all('table')
#get the index of target table

for i in range(len(all_table)):

    headings = [cell.get_text().strip() for cell in all_table[i].find('tr').find_all('th')]

    try:

        if 'Country' in headings:

            idx = i

            print('='*30)

            print('Target found!!!')

            print('='*30, '\n')

            break

    except:

        continue
#get the table body

all_rows = [cell for cell in all_table[idx].find('tbody').find_all('tr')]

len(all_rows)
#get the rows

data_row = []

for row in tqdm(all_rows[1:128], desc = 'Pulling the data'):

    cell_value = row.find_all('td')

    data_set = [i.get_text().strip() for i in cell_value]

    data_row.append(data_set)
#get the countries

data_country = []

for row in all_rows[1:128]:

    cell_count = row.find_all('th')

    th = [i.get_text().strip() for i in cell_count]

    data_country.append(th)



#get the columns

cols = [i.get_text().strip() for i in all_rows[0].find_all('th')]
#reshaping to 1d array

data_country = np.array(data_country).reshape(-1,)

#create a data frame

df = pd.DataFrame(index=data_country, data =data_row).reset_index()

df.columns = cols

#drop the ref feature

df.drop('Ref.', axis=1, inplace=True)
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):

    return ''.join(random.choice(chars) for _ in range(size))



date_now = str(datetime.now())[:10]

folder_name = id_generator(10)

os.mkdir(folder_name)
df.to_csv(folder_name + '/' + 'covid-19_testing_data.csv')

print('The data is ready.\n', 'Folder name: ', folder_name)
df.head()