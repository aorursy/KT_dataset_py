#The script will extract data from the https://kenpom.com/ for the years for which the data is available.

#No data behind paywall can be accessed by this

from bs4 import BeautifulSoup

import numpy as np

import pandas as pd

import requests

import re
urls = np.arange(2002, 2021, 1)
for url in urls:

    print(f"http://kenpom.com/index.php?y={url}")
table = {'year' : [],

        'rank': [],

        'team': [],

        'W-L': [],

        'AdjEM':[],

        'AdjO':[],

        'AdjD':[],

        'AdjT':[],

        'Luck':[],

        'SoSAdjEM':[],

        'SoSOppO':[],

        'SoSOppD':[]}
from tqdm import tqdm

for url in tqdm(urls):

    r = requests.get(f"http://kenpom.com/index.php?y={url}")

    soup = BeautifulSoup(r.text, "lxml")

    for elem in soup.find_all('tbody'):

        for row in elem.find_all('tr'):

            try:            

                table['rank'].append(int(row.find('td', class_='hard_left').text))

                table['year'].append(url)

                table['team'].append(str(row.find('td', class_='next_left').text))

                table['W-L'].append(str(row.find('td', class_='wl').text))

                table['AdjEM'].append(float(row.find('td', class_=None).text))

                for i, row_elem in enumerate(row.find_all('td', class_='td-left divide')):

                    if i==0:

                        table['AdjO'].append(float(row_elem.text))

                    elif i==1:

                        table['AdjT'].append(float(row_elem.text))

                    elif i==2:

                        table['Luck'].append(float(row_elem.text))

                    elif i==3:

                        table['SoSAdjEM'].append(row_elem.text)

                for i, row_elem in enumerate(row.find_all('td', class_='td-left')):

                    if i== 1:

                        table['AdjD'].append(float(row_elem.text))

                    elif i==5:

                        table['SoSOppO'].append(float(row_elem.text))

                    elif i==6:

                        table['SoSOppD'].append(float(row_elem.text))

            except:

                pass
ken_pom_data = pd.DataFrame(table)
ken_pom_data['team'] = ken_pom_data.team.apply(lambda x: re.sub('[^a-zA-Z]+', '', x.lower()))
#ken_pom_data.to_csv('Data/ken_pom.csv', index =False)