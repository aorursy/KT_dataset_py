import pandas as pd

from json import load

import urllib.request, json 

from pandas.io.json import json_normalize

import seaborn as sns

import pylab as plt

%matplotlib inline
!mkdir outputs
cur_url = "https://zenodo.org/api/records/?sort=mostrecent&type=dataset&access_right=open&size=1000"

counter = 0

while True:

    print(cur_url)

    with urllib.request.urlopen(cur_url) as url:

        data = json.loads(url.read().decode())

        with open('outputs/%02d.json'%(counter), 'w') as outfile:

            json.dump(data, outfile, sort_keys = True, indent = 4,

               ensure_ascii = False)

            counter += 1

    if 'next' in data['links']:

        next_url = data['links']['next']

        next_page = int(next_url.split('page=')[1].split('&')[0])

        if next_page == 10:

            last_date = data['hits']['hits'][-1]['created'].split('+')[0]

            next_url = 'https://zenodo.org/api/records/?sort=mostrecent&q=created%3A%5B%2A+TO+' + last_date + '%5D&page=1&type=dataset&access_right=open&size=1000'

        cur_url = next_url

    else:

        break