# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib
from urllib import request
import re

Wiki_url = ['https://en.wikipedia.org/wiki/Sverdlovsk_Oblast',
 'https://en.wikipedia.org/wiki/Samara_Oblast',
 'https://en.wikipedia.org/wiki/Rostov_Oblast',
 'https://en.wikipedia.org/wiki/Tatarstan',
 'https://en.wikipedia.org/wiki/Volgograd_Oblast',
 'https://en.wikipedia.org/wiki/Nizhny Novgorod_Oblast',
 'https://en.wikipedia.org/wiki/Perm_Krai',
 'https://en.wikipedia.org/wiki/Orenburg_Oblast',
 'https://en.wikipedia.org/wiki/Khanty-Mansi Autonomous_Okrug',
 'https://en.wikipedia.org/wiki/Bashkortostan',
 'https://en.wikipedia.org/wiki/Krasnodar_Krai',
 'https://en.wikipedia.org/wiki/Novosibirsk_Oblast',
 'https://en.wikipedia.org/wiki/Omsk_Oblast',
 'https://en.wikipedia.org/wiki/Chelyabinsk_Oblast',
 'https://en.wikipedia.org/wiki/Voronezh_Oblast',
 'https://en.wikipedia.org/wiki/Kemerovo_Oblast',
 'https://en.wikipedia.org/wiki/Saratov_Oblast',
 'https://en.wikipedia.org/wiki/Vladimir_Oblast',
 'https://en.wikipedia.org/wiki/Krasnoyarsk_Krai',
 'https://en.wikipedia.org/wiki/Belgorod_Oblast',
 'https://en.wikipedia.org/wiki/Yaroslavl_Oblast',
 'https://en.wikipedia.org/wiki/Kaliningrad_Oblast',
 'https://en.wikipedia.org/wiki/Tyumen_Oblast',
 'https://en.wikipedia.org/wiki/Udmurtia',
 'https://en.wikipedia.org/wiki/Altai_Krai',
 'https://en.wikipedia.org/wiki/Irkutsk_Oblast',
 'https://en.wikipedia.org/wiki/Stavropol_Krai',
 'https://en.wikipedia.org/wiki/Tula_Oblast']

want=["Density", "Time zone", "Rural", "Urban", "Total"]
dictionary = {}

def Scrape_info(url):
    wiki_url = url
    udr = {'User-Agent': 'Mozilla/5.0'}
    try:
        page = urllib.request.urlopen(wiki_url).read()
        soup = BeautifulSoup(page, "html.parser")
        table = soup.find('table', class_='infobox geography vcard')
    except:
        table = None
    
    country = url.split("/")[-1]
    
    if table is not None:
        result = {}
    
        exceptional_row_count = 0
        for tr in table.find_all('tr'):
            if tr.find('th'):
                if tr.find("td"):
                    result[tr.find('th').text] = tr.find('td').text
    
 
        dictionary[country]={}
        for i in result.keys():
            for j in want:
                if j in i:
                    dictionary[country][j] = result[i]
    elif  table is None:
        dictionary[country] = {}
        for i in want:
            dictionary[country][i] = {np.nan}
                
for i in Wiki_url:
    Scrape_info(i)

reg = list(dictionary.keys())
vals = [dictionary[reg[i]] for i in range(len(reg)) ]

regional_data = pd.DataFrame(vals, index = reg)

dens = np.array([float(i.split()[0].split("/")[0]) for i in regional_data["Density"].values])
rul = np.array([float(i.split("%")[0]) for i in regional_data["Rural"]])
tim = np.array([i.split()[0] for i in regional_data["Time zone"]])
tot = np.array([int("".join(i.split()[0].split("[")[0].split(","))) for i in regional_data["Total"].values])
urb = np.array([float(i.split("%")[0]) for i in regional_data["Urban"]])

regional_data["Density_of_region(km2)"] = dens
regional_data["Rural_%"] = rul
regional_data["Time_zone"] = tim
regional_data["Total_population"] = tot
regional_data["Urban%"] = urb

del regional_data["Density"]
del regional_data["Rural"]
del regional_data["Time zone"]
del regional_data["Total"]
del regional_data["Urban"]

regional_data.to_csv("/Users/HongSukhyun/Desktop/Python/Kaggle competition/Avito/regional.csv", encoding = "utf-8")
