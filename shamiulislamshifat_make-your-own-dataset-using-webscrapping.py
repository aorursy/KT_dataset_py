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
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
#lets get the website
#we will make corona virus dataset
website='https://www.worldometers.info/coronavirus/#countries'
website_url=requests.get(website).text
soup = BeautifulSoup(website_url,'html.parser')
my_table = soup.find('tbody')
table_data = []

for row in my_table.findAll('tr'):
    #row_data = []
    for cell in row.findAll('td'):
        row_data.append(cell.text)
        if(len(row_data) >0):
                   data_item = {"Country": row_data[0],
                             "TotalCases": row_data[1],
                             "NewCases": row_data[2],
                             "TotalDeaths": row_data[3],
                             "NewDeaths": row_data[4],
                             "TotalRecovered": row_data[5],
                             "ActiveCases": row_data[6],
                             "CriticalCases": row_data[7],
                             "Totcase1M": row_data[8],
                             "Totdeath1M": row_data[9],
                             "TotalTests": row_data[10],
                             "Tot_1M": row_data[11]}
    table_data.append(data_item)
for cell in row.findAll('td'):
        row_data.append(cell.text)
len(row_data)
df=pd.DataFrame(table_data)
df
df.to_csv('Covid19_data.csv', index=True)
