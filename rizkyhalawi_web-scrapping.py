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
#import library yang dibutuhkan
import pandas as pd
import requests
from bs4 import BeautifulSoup
#buatlah request ke website
website_url = requests.get('https://id.wikipedia.org/wiki/Demografi_Indonesia').text
soup = BeautifulSoup(website_url, 'lxml')
#ambil table dengan class 'wikitable sortable'
my_table = soup.find('table', {'class':'wikitable sortable'})
#cari data dengan tag 'td'
links = my_table.findAll('td')
#buatlah lists kosong
kode_bps = []
nama = []
ibu_kota = []
populasi = []
luas_km = []
pulau = []
#memasukkan data ke dalam list berdasarkan pola HTML
for i, link in enumerate(links):
    if i in range(0, len(links), 9):
        kode_bps.append(link.get_text())
    if i in range(2, len(links), 9):
        nama.append(link.get_text())
    if i in range(4, len(links), 9):
        ibu_kota.append(link.get_text())
    if i in range(5, len(links), 9):
        populasi.append(link.get_text())
    if i in range(6, len(links), 9):
        luas_km.append(link.get_text())
    if i in range(8, len(links), 9):
        pulau.append(link.get_text()[:-1])
#buatlah DatFrame dan masukkan ke CSV
df = pd.DataFrame()
df['Kode BPS'] = kode_bps
df['Nama'] = nama
df['Ibu Kota'] = ibu_kota
df['Populasi'] = populasi
df['Luas km'] = luas_km
df['Pulau'] = pulau
df.to_csv('Indonesia_Demography_by_Province.csv', index=False, encoding='utf-8', quoting=1)


print(df)
