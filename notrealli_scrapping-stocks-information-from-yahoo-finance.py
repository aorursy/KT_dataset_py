# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import requests

from bs4 import BeautifulSoup as bs

from string import ascii_uppercase



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def get_symbols_from_alphabet(alpha):

    r = requests.get('http://www.eoddata.com/stocklist/TSX/{}.htm'.format(alpha))

    soup = bs(r.content)

    div = soup.find('div', attrs={'id': 'col1w'})

    table = div.find('table', attrs={'class': 'quotes'})

    quotes = table.find_all('a')

    return [quote.text for quote in quotes if quote.text != '']
def get_stats_from_symbol(symbol):

    r = requests.get('https://finance.yahoo.com/quote/{}/key-statistics'.format(symbol))

    soup = bs(r.content)

    

    stats = {}

    

    if soup.find('table', class_='W(100%) Bdcl(c) M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)') != None:

        # name

        name = soup.find('h1', class_='D(ib) Fz(18px)').text 



        stats['name'] = name



        # valuation measures

        vm_table = soup.find('table', attrs={'class': 'W(100%) Bdcl(c) M(0) Whs(n) BdEnd Bdc($seperatorColor) D(itb)'})

        key_rows = vm_table.find_all('td', class_=lambda c: c and c.startswith('Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pend(10px)'))

        keys = [row.find('span').text for row in key_rows]



        value_rows = vm_table.find_all('td', attrs={'class': 'Ta(c) Pstart(10px) Miw(60px) Miw(80px)--pnclg Bgc($lv1BgColor) fi-row:h_Bgc($hoverBgColor)'})

        values = [row.text for row in value_rows]



        for i in range(len(keys)):

            stats[keys[i]] = values[i]



        # financial highlights

        fh_div = soup.find('div', attrs={'class': 'Mb(10px) Pend(20px) smartphone_Pend(0px)'})

        key_rows = fh_div.find_all('td', class_=lambda c: c and c.startswith('Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pend(10px)'))

        keys = [row.find('span').text for row in key_rows]



        value_rows = fh_div.find_all('td', class_='Fw(500) Ta(end) Pstart(10px) Miw(60px)')

        values = [row.text if row.text != None else row.find('span').text for row in value_rows]



        for i in range(len(keys)):

            stats[keys[i]] = values[i]



        # trading_information

        ti_div = soup.find('div', class_='Pstart(20px) smartphone_Pstart(0px)')

        key_rows = ti_div.find_all('td', class_=lambda c: c and c.startswith('Pos(st) Start(0) Bgc($lv2BgColor) fi-row:h_Bgc($hoverBgColor) Pend(10px)'))

        keys = [row.find('span').text for row in key_rows]



        value_rows = ti_div.find_all('td', class_='Fw(500) Ta(end) Pstart(10px) Miw(60px)')

        values = [row.text if row.text != None else row.find('span').text for row in value_rows]



        for i in range(len(keys)):

            stats[keys[i]] = values[i]



    return stats
infos = []



for alpha in ascii_uppercase[:10]:

    symbols = get_symbols_from_alphabet(alpha)

    temp = [get_stats_from_symbol(symbol) for symbol in symbols]

    infos += [info for info in temp if info]

    print(alpha, len(infos))

infos
tsx = pd.DataFrame(infos).iloc[:, :60]

tsx.head()
tsx.info()
tsx.to_csv('tsx_stocks.csv')