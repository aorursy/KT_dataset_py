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
import json

df = pd.DataFrame(columns=['iso2', 'iso3', 'name', 'capital', 'phone', 'currency'])

with open('../input/names.json', 'r') as f_names:
    with open('../input/iso3.json', 'r') as f_iso3:
        with open('../input/phone.json', 'r') as f_phone:
            with open('../input/currency.json', 'r') as f_currency:
                with open('../input/capital.json', 'r') as f_capital:
                    
                    names = json.load(f_names)
                    iso3 = json.load(f_iso3)
                    phone = json.load(f_phone)
                    currency = json.load(f_currency)
                    capital = json.load(f_capital)
                    for iso2 in names:
                        df = df.append({'iso2': iso2, 'iso3': iso3[iso2], 'name': names[iso2], 'capital': capital[iso2], 'currency': currency[iso2], 'phone': phone[iso2]}, ignore_index=True)
                        #print(iso2 + '/' + iso3[iso2] + ' - ' + names[iso2] + ' - ' + capital[iso2] + ' - ' + phone[iso2] + ' - ' + currency[iso2])
df