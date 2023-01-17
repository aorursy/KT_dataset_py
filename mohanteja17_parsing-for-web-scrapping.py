# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from lxml import html

import requests
page = requests.get('http://econpy.pythonanywhere.com/ex/001.html')

tree = html.fromstring(page.content)
#This will create a list of buyers:

buyers = tree.xpath('//div[@title="buyer-name"]/text()')

#This will create a list of prices

prices = tree.xpath('//span[@class="item-price"]/text()')
print('Buyers: ', buyers)
print('Prices: ', prices)