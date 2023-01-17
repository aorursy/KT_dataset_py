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
menu = pd.read_csv('/kaggle/input/sg-kopi/kopitiam.csv')

menu
import re

subs = []

leng = []

for i in range(menu['English'].shape[0]):

    a = menu['English'][i]

    sep1 = 'without'

    ingredient = a.split(sep1, 1)[0]

    ingredient = re.split('with | but | and|,',ingredient)

    print(i)

    print(menu['Singlish'][i], ingredient, len(ingredient))

    subs.append(ingredient)

    leng.append(len(ingredient))

    

contain = pd.DataFrame(subs, columns = ['Ingredient 1','Ingredient 2','Ingredient 3','Ingredient 4','Ingredient 5'])

length  = pd.DataFrame(leng, columns = ['Ingredient count'])

contain['Ingredient 1'] = contain['Ingredient 1'].str.strip()

contain['Ingredient 2'] = contain['Ingredient 2'].str.strip()

contain['Ingredient 3'] = contain['Ingredient 3'].str.strip()

contain['Ingredient 4'] = contain['Ingredient 4'].str.strip()

contain['Ingredient 5'] = contain['Ingredient 5'].str.strip()
menu = menu.join(contain)

menu = menu.join(length)
menu = menu.drop('Source',axis = 1)
print(menu['Ingredient 1'].value_counts(), '\n\n')

print(menu['Ingredient 2'].value_counts(), '\n\n')

print(menu['Ingredient 3'].value_counts(), '\n\n')

print(menu['Ingredient 4'].value_counts(), '\n\n')

print(menu['Ingredient 5'].value_counts(), '\n\n')
menu.fillna(value=pd.np.nan, inplace=True)

menu.head(5)
menu['price (SGD)'] = menu['Ingredient count']
menu.head(5)
menu['Ingredient 1 stock'] =100

menu['Ingredient 2 stock'] =50

menu['Ingredient 3 stock'] =30

menu['Ingredient 4 stock'] =20

menu['Ingredient 5 stock'] =10
menu.head(5)
order_singlish = menu['Singlish'].tolist()

order_english = menu['English'].tolist()

order = np.concatenate((order_singlish, order_english))
print('Menus: ',order)
buy = []

many = []

import random

random.seed(42)

for i in range(100):

    kind = random.randint(1,3)

    for k in range(kind):

        pick = random.choice(order)

        buy.append(pick)

        q = random.randint(1,4)

        many.append(q)

        

    print(pick,q)

buy = pd.DataFrame(buy, columns = ['Ordered beverage'])

many = pd.DataFrame(many, columns = ['Counts'])
buy = buy.join(many)
buy.head(10)