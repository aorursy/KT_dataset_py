# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
menu = pd.read_csv('/kaggle/input/nutrition-facts/menu.csv')
menu.shape
menu.sort_values('Serving Size' ).tail(10)
menu.loc[ menu.Sugars.idxmax() ].Item
menu.set_index('Item').loc['Egg McMuffin', 'Calories']
menu.Category.value_counts()
menu.groupby("Category").Calories.mean().round(2)
menu.plot.scatter(x='Carbohydrates', y='Total Fat')
menu.Category.value_counts().plot.pie()