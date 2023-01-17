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
import pandas as pd 

df = pd.read_csv('/kaggle/input/california-housing-prices/housing.csv')

display(df.head())
def convert(n):

    return n * 10000



converted = df['median_income'].apply(convert)

display(converted.head())



# update value 

df['median_income'] = converted

display(df.head())
def category(n):

    value = n / 10000

    if value > 10:

        return 'high-income'

    elif value > 2 and value < 10:

        return 'moderate-income'

    else: 

        return 'low-income'

    

categories = df['median_income'].apply(category)

df['income-category'] = categories

display(df.head())



print(df['income-category'].value_counts())