# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from glob import glob



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
drinks = pd.read_csv('http://bit.ly/drinksbycountry')

movies = pd.read_csv('http://bit.ly/imdbratings')

orders = pd.read_csv('http://bit.ly/chiporders', sep='\t')

orders['item_price'] = orders.item_price.str.replace('$', '').astype('float')

stocks = pd.read_csv('http://bit.ly/smallstocks', parse_dates=['Date'])

titanic = pd.read_csv('http://bit.ly/kaggletrain')

ufo = pd.read_csv('http://bit.ly/uforeports', parse_dates=['Time'])
pd.__version__
pd.show_versions()
df=pd.DataFrame({"col one":[100,200], "col two":[300,400]})

df
pd.DataFrame(np.random.rand(4,8))
pd.DataFrame(np.random.rand(4,5),columns=list("queen"))
df
df =df.rename({"col one":"col_one","col two":"col_two"},axis="columns")

df
df.columns=["colone","coltwo"]

df
df.columns = df.columns.str.replace("col","wol")

df
df.add_prefix("x_")
df.add_suffix("x_")
drinks.head()
drinks.loc[::-1].head()
drinks.loc[::-1].reset_index(drop=True).head()
drinks.head()
drinks.loc[:,::-1].head()
drinks.dtypes
drinks.select_dtypes(include="number").head()
drinks.select_dtypes(include="object").head()
drinks.select_dtypes(include=["number","object","category","datetime"]).head()
drinks.select_dtypes(exclude="number").head()
df=pd.DataFrame({"col_one":["1.1","2.2","3.3"],

                 "col_two":["4.4","5.5","6.6"],

                 "col_three":["7.7","8.8","-"]

                })

df
df.dtypes

df.astype({"col_one":"float","col_two":"float"}).dtypes
pd.to_numeric(df.col_three,errors="coerce")
pd.to_numeric(df.col_three,errors="coerce").fillna(0)
df=df.apply(pd.to_numeric,errors="coerce").fillna(0)

df
df.dtypes
drinks.info(memory_usage="deep")
cols=["beer_servings","continent"]

small_drinks=pd.read_csv("http://bit.ly/drinksbycountry",usecols=cols)

small_drinks.info(memory_usage="deep")
dtypes={"continent":"category"}

smaller_drinks=pd.read_csv("http://bit.ly/drinksbycountry",usecols=cols,dtype=dtypes)

smaller_drinks.info(memory_usage="deep")