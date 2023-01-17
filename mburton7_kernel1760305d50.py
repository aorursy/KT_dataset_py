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
from sodapy import Socrata
# Unauthenticated client only works with public data sets. Note 'None'

# in place of application token, and no username or password:

client = Socrata("data.seattle.gov", None)



# Example authenticated client (needed for non-public datasets):

# client = Socrata(data.seattle.gov,

#                  MyAppToken,

#                  userame="user@example.com",

#                  password="AFakePassword")



# First 2000 results, returned as JSON from API / converted to Python list of

# dictionaries by sodapy.

results = client.get("ht3q-kdvx", limit=2000)



# Convert to pandas DataFrame

land = pd.DataFrame.from_records(results)
land.head()
land.shape
land.statuscurrent.value_counts()
land.isnull().sum()
land.dropna(subset=["statuscurrent"],inplace=True) 

land.drop(["contractorcompanyname"],axis=1,inplace=True)

land.drop(["permitnum"],axis=1,inplace=True)

land.shape
land['housingunits'].describe()
import altair as alt



alt.Chart(land).mark_point().encode(

    y='average(estprojectcost)',

    x='permitclass',

    color='permitclass',

).interactive()
alt.Chart(land).mark_point().encode(

    y='housingunitsremoved',

    x='housingunitsadded',

    color='statuscurrent',

).interactive()
land['housingunits'].sum(skipna=True)
client = Socrata("data.seattle.gov", None)

results = client.get("i6qv-ar46", limit=2000)

capacity = pd.DataFrame.from_records(results)

capacity.shape

capacity.head()
alt.Chart(capacity).mark_point().encode(

    y='empl_per_sqft',

    x='du_acre',

    color='class_description',

).interactive()
capacity.plot('maxheight',y,kind='scatter')
capacity['category'].hist()