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
df = pd.read_csv('../input/procurement-notices.csv')

df.head()
import pycountry
mapping = {country.alpha_2: country.name for country in pycountry.countries}
print(mapping['EG'])
def stringify(x):
    if str(x) in mapping.keys():
        return mapping[str(x)]
    return "not present"
df.dtypes
from datetime import datetime

df['Deadline Date'] = pd.to_datetime(df['Deadline Date'])
df = df  = df[df['Deadline Date'] > datetime.now()]
df = df[~df['Country Code'].isna()]
df['Country Code'] = df['Country Code'].apply(stringify)
df = df[df['Country Code'] != "not present"]
df = df.groupby(['Country Code'])['ID'].count()
df
import geopandas

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

world.head()
world = world.set_index('name').join(df)
world.count()
world = world[~world['ID'].isna()]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, figsize=(20, 12))
world.plot(column='ID',ax = ax)