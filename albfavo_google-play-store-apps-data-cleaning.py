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
df_apps=pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')
df_apps.columns.values
df_apps.Category.unique()
trash_data = df_apps[df_apps['Category'] == "1.9"]

df_apps.drop(trash_data.index, inplace=True)
df_apps.Rating.unique()
trash_data = df_apps[df_apps['Rating'] == "nan"]

df_apps.drop(trash_data.index, inplace=True)

df_apps["Rating"] = df_apps["Rating"].astype(float)
df_apps["Reviews"] = df_apps["Reviews"].astype(int)
df_apps.Size.unique()
trash_data = df_apps[df_apps['Size'] == "Varies with device"]

df_apps.drop(trash_data.index, inplace=True)



df_apps["Size"] = df_apps["Size"].apply(lambda x: str(x).strip('M').replace('M', ''))

df_apps["Size"] = df_apps["Size"].apply(lambda x: str(x).strip('M').replace('k', ''))

df_apps["Size"] = df_apps["Size"].astype(float)
df_apps.Installs.unique()
df_apps["Installs"] = df_apps["Installs"].apply(lambda x: str(x).strip('+').replace('+', ''))

df_apps["Installs"] = df_apps["Installs"].apply(lambda x: str(x).strip(',').replace(',', '')).astype(int)
df_apps.Price.unique()
df_apps["Price"] = df_apps["Price"].apply(lambda x: str(x).strip('$').replace('$', ''))

df_apps["Price"] = df_apps["Price"].astype(float)
df_apps["Content Rating"].unique()
trash_data = df_apps[df_apps["Content Rating"] == "Unrated"]

df_apps.drop(trash_data.index, inplace=True)
df_apps["Current Ver"].unique()
trash_data = df_apps[df_apps["Current Ver"] == "Varies with device"]

df_apps.drop(trash_data.index, inplace=True)

trash_data = df_apps[df_apps["Current Ver"] == "nan"]

df_apps.drop(trash_data.index, inplace=True)
df_apps["Android Ver"].unique()
trash_data = df_apps[df_apps["Android Ver"] == "Varies with device"]

df_apps.drop(trash_data.index, inplace=True)

trash_data = df_apps[df_apps["Android Ver"] == "nan"]

df_apps.drop(trash_data.index, inplace=True)
df_apps.head()