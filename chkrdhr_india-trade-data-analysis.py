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
import seaborn as sns

import matplotlib.pyplot as plt
export_data = pd.read_csv('../input/india-trade-data/2018-2010_export.csv')

import_data = pd.read_csv('../input/india-trade-data/2018-2010_import.csv')
export_data.head()
import_data.head()
export_data.isnull().sum()
import_data.isnull().sum()
import_data.duplicated().sum()
export_data.duplicated().sum()
def cleanup(data_df):

    #setting country UNSPECIFIED to nan

    data_df['country']= data_df['country'].apply(lambda x : np.NaN if x == "UNSPECIFIED" else x)

    #ignoring where import value is 0 . 

    data_df = data_df[data_df.value!=0]

    data_df.dropna(inplace=True)

    data_df.year = pd.Categorical(data_df.year)

    data_df.drop_duplicates(keep="first",inplace=True)

    return data_df
export_clean_data = cleanup(export_data)
import_clean_data = cleanup(import_data)
import_clean_data.isnull().sum()
gcd = export_clean_data.groupby(['year','Commodity'],as_index=False,sort=False)['value'].sum()
best_export_year_wise = gcd.loc[gcd.groupby(['year'],as_index=False,sort=False)['value'].idxmax()]

best_export_year_wise
#Highest imports from China

import_country_wise = import_clean_data.groupby('country',as_index=False,sort=False)['value'].sum()

subset_import = import_country_wise.sort_values('value').tail(10)
f, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(x="value", y="country", data=subset_import)
#Highest exports to USA

export_country_wise = export_clean_data.groupby('country',as_index=False,sort=False)['value'].sum()

subset_export = export_country_wise.sort_values('value').tail(10)
f, ax = plt.subplots(figsize=(12, 8))

ax = sns.barplot(x="value", y="country", data=subset_export)