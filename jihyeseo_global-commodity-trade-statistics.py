# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import matplotlib 

import matplotlib.pyplot as plt

import sklearn

%matplotlib inline

import matplotlib.pyplot as plt 

plt.rcParams["figure.figsize"] = [16, 12]

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
filename = check_output(["ls", "../input"]).decode("utf8").strip()

df = pd.read_csv("../input/" + filename, thousands=",")
df.dtypes
df.head()
df['commodity'].unique()
df['category'].unique()
def subsetWithCategory(catName):

    return df[df['category'] == catName] 

    

dA = subsetWithCategory('01_live_animals')

dB = subsetWithCategory('49_printed_books_newspapers_pictures_etc')

dC = subsetWithCategory('45_cork_and_articles_of_cork')
dA.head()
dA['commodity'].unique()
dG = dA[dA['commodity']=='Goats, live']
dG.head()
dGE = dG[dG['flow'] == 'Export']

dGI = dG[dG['flow'] == 'Import']
dGE.sort_values('trade_usd', ascending = False).head(10)
dGI.sort_values('trade_usd', ascending = False).head(10)
goatExportingCountries = set(dGE['country_or_area'].unique())

goatImportingCountries = set(dGI['country_or_area'].unique())

print(len(goatExportingCountries), "countries export live goats, while", len(goatImportingCountries), "countries import.")
importandexport  = goatImportingCountries.intersection(goatExportingCountries)
len(importandexport)