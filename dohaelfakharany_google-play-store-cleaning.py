import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns 



apps = pd.read_csv(r"../input/google-play-store-apps/googleplaystore.csv") 
apps.head()
apps.tail()
apps.shape
apps.info()
# replacing NaNs in Type column with 0 

apps = apps[apps['Type'].isna() == 0]



# viewing the dataset Type NaNs to check that it works

apps[apps["Type"].isna()]
# we removed this index because it has a null value 

apps = apps.drop(axis=0 , index=10472)
apps = apps[apps['Current Ver'].isna() == 0.0]

# Making sure that the code works



apps[apps['Current Ver'].isna()]
apps = apps[apps['Android Ver'].isna()== 0.0]

# Making sure that code works

apps[apps['Android Ver'].isna()]
# replacing NaNs in Type column with 0 

apps = apps[apps['Rating'].isna() == 0]



# viewing the dataset Type NaNs to check that it works

apps[apps["Rating"].isna()]
apps.dtypes
apps['Reviews'] = apps['Reviews'].replace(r'^\s*$', np.NaN, regex=True)

apps['Reviews'].fillna(0)

apps['Reviews'] = pd.to_numeric(apps['Reviews'])
apps['Size'] = apps['Size'].replace(r'^\s*$', np.NaN, regex=True)

apps['Size'].fillna(0)

apps["Size"] = pd.to_numeric(apps.loc[:, "Size"], errors='coerce').fillna(0)
apps.Installs  = apps['Installs'].str.replace("+","")

apps.Installs = apps.Installs.str.replace(",","")

apps['Installs'] = pd.to_numeric(apps['Installs'])
apps.Type = apps.Type.str.replace("Free","1")

apps.Type = apps.Type.str.replace("Paid","0")

apps['Type'] = pd.to_numeric(apps.Type)

apps['Type'] = apps['Type'].astype("category")
apps.Price = apps.Price.str.replace("$","")

apps.Price = pd.to_numeric(apps.Price)
apps['Content Rating'] = apps['Content Rating'].str.replace("Kids 10+","Kids 10")
apps['Genres'] = apps.Genres.str.lower()

apps['Genres'] = apps.Genres.str.replace(";",",")
apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])

apps['Last Updated'] = apps['Last Updated'].dt.strftime('%d/%m/%Y')

apps['Last Updated'] = pd.to_datetime(apps['Last Updated'])
apps.drop( apps[apps["Current Ver"] == "Varies with device"].index, inplace=True)

apps.drop(apps[apps["Current Ver"] == "nan"].index, inplace=True)
apps.App.str.lower()