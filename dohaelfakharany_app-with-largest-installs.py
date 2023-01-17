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
apps = apps[apps['Current Ver'].isna() == 0.0]

# Making sure that the code works



apps[apps['Current Ver'].isna()]
apps['Size'] = apps['Size'].str.replace("M",'', regex=True)

apps['Size'] = apps['Size'].str.replace("Varies with device","0")

apps['Size'] = apps['Size'].str.replace("k","000")

apps['Size'] = apps['Size'].str.replace(",","")

apps['Size'] = apps['Size'].str.replace("+","")

apps["Size"] = pd.to_numeric(apps["Size"])

apps.Size = pd.to_numeric(apps.Size)
# there is no issing data

apps[apps['Installs'].isna()]
apps.Installs
apps.Installs = apps.Installs.str.replace("+","")

apps.Installs = apps.Installs.str.replace(",","")

apps.Installs = apps.Installs.str.replace("Free","0")
apps['Installs'] = pd.to_numeric(apps['Installs'])
apps = apps.drop_duplicates(subset=['App'])
apps.groupby("Category") ['Installs'].sum().sort_values(ascending=False).head(1)
apps.sort_values(["Installs","Reviews","Rating"],ascending=False)