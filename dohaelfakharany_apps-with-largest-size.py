import pandas as pd 

import numpy as np 

import seaborn as sns

import matplotlib.pyplot as plt



apps = pd.read_csv(r"../input/google-play-store-apps/googleplaystore.csv")
apps['Size'] = apps['Size'].str.replace("M",'', regex=True)

apps['Size'] = apps['Size'].str.replace("Varies with device","0")

apps['Size'] = apps['Size'].str.replace("k","000")

apps['Size'] = apps['Size'].str.replace(",","")

apps['Size'] = apps['Size'].str.replace("+","")

apps["Size"] = pd.to_numeric(apps["Size"])
apps.Size = pd.to_numeric(apps.Size)
top_apps = apps.sort_values("Size",ascending=False).head(10)

top_apps
fig , ax = plt.subplots(figsize=(10,8))

ax.bar(top_apps['App'],top_apps['Size'].mean().value_counts().head(10) ,color=['blue',"violet","pink","red","orange","yellow","white"])



for tick in ax.get_xticklabels():

    tick.set_rotation(90)