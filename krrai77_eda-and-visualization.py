# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns

import matplotlib.pyplot as plt
chem=pd.read_csv("/kaggle/input/chemicals-in-cosmetics/cscpopendata.csv")
chem.head()
chem.columns
chem.info()
chem['MostRecentDateReported']=pd.to_datetime(chem['MostRecentDateReported'])
chemcos=chem[['ProductName','ChemicalName','CompanyName','MostRecentDateReported','DiscontinuedDate','ChemicalCount']]
chemcos.ProductName.unique()
sns.countplot(x=chemcos['ChemicalCount'])
chemcos['year']=chemcos['MostRecentDateReported'].dt.year
chemgrp=chemcos.groupby('year')

grp20=chemgrp.get_group(2020)
grp20
grp20.ChemicalName.value_counts()
grptd=grp20.loc[grp20['ChemicalName'] == "Titanium dioxide"]
plt.figure(figsize=(25,10))

grptd.ProductName.value_counts().plot()
plt.figure(figsize=(20,10))

grptd.CompanyName.value_counts().plot()
import plotly.express as px



fig = px.line(grp20, x="MostRecentDateReported", y="ChemicalName", title='Chemical Detection', color='CompanyName')

fig.show()
fig = px.bar(grp20, x='ProductName', y='ChemicalCount')

fig.show()
fig = px.bar(grp20, x='ChemicalCount', y='CompanyName')

fig.show()