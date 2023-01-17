import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
from sodapy import Socrata

client = Socrata("data.seattle.gov", None)

results = client.get("ht3q-kdvx", limit=2000)

land = pd.DataFrame.from_records(results)

land.head()
land.shape
land.describe()
land.statuscurrent.value_counts()
landShaped = land[['permitnum','applieddate','permitclassmapped','permitclass','statuscurrent','description','originaladdress1','estprojectcost']]

landShaped.head()
landShaped['estprojectcost'] = pd.to_numeric(land['estprojectcost'])

landShaped['estprojectcost'].mean()
landShaped['estprojectcost'].dropna().sort_values(ascending=False)
landShaped['estprojectcost'].count()
import matplotlib.pyplot as plt



plt.rcdefaults()

fig, ax = plt.subplots()



permit = landShaped['permitnum'].iloc[1:20]

y_pos = np.arange(len(permit))

y_label = landShaped['originaladdress1']

bars = landShaped['estprojectcost'].sort_values(ascending=False).iloc[1:20]



ax.barh(y_pos, bars, align='center')

ax.set_yticks(y_pos)

ax.set_yticklabels(y_label)



ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Cost of Permit Project')

ax.set_title('20 Highest Permit Project Estimates')



plt.show()
land['housingunits'] = pd.to_numeric(land['housingunits']).dropna()

land.plot(y="housingunits")
land['housingunits'].sum()
land['housingunits'].mean()
client = Socrata("data.seattle.gov", None)

results = client.get("i6qv-ar46", limit=2000)

capacity = pd.DataFrame.from_records(results)

capacity.head()
capacity.shape
1320*246 / 5280
capacity.class_description.value_counts()
capacity.empl_per_sqft.describe()
capacity.empl_per_sqft.sum()
capacity['empl_per_sqft'] = pd.to_numeric(capacity['empl_per_sqft']).dropna()

capacity.plot(y='empl_per_sqft')
capacity['empl_per_sqft'].mean()
import re 

perc =[.20, .40, .60, .80] 

include =['object', 'float', 'int'] 

desc = capacity.describe(percentiles = perc, include = include) 

desc 