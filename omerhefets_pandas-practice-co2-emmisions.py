import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
co2_emmisions_data = pd.read_csv("/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv")
co2_emmisions_data
co2_emmisions_data['Entity']
co2_emmisions_data['Entity'].loc[0]
co2_emmisions_data['Entity'].loc[20852]
co2_emmisions_data.loc[co2_emmisions_data["Entity"] == "Afghanistan", 'Entity']
co2_emmisions_data['Entity'].loc[co2_emmisions_data["Entity"] == "Afghanistan"]
afghanistan_co2_emmisions = co2_emmisions_data.loc[co2_emmisions_data["Entity"] == "Afghanistan",

                                                   ['Entity', 'Year', 'Annual CO₂ emissions (tonnes )']]
afghanistan_co2_emmisions.head()
afghanistan_co2_emmisions = afghanistan_co2_emmisions.rename(columns={"Annual CO₂ emissions (tonnes )": "Annual Emissions (tonnes)"})
plt.figure(figsize=(15,7.5))

sns.lineplot(data=afghanistan_co2_emmisions['Annual Emissions (tonnes)'])
print("The mean emmisions in afghanistan over {} years is: {}".format(afghanistan_co2_emmisions['Year'].count(), 

                                                                      afghanistan_co2_emmisions['Annual Emissions (tonnes)'].mean()))
plt.figure(figsize=(15,7.5))

sns.regplot(x=afghanistan_co2_emmisions['Year'],

            y=afghanistan_co2_emmisions['Annual Emissions (tonnes)'])