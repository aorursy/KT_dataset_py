import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/co2-ghg-emissionsdata/co2_emission.csv").dropna()

data.head()
data = data.drop( labels = "Entity", axis = 1 )

data.head()
codes = pd.Series( data["Code"].unique() )

print( codes.to_list() )
print(len(codes))
maxEmission = data[["Code", "Annual CO₂ emissions (tonnes )"]].groupby("Code").max()

minEmission = data[["Code", "Annual CO₂ emissions (tonnes )"]].groupby("Code").min()
plt.plot(maxEmission["Annual CO₂ emissions (tonnes )"], maxEmission.index)

plt.title("Max emission")

plt.show()
plt.plot(minEmission["Annual CO₂ emissions (tonnes )"], minEmission.index)

plt.title("Min emission")

plt.show()
diffEmission = maxEmission["Annual CO₂ emissions (tonnes )"].sub(minEmission["Annual CO₂ emissions (tonnes )"])

sns.barplot( diffEmission.index, np.log(diffEmission),  )