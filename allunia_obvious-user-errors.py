import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set_style("darkgrid")

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
data = pd.read_csv("../input/nutrition_table.csv", index_col=0)
data.head()
data.shape[0]
sugar_errors = data[data.sugars_100g > data.carbohydrates_100g].copy()
sugar_errors.head(6)
sugar_errors.shape[0] / data.shape[0] * 100
sugar_errors["differences"] = sugar_errors["sugars_100g"].values - sugar_errors["carbohydrates_100g"]
plt.figure(figsize=(20,5))
sns.distplot(sugar_errors.differences, color="Fuchsia", kde=False, bins=30)
plt.xlabel("Difference: Sugars - Carbohydrates")
plt.ylabel("Frequency")
plt.title("Error Kind: Sugars exceed Carbohydrates")
sugar_errors[sugar_errors.differences > 80]
sugar_errors[sugar_errors.differences < 1].head(5)
energy_errors = np.zeros(data.shape[0])
energy_errors[(data.energy_100g > 3900) | (data.reconstructed_energy > 3900)] = 1
energy_errors[(data.energy_100g == 0) & (data.reconstructed_energy > 0)] = 2
energy_errors[(data.energy_100g > 0) & (data.reconstructed_energy == 0)] = 3
fig, ax = plt.subplots(1,2,figsize=(20,5))
ax[0].scatter(data.energy_100g,
              data.reconstructed_energy,
              color="tomato",
              s=0.2)
ax[0].set_xlabel("User provided energy")
ax[0].set_ylabel("Reconstructed energy")
ax[0].set_title("Energy with obvious errors")
ax[1].scatter(data.energy_100g.values[energy_errors == 0],
              data.reconstructed_energy.values[energy_errors == 0],
              color="mediumseagreen",
              s=0.2)
ax[1].set_xlabel("User provided energy")
ax[1].set_ylabel("Reconstructed energy")
ax[1].set_title("Energy without obvious errors")
data[energy_errors == 2].head(10)
data[energy_errors == 3].head(10)
data.min()[data.min() < 0]
data[(data.sugars_100g < 0) | (data.proteins_100g < 0) ]
data["exceeded"] = np.where(((data.g_sum > 100) | (data.g_sum < 0)), 1, 0)
data[data.exceeded==1].shape[0] / data.shape[0] * 100
exceeds = data[(data.g_sum < 0) | (data.g_sum > 100)].g_sum.value_counts()
exceeds = exceeds.iloc[0:10]

plt.figure(figsize=(20,5))
sns.barplot(x=exceeds.index.values, y=exceeds.values, order=exceeds.index, palette="Reds")
plt.title("Common exceeding amouts of proteins, fat, carbohydrates")
plt.xlabel("Summed amounts of proteins, fat, carbohydrates")
plt.ylabel("Frequency")
data["g_sum"] = np.round(data.g_sum)
g_sum_errors = data[(data.g_sum < 0) | (data.g_sum > 100)].g_sum.value_counts().sum()
g_sum_errors /data.shape[0]