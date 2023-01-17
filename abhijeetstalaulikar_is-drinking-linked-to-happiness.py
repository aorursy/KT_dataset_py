import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

%matplotlib inline

import seaborn as sns
raw_data = pd.read_csv('../input/happiness-and-alcohol-consumption/HappinessAlcoholConsumption.csv')

raw_data.head()
raw_data['Total_Alcohol_PerCapita'] = raw_data.Beer_PerCapita + raw_data.Spirit_PerCapita + raw_data.Wine_PerCapita
print(raw_data.Hemisphere.value_counts())

raw_data.Hemisphere = raw_data.Hemisphere.str.replace('noth','north')
north_hemisphere = raw_data[raw_data.Hemisphere == "north"]

south_hemisphere = raw_data[raw_data.Hemisphere == "south"]



north_mean_consumption = north_hemisphere.drop(['HappinessScore','HDI','GDP_PerCapita'], axis=1).mean()

north_mean_consumption['Hemisphere'] = 'north'

south_mean_consumption = south_hemisphere.drop(['HappinessScore','HDI','GDP_PerCapita'], axis=1).mean()

south_mean_consumption['Hemisphere'] = 'south'



means = pd.DataFrame(data = [north_mean_consumption, south_mean_consumption])

melted = pd.melt(means, id_vars = ['Hemisphere'], value_vars = ['Beer_PerCapita','Spirit_PerCapita','Wine_PerCapita'])

#display(melted)



sns.barplot(x = 'variable', y='value', hue = 'Hemisphere', data = melted);
means = raw_data.drop(['HappinessScore','HDI','GDP_PerCapita'], axis=1).groupby(by = 'Region').mean().reset_index()

melted = pd.melt(means, id_vars = ['Region'], value_vars = ['Beer_PerCapita','Spirit_PerCapita','Wine_PerCapita','Total_Alcohol_PerCapita'])



plt.figure(figsize=(14,9))

sns.barplot(x = 'variable', y='value', hue = 'Region', data = melted);
plt.figure(figsize=(10,6))

plt.xticks(rotation=-45)

sns.violinplot(x='Region',

               y='Total_Alcohol_PerCapita', 

               data=raw_data);
melted = raw_data.drop(['Country','Region','Hemisphere','HappinessScore','HDI','GDP_PerCapita','Total_Alcohol_PerCapita'],axis=1).melt()



plt.figure(figsize=(10,6))

sns.violinplot(x='variable',

               y='value', 

               data=melted);
sns.kdeplot(raw_data.HDI, raw_data.Total_Alcohol_PerCapita, shade = True);
sns.lmplot(x = 'HappinessScore', y = 'Beer_PerCapita', data = raw_data);

sns.jointplot(x = 'HDI', y = 'Beer_PerCapita', data = raw_data);
sns.lmplot(x = 'HappinessScore', y = 'Spirit_PerCapita', data = raw_data);

sns.jointplot(x = 'HDI', y = 'Spirit_PerCapita', data = raw_data);
sns.lmplot(x = 'HappinessScore', y = 'Wine_PerCapita', data = raw_data);

sns.jointplot(x = 'HDI', y = 'Wine_PerCapita', data = raw_data);