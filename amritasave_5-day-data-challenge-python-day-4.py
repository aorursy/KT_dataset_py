import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from scipy.stats import ttest_ind

from subprocess import check_output





cereal_df = pd.read_csv('../input/cereal.csv')





values_dict = {'N': 'Nabisco', 'Q': 'Quaker Oats', 'K': 'Kelloggs', 'R': 'Raslston Purina', 'G': 'General Mills' , 'P' :'Post' , 'A':'American Home Foods Products'}

cereal_df['mfr_name'] = cereal_df['mfr'].map(values_dict)



#sns.set(style="whitegrid")

ax = plt.subplots(figsize=(12,8))

ax = sns.countplot(x="mfr_name", hue="type",data=cereal_df,palette='summer')

plt.title('Number of cereals as per Manufacturer and Type')

plt.xlabel('Manufacturer')

plt.ylabel('Count')

plt.xticks(rotation=30)
sns.factorplot(x = 'sugars', 

               col='type', kind ='count',

               data= cereal_df )

plt.figure(figsize = (10, 4))

sns.countplot(x = 'vitamins',

              hue = 'mfr_name',

              edgecolor = sns.color_palette("winter"),

              data = cereal_df ).set_title('Vitamins count as per manufacturers')

plt.legend(loc='upper left')
plt.figure(figsize = (10, 4))

sns.boxplot(cereal_df['mfr_name'],cereal_df['calories'])

plt.xticks(rotation=10)

plt.title('Calories of cereals as per Manufacturer')

plt.xlabel('Manufacturer')

plt.ylabel('Calories')
plt.figure(figsize = (10, 4))

sns.violinplot(cereal_df['mfr_name'],cereal_df['sugars'],scale='count' ,palette='bright')

plt.xticks(rotation=10)

plt.title('Sugars of cereals as per Manufacturer')

plt.xlabel('Manufacturer')

plt.ylabel('Sugars')
plt.figure(figsize = (10, 4))

sns.swarmplot('mfr_name','calories',hue='type',data=cereal_df, palette='muted')

plt.title('Calories of cereals as per Manufacturer and Type')

plt.xlabel('Manufacturer')

plt.ylabel('Calories')

plt.xticks(rotation=10)