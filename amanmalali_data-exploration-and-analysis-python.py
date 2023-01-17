import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib import style

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
df=pd.read_csv("../input/train.csv")

df_macro=pd.read_csv("../input/macro.csv")
print (df.columns)

print (df.shape)
#correlation matrix for all the variables

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
#Heatmap of Indoor Characteristics

df_indoor=df[['price_doc','full_sq','life_sq','floor','max_floor','state','kitch_sq','num_room']]

corrmat = df_indoor.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat,cbar=True, annot=True, square=True);

df_indoor.fillna(method='bfill',inplace=True)

plt.show()
sns.set()

cols = ['price_doc','full_sq','life_sq','floor','state','num_room']

sns.pairplot(df_indoor[cols], size = 5)

plt.show();
# Heatmap of nearby Recreational Characteristics

df_rec=df[['price_doc','sport_objects_raion', 'culture_objects_top_25_raion', 'shopping_centers_raion','sport_count_1000','sport_count_1500','sport_count_3000','cafe_count_1000','cafe_count_1500','cafe_count_3000']]

corrmat = df_rec.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat,cbar=True, annot=True, square=True);

plt.show()