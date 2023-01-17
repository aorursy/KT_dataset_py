import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Read cereal data into a dataframe

cereals = pd.read_csv('../input/cereal.csv')

print(cereals.shape)

print(cereals.head())
#How many cereals of each type, cold and hot

cereals_type_cold = cereals[cereals['type']=='C']

cereals_type_hot = cereals[cereals['type']=='H']

cereals_type_cold_count = cereals_type_cold.shape[0]

cereals_type_hot_count = cereals_type_hot.shape[0]

print(cereals_type_cold_count)

print(cereals_type_hot_count)
types = ['C', 'H']

y_pos = np.arange(len(types))

count = [cereals_type_cold_count, cereals_type_hot_count]

plt.bar(y_pos, count, align='center', alpha=0.5, edgecolor='black')

plt.xticks(y_pos, types)

plt.ylabel('Count')

plt.title('Cereal count of each type')
#what is the average cereal rating for each manufacturer

manufacturer_rating = {}

for mfr, data in cereals.groupby('mfr'):

    avg_rating = data['rating'].mean()

    print('Average rating for manufacturer {} is {:.2f}'.format(mfr, avg_rating))

    manufacturer_rating[mfr] = avg_rating
#Let's create a horizontal bar chart for the same

manufacturers = manufacturer_rating.keys()

ratings = manufacturer_rating.values()

y_pos1 = np.arange(len(manufacturers))

plt.barh(y_pos1, ratings, align='center', alpha=0.5, edgecolor='black')

plt.yticks(y_pos1, manufacturers)

plt.xlabel('Rating')

plt.title('Average cereal rating for manufacturer')