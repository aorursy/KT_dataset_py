import numpy as np                                            # linear algebra

import pandas as pd                                           # data processing, CSV file I/O (e.g. pd.read_csv)

import gc                                                     # Garbage collection

from sklearn import preprocessing                             #For categorisation of variable

import matplotlib.pyplot as plt                               # Plotting

import math                                                   # Simple mathematical computations

from itertools import groupby



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
KAGGLE_DATA_FOLDER = '/kaggle/input/m5-forecasting-accuracy/' # Path to the data

START_DAY = 350

N_TRAIN_DAYS = 1913                                           # 1913 days are available for training, subsequent days are for submission

WINDOW_SIZE = 14                                              # Size of series we train/validate on

TEST_SIZE = 28
def load_data(categorise=True):

   

    sales = pd.read_csv(os.path.join(KAGGLE_DATA_FOLDER, 'sales_train_validation.csv'))

    cat = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    for feature in cat:

        encoder = preprocessing.LabelEncoder()

        sales[feature] = encoder.fit_transform(sales[feature])

    dtypes = ['object','int16','int8','int8','int8','int8','int16'] + ['int16' for i in range(N_TRAIN_DAYS)]

    dtypes_dict = {col:dtype for col, dtype in zip(sales.columns, dtypes)}

    sales = sales.astype(dtypes_dict)

    sales['id'] = sales['id'].apply(lambda id_str: id_str[:18])

    

    return sales
df = load_data()
df.head()
sales = df.drop(['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], axis = 1)

sales = sales.set_index('id')

del df
beginning = sales.iloc[:,0:637]

middle = sales.iloc[:, 638:(638+637)]

end = sales.iloc[:, 1275:(1276+638)]

del sales
beginning
beginning_zeros = (beginning == 0).sum(axis = 1); beginning_non_zeros = (beginning != 0).sum(axis=1)

print(beginning_zeros, beginning_non_zeros, sep="\n")
beginning_zeros_mean = beginning_zeros.mean(); beginning_non_zeros_mean = beginning_non_zeros.mean()

print(beginning_zeros_mean, beginning_non_zeros_mean, sep="\n")
middle
middle_zeros = (middle == 0).sum(axis = 1); middle_non_zeros = (middle != 0).sum(axis = 1)

print(middle_zeros, middle_non_zeros, sep="\n")
middle_zeros_mean = middle_zeros.mean(); middle_non_zeros_mean = middle_non_zeros.mean()

print(middle_zeros_mean, middle_non_zeros_mean, sep="\n")
end
end_zeros = (end == 0).sum(axis = 1); end_non_zeros = (end != 0).sum(axis = 1)

print(end_zeros, end_non_zeros, sep="\n")
end_zeros_mean = end_zeros.mean(); end_non_zeros_mean = end_non_zeros.mean()

print(end_zeros_mean, end_non_zeros_mean, sep="\n")
zero_data = {'Beginning': beginning_zeros_mean, 'Middle': middle_zeros_mean, 'End': end_zeros_mean}

non_zero_data = {'Beginning': beginning_non_zeros_mean, 'Middle': middle_non_zeros_mean, 'End': end_non_zeros_mean}
names = list(zero_data.keys())

zero_values = list(zero_data.values())

non_zero_values = list(non_zero_data.values())
plt.figure(figsize=(8,6))

plt.title('Average Amount of (Non-)Zero Days per Period')

plt.xlabel("Period")

plt.ylabel("Average Amount of (Non-)Zero Days")

plt.grid(zorder=1)

for i, (zero_mean, non_zero_mean) in enumerate(zip(zero_values, non_zero_values)):

    plt.bar(1 + 3*i, zero_mean, zorder=2, label = "Zero" if i==0 else "", color="orange")

    plt.bar(2 + 3*i, non_zero_mean, zorder=2, label = "Non-zero" if i==0 else "", color="blue")

plt.xticks([1.5,4.5,7.5], names)

plt.legend()

plt.savefig("avg-zeros.png", bbox_inches="tight")

plt.show()
%%time



beginning['gaps'] = [[len(list(group)) for flag, group in groupby(row, key = bool) if not flag] for row in beginning.values]

middle['gaps'] = [[len(list(group)) for flag, group in groupby(row, key = bool) if not flag] for row in middle.values]

end['gaps'] = [[len(list(group)) for flag, group in groupby(row, key = bool) if not flag] for row in end.values]
GAP_SIZE = 180
beginning_gaps = beginning.explode('gaps')

beginning_gaps_large = (beginning_gaps['gaps'] > GAP_SIZE).sum()



beginning_gaps_large
middle_gaps = middle.explode('gaps')

middle_gaps_large = (middle_gaps['gaps'] > GAP_SIZE).sum()

middle_gaps_large
end_gaps = end.explode('gaps')

end_gaps_large = (end_gaps['gaps'] > GAP_SIZE).sum()

end_gaps_large
plt.figure(figsize=(8,6))

plt.title('Amount of Gaps Larger than {} Consecutive Days per Period'.format(GAP_SIZE))

plt.xlabel("Period")

plt.ylabel("Amount of Gaps")

plt.grid(zorder=1)

plt.bar(names, [beginning_gaps_large, middle_gaps_large, end_gaps_large], width=0.8, color="blue", zorder=2)

plt.xticks()

plt.savefig("180_day_gaps.png", bbox_inches="tight")

plt.show()