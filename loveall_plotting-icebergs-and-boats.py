import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns
INPUT_PATH = '../input/'
# Read the json files into a pandas dataframe

df_train = pd.read_json(INPUT_PATH + 'train.json')
# FIrst get 9 random icebergs

df_plot = data.loc[data['is_iceberg'] == 1].sample(9)



plt.figure(figsize=(16,16))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.scatter(df_plot.iloc[i]['band_1'],df_plot.iloc[i]['band_2'])

    plt.xlabel('Band 1')

    plt.ylabel('Band 2')

plt.suptitle('ICEBERGS!',fontsize=20)





# FIrst get 9 random icebergs

df_plot = data.loc[data['is_iceberg'] == 0].sample(9)



plt.figure(figsize=(16,16))

for i in range(9):

    plt.subplot(3,3,i+1)

    plt.scatter(df_plot.iloc[i]['band_1'],df_plot.iloc[i]['band_2'])

    plt.xlabel('Band 1')

    plt.ylabel('Band 2')

    

plt.suptitle('BOATS!', fontsize=20)



plt.tight_layout()