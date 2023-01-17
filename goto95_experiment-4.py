import pandas as pd

import numpy as np



from matplotlib import pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/work_3.csv', delimiter=';', usecols=['metrics', 'group'])

df['metrics'] = df['metrics'].str.replace(',', '.')

df['metrics'] = df['metrics'].astype(float)

df_0, df_1 = df[df['group'] == 0], df[df['group'] == 1]
plt.figure(figsize=(15, 8))

sns.distplot(df_0['metrics'])

sns.distplot(df_1['metrics'])
# Разделяю выборку на децили

def split_to_deciles(df):

    deciles_values = np.percentile(df['metrics'].values, np.arange(0, 100, 10))

    deciles_values = np.append(deciles_values, df['metrics'].max())

    deciles = []

    for i in range(1, len(deciles_values)):

        deciles.append(df[(df['metrics'] >= deciles_values[i - 1]) & (df['metrics'] < deciles_values[i])]['metrics'].values)

    return deciles

deciles_0 = split_to_deciles(df_0)

deciles_1 = split_to_deciles(df_1)
def split_to_buckets(data, bucket_count):

    idx = np.arange(0, len(data))

    np.random.shuffle(idx)

    buckets = [data[bucket_idx] for bucket_idx in np.array_split(idx, bucket_count)]

    return buckets
bucket_count = 1000



fig_shape = (5, 2)

fig, axs = plt.subplots(*fig_shape)

fig.set_figheight(25)

fig.set_figwidth(15)



for decile_num in range(len(deciles_0)):

    # Каждый дециль в каждой выборке разбиваю на бакеты и строю распределения средних в бакетах

    buckets_0 = split_to_buckets(deciles_0[decile_num], bucket_count)

    means_0 = [b.mean() for b in buckets_0]

    ci_0 = np.percentile(means_0, [2.5, 97.5])

    

    buckets_1 = split_to_buckets(deciles_1[decile_num], bucket_count)

    means_1 = [b.mean() for b in buckets_1]

    ci_1 = np.percentile(means_1, [2.5, 97.5])

    

    #Рисую графики

    row = decile_num // fig_shape[1]

    column = decile_num - fig_shape[1] * row

    

    sns.distplot(means_0, color='red', axlabel='decile ' + str(decile_num + 1), ax=axs[row][column])

    sns.distplot(means_1, color='blue', axlabel='decile ' + str(decile_num + 1), ax=axs[row][column])