import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/hanziDB.csv')

print('data shape', data.shape)

print('unique char length', len(data['charcter'].unique()))

data.sort_values('frequency_rank').head(10)
# data_copy = data.copy()

data = data.drop_duplicates()

print('data shape', data.shape)

print('unique char length', len(data['charcter'].unique()))

data.sort_values('frequency_rank').head(10)
import matplotlib

print('test chinese characters')

plt.xlabel(u"横坐标xlabel")
def dist_by_radical(df, by='radical', top_n_rank=100):

    grp = df[df['frequency_rank']<=top_n_rank].groupby(by)

    res = pd.DataFrame(grp['charcter'].apply(len))

    res.columns = ['length']

    res['median_rank'] = grp['frequency_rank'].apply(np.median)

    res['mean_rank'] = grp['frequency_rank'].apply(np.mean)

    res['mean_median_avg'] = (res['median_rank'] + res['mean_rank']) / 2.

    res['chars'] = grp['charcter'].apply(sum)

    res['radical'] = res.index

    res.index = range(len(res))

    return res

def bar_plot(res, top_n_rank):

    plt.figure(figsize=(12, 8))

    plt.bar(res.index, res.mean_rank)

    plt.bar(res.index, res.median_rank)

    plt.legend(['mean', 'median'])

    plt.xlabel('radicals')

    plt.ylabel('mean & median rank of chars')

    plt.title('top ' + str(top_n_rank))

    plt.show()

#res = dist_by_radical(data)

for top_n_rank in [100, 500, 1000]:

    res = dist_by_radical(data, top_n_rank=top_n_rank)

    print(list(res.radical))

    bar_plot(res, top_n_rank)

dist_by_radical(data, top_n_rank=100).head(30)
dist_by_radical(data, top_n_rank=500).head(30)
dist_by_radical(data, top_n_rank=1000).head(50)
try:

    data['stroke_count'] = data['stroke_count'].astype(np.int)

except Exception as e:

    print(e)
print(data[data['stroke_count']=='8 9'])
data.loc[804, 'stroke_count'] = '8'

data['stroke_count'] = data['stroke_count'].astype(np.int)

data['stroke_count'].dtype
data[['stroke_count']].plot(figsize=(10, 8))

plt.xlabel('frequency_rank')

plt.ylabel('stroke_count')

plt.show()

sns.jointplot(data[['frequency_rank']].values, data[['stroke_count']].values,

              size=10, color='g')

plt.show()
hsk_strk = data[['hsk_levl', 'stroke_count']].dropna()

hsk_strk.head(20)
plt.figure(figsize=(10, 6))

sns.violinplot(x='hsk_levl', y=hsk_strk['stroke_count'].values,

               data=hsk_strk, size=10, color='g')

plt.ylabel('stroke_count')

plt.show()