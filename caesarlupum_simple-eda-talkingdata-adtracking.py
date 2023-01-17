import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import mlcrate as mlc

import os

import gc

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



pal = sns.color_palette()



print('# File sizes')

for f in os.listdir('../input'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/' + f) / 1000000, 2)) + 'MB')
import subprocess

print('# Line count:')

for file in ['train.csv', 'test.csv', 'train_sample.csv']:

    lines = subprocess.run(['wc', '-l', '../input/{}'.format(file)], stdout=subprocess.PIPE).stdout.decode('utf-8')

    print(lines, end='', flush=True)
df_train = pd.read_csv('../input/train.csv', nrows=1000000)

df_test = pd.read_csv('../input/test.csv', nrows=1000000)
print('Training set:')

df_train.head()
print('Test set:')

df_test.head()
features = df_train.columns.values[0:30]

unique_max_train = []

unique_max_test = []

for feature in features:

    values = df_train[feature].value_counts()

    perc = values.max() / ( df_train.shape[0]*100 )

    unique_max_train.append([feature, values.max(), values.idxmax(), perc])

    

np.transpose((pd.DataFrame(unique_max_train, columns=['Feature', 'Max duplicados', 'Valor', 'Percentage'])).\

            sort_values(by = 'Max duplicados', ascending=False).head(15))
plt.figure(figsize=(15, 8))

cols = ['ip', 'app', 'device', 'os', 'channel']

uniques = [len(df_train[col].unique()) for col in cols]

sns.set(font_scale=1.2)

ax = sns.barplot(cols, uniques, palette=pal, log=True)

ax.set(xlabel='Feature', ylabel='log(unique count)', title='Number of unique values per feature')

for p, uniq in zip(ax.patches, uniques):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height + 10,

            uniq,

            ha="center") 

# for col, uniq in zip(cols, uniques):

#     ax.text(col, uniq, uniq, color='black', ha="center")
# checking missing data

total = df_train.isnull().sum().sort_values(ascending = False)

percent = (df_train.isnull().sum()/df_train.isnull().count()*100).sort_values(ascending = False)

missing__train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing__train_data.head(10)
# !pip install quilt
# import missingno as msno

# msno.matrix(df_train.head(20000))
# msno.heatmap(df_train)
for col, uniq in zip(cols, uniques):

    counts = df_train[col].value_counts()



    sorted_counts = np.sort(counts.values)

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    line, = ax.plot(sorted_counts, color='red')

    ax.set_yscale('log')

    plt.title("Distribution of value counts for {}".format(col))

    plt.ylabel('log(Occurence count)')

    plt.xlabel('Index')

    plt.show()

    

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1)

    plt.hist(sorted_counts, bins=50)

    ax.set_yscale('log', nonposy='clip')

    plt.title("Histogram of value counts for {}".format(col))

    plt.ylabel('Number of IDs')

    plt.xlabel('Occurences of value for ID')

    plt.show()

    

    max_count = np.max(counts)

    min_count = np.min(counts)

    gt = [10, 100, 1000]

    prop_gt = []

    for value in gt:

        prop_gt.append(round((counts > value).mean()*100, 2))

    print("Variable '{}': | Unique values: {} | Count of most common: {} | Count of least common: {} | count>10: {}% | count>100: {}% | count>1000: {}%".format(col, uniq, max_count, min_count, *prop_gt))

    
plt.figure(figsize=(8, 8))

sns.set(font_scale=1.2)

mean = (df_train.is_attributed.values == 1).mean()

ax = sns.barplot(['Fraudulent (1)', 'Not Fradulent (0)'], [mean, 1-mean], palette=pal)

ax.set(xlabel='Target Value', ylabel='Probability', title='Target value distribution')

for p, uniq in zip(ax.patches, [mean, 1-mean]):

    height = p.get_height()

    ax.text(p.get_x()+p.get_width()/2.,

            height+0.01,

            '{}%'.format(round(uniq * 100, 2)),

            ha="center") 
corrs = df_train.corr()

corrs
plt.figure(figsize = (20, 8))



# Heatmap of correlations

sns.heatmap(corrs, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');