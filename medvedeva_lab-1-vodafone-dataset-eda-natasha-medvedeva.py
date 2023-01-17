# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from scipy.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import seaborn as sns
vodafone_subset_6 = pd.read_csv("../input/vodafone-subset-6.csv")
vodafone_subset_6.head(10)
vodafone_subset_6.shape
vodafone_subset_6.info()
# vodafone_subset_6['target'].value_counts()
# vodafone_subset_6.iloc[:, :10].describe().T
# vodafone_subset_6.columns
# vodafone_subset_6.iloc[:10, 101:107]
# vodafone_subset_6.loc[:, 'badoo_volume':'tinder_count']
# vodafone_subset_6[ (vodafone_subset_6['target']==6) & (vodafone_subset_6['calls_count_out_weekends']>4.3480) ]
df = vodafone_subset_6[['ROUM', 'phone_value', 'DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS', 'target']]
df.head()
df.info()
df['target'].value_counts()
df['target'].value_counts(normalize=True)
df['target'].value_counts().plot(kind='bar');
df['ROUM'].value_counts()
df['ROUM'].value_counts(normalize=True)
df['phone_value'].value_counts()
df['phone_value'].value_counts(normalize=True)
df.groupby('ROUM')['phone_value'].mean().plot(kind='bar') 
plt.ylabel('ROUM') # добавляем подпись на оси Оу
plt.show();
pd.crosstab(df['ROUM'], df['phone_value'])
sns.heatmap(pd.crosstab(df['ROUM'], df['phone_value']), 
            cmap="YlGnBu", annot=True, cbar=False);
from scipy.stats import chi2_contingency, fisher_exact
chi2_contingency(pd.crosstab(df['ROUM'], df['phone_value']))
plt.figure(figsize=(20, 8)) # увеличим размер картинки
sns.countplot(y='phone_value', hue='ROUM', data=df);
df[df['ROUM']==1]['phone_value'].value_counts(normalize=True)
df[df['ROUM']==0]['phone_value'].value_counts(normalize=True)
df[['DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS']].describe()
df['DATA_VOLUME_WEEKDAYS'].hist(bins=100)
df['DATA_VOLUME_WEEKENDS'].hist(bins=100)
plt.figure(figsize=(20,6))
sns.boxplot(df['DATA_VOLUME_WEEKDAYS'])
plt.figure(figsize=(20,6))
sns.boxplot(df['DATA_VOLUME_WEEKENDS'])
plt.scatter(df['DATA_VOLUME_WEEKDAYS'], df['DATA_VOLUME_WEEKENDS'])
sns.jointplot(x='DATA_VOLUME_WEEKDAYS', y='DATA_VOLUME_WEEKENDS', data=df);
col = ['target', 'DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS']
sns.pairplot(df[col]);
df.pivot_table(values=['DATA_VOLUME_WEEKDAYS', 'DATA_VOLUME_WEEKENDS'], index=['target'], aggfunc='mean')
df.groupby('target')['DATA_VOLUME_WEEKDAYS'].mean().plot(kind='bar') 
plt.ylabel('DATA_VOLUME_WEEKDAYS') # добавляем подпись на оси Оу
plt.show();
df.groupby('target')['DATA_VOLUME_WEEKENDS'].mean().plot(kind='bar') 
plt.ylabel('DATA_VOLUME_WEEKENDS') # добавляем подпись на оси Оу
plt.show();
df[col].corr(method='spearman')
sns.heatmap(df[col].corr(method='spearman'));
def outliers_indices(feature):
    '''
    Будем считать выбросами все точки, выходящие за пределы трёх сигм.
    '''
    mid = df[feature].mean()
    sigma = df[feature].std()
    return df[(df[feature] < mid - 3*sigma) | (df[feature] > mid + 3*sigma)].index
wrong_weekdays = outliers_indices('DATA_VOLUME_WEEKDAYS')
wrong_weekends = outliers_indices('DATA_VOLUME_WEEKENDS')


out = set(wrong_weekdays) | set(wrong_weekends)

print(len(out))
df.drop(out, inplace=True)
sns.pairplot(df[col]);
df[col].corr(method='spearman')
r = spearmanr(df['DATA_VOLUME_WEEKDAYS'], df['DATA_VOLUME_WEEKENDS'])
print('Spearmanr correlation:', r[0], 'p-value:', r[1])
weekdays_age = spearmanr(df['DATA_VOLUME_WEEKDAYS'], df['target'])
print('weekdays_age:', weekdays_age)
weekends_age = spearmanr(df['DATA_VOLUME_WEEKENDS'], df['target'])
print('weekends_age:', weekends_age)
r = kendalltau(df['DATA_VOLUME_WEEKDAYS'], df['DATA_VOLUME_WEEKENDS'])
print('Kendalltau correlation:', r[0], 'p-value:', r[1])
weekdays_age = kendalltau(df['DATA_VOLUME_WEEKDAYS'], df['target'])
print('weekdays_age:', weekdays_age)
weekends_age = kendalltau(df['DATA_VOLUME_WEEKENDS'], df['target'])
print('weekends_age:', weekends_age)