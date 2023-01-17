# Imports



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sbn

%matplotlib inline



churn_data_path = '../input/kkbox-churn-prediction-challenge/'

recommend_data_path = '../input/kkbox-music-recommendation-challenge/'
df_members = pd.read_csv(recommend_data_path + 'members.csv')

members = pd.DataFrame(df_members['msno'])
user_data = pd.DataFrame()

for chunk in pd.read_csv(churn_data_path + 'user_logs.csv', chunksize=500000):

    merged = members.merge(chunk, on='msno', how='inner')

    user_data = pd.concat([user_data, merged])
# Almost all members have additional information now

print (str(len(members['msno'].unique())) + " unique members in Music Recommendation Challenge")

print (str(len(user_data['msno'].unique())) + " users now have additional information")
user_data.to_csv('user_logs2.csv', index=False)
print (user_data.head())
for col in user_data.columns[1:]:

    outlier_count = user_data['msno'][user_data[col] < 0].count()

    print (str(outlier_count) + " outliers in column " + col)

user_data = user_data[user_data['total_secs'] >= 0]

print (user_data['msno'][user_data['total_secs'] < 0].count())
del user_data['date']



print (str(np.shape(user_data)) + " -- Size of data large due to repeated msno")

counts = user_data.groupby('msno')['total_secs'].count().reset_index()

counts.columns = ['msno', 'days_listened']

sums = user_data.groupby('msno').sum().reset_index()

user_data = sums.merge(counts, how='inner', on='msno')



print (str(np.shape(user_data)) + " -- New size of data matches unique member count")

print (user_data.head())
df_train = pd.read_csv(recommend_data_path + 'train.csv')

train = df_train.merge(user_data, how='left', on='msno')



def repeat_chance_plot(groups, col, plot=False):

    x_axis = [] # Sort by type

    repeat = [] # % of time repeated

    for name, group in groups:

        count0 = float(group[group.target == 0][col].count())

        count1 = float(group[group.target == 1][col].count())

        percentage = count1/(count0 + count1)

        x_axis = np.append(x_axis, name)

        repeat = np.append(repeat, percentage)

    plt.figure()

    plt.title(col)

    sbn.barplot(x_axis, repeat)



for col in user_data.columns[1:]:

    tmp = pd.DataFrame(pd.qcut(train[col], 15, labels=False))

    tmp['target'] = train['target']

    groups = tmp.groupby(col)

    repeat_chance_plot(groups, col)
corrmat = user_data[user_data.columns[1:]].corr()

f, ax = plt.subplots(figsize=(12, 9))

sbn.heatmap(corrmat, vmax=1, cbar=True, annot=True, square=True);

plt.show()
del user_data['num_75']

del user_data['num_unq']
from sklearn.preprocessing import StandardScaler



cols = user_data.columns[1:]

log_user_data = user_data.copy()

log_user_data[cols] = np.log1p(user_data[cols])

ss = StandardScaler()

log_user_data[cols] = ss.fit_transform(log_user_data[cols])



for col in cols:

    plt.figure(figsize=(15,7))

    plt.subplot(1,2,1)

    sbn.distplot(user_data[col].dropna())

    plt.subplot(1,2,2)

    sbn.distplot(log_user_data[col].dropna())

    plt.figure()
log_user_data.to_csv('user_logs_final.csv', index=False)