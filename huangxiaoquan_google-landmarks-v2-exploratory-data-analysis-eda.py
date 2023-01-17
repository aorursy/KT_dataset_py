import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import os

print(os.listdir("../input"))
train_file_path = '../input/google-landmarks-dataset-v2/train.csv'

index_file_path = '../input/google-landmarks-dataset-v2/index.csv'

test_file_path = '../input/google-landmarks-dataset-v2/test.csv'
df_train = pd.read_csv(train_file_path)

df_index = pd.read_csv(index_file_path)

df_test = pd.read_csv(test_file_path)
print("Training data size:", df_train.shape)

print("Training data columns:",df_train.columns)

print(df_train.info())
df_train.head(3)
df_train.sample(3).sort_index()
df_train.tail(3)
select = [4444, 10000, 14005]

df_train.iloc[select,:]
print("Index data size", df_index.shape)

print(df_index.columns)

print(df_index.info())

df_index.head(3)
print("Test data size", df_test.shape)

print(df_test.columns)

print(df_test.info())

df_test.head(3)
print('data is None:')

missing = df_train.isnull().sum()

percent = missing/df_train.count()

missing_train_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_train_data.head()
print('data is None:')

missing = df_index.isnull().sum()

percent = missing/df_index.count()

missing_index_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_index_data.head()
print('data is None:')

missing = df_test.isnull().sum()

percent = missing/df_test.count()

missing_test_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_test_data.head()
print('data is \'None\':')

missing = (df_train == 'None').sum()

percent = missing/df_train.count()

missing_train_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_train_data.head()
print('data is \'None\':')

missing = (df_index == 'None').sum()

percent = missing/df_index.count()

missing_index_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_index_data.head()
print('data is \'None\':')

missing = (df_test == 'None').sum()

percent = missing/df_test.count()

missing_test_data = pd.concat([missing, percent], axis=1, keys=['Missing', 'Percent'])

missing_test_data.head()
df_train['landmark_id'].describe()
sns.set()

print(df_train.nunique())

df_train['landmark_id'].value_counts().hist()
sns.set()

# plt.figure(figsize = (8, 5))

plt.title('Landmark_id Distribuition')

sns.distplot(df_train['landmark_id'])
sns.set()

plt.title('Training set: number of images per class(line plot)')

sns.set_color_codes("pastel")

landmarks_fold = pd.DataFrame(df_train['landmark_id'].value_counts())

landmarks_fold.reset_index(inplace=True)

landmarks_fold.columns = ['landmark_id','count']

ax = landmarks_fold['count'].plot(logy=True, grid=True)

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
sns.set()

# plt.title('Training set: number of images per class')

landmarks_fold_sorted = pd.DataFrame(df_train['landmark_id'].value_counts())

landmarks_fold_sorted.reset_index(inplace=True)

landmarks_fold_sorted.columns = ['landmark_id','count']

landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')

ax = landmarks_fold_sorted.plot.scatter(\

     x='landmark_id',y='count',

     title='Training set: number of images per class(statter plot)')

locs, labels = plt.xticks()

plt.setp(labels, rotation=30)

ax.set(xlabel="Landmarks", ylabel="Number of images")
sns.set()

ax = landmarks_fold_sorted.boxplot(column='count')

ax.set_yscale('log')
sns.set()

res = stats.probplot(df_train['landmark_id'], plot=plt)
threshold = [2, 3, 5, 10, 20, 50, 100]

for num in threshold:    

    print("Number of classes under {}: {}/{} "

          .format(num, (df_train['landmark_id'].value_counts() < num).sum(), 

                  len(df_train['landmark_id'].unique()))

          )
temp = pd.DataFrame(df_train.landmark_id.value_counts().head(10))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id', 'count']

temp
sns.set()

# plt.figure(figsize=(9, 8))

plt.title('Most frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
temp = pd.DataFrame(df_train.landmark_id.value_counts().tail(10))

temp.reset_index(inplace=True)

temp.columns = ['landmark_id', 'count']

temp
sns.set()

# plt.figure(figsize=(9, 8))

plt.title('Least frequent landmarks')

sns.set_color_codes("pastel")

sns.barplot(x="landmark_id", y="count", data=temp,

            label="Count")

locs, labels = plt.xticks()

plt.setp(labels, rotation=45)

plt.show()
# Extract site_names for train data

temp_list = list()

for path in df_train['url']:

    temp_list.append((path.split('//', 1)[1]).split('/', 1)[0])

df_train['site_name'] = temp_list

data_sources = pd.DataFrame(df_train['site_name'].value_counts())

data_sources.reset_index(inplace=True)

data_sources.columns = ['site_name', 'count']

data_sources.head()
# Plot the Sites with their count

sns.set()

data_sources.plot.bar(x="site_name", y="count", rot=0,

                      title='Sites with their count')