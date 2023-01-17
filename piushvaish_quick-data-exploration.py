import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore') 
sns.set_style('whitegrid')
pd.set_option('display.max_columns', None) # display all columns
# importing the dataset
types = {'row_id': np.dtype(int),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(int),
         'place_id': np.dtype(int) }
#This will ensure that pandas is loading the data into the right objects (not strings for instance, which can take up a lot of memory)

df_train = pd.read_csv('../input/train.csv',dtype=types, index_col='row_id')
df_test = pd.read_csv('../input/test.csv', index_col='row_id')
df_train.head(3)
print('Reading train data')
print('\nSize of training data: ' + str(df_train.shape))
print('Columns:' + str(df_train.columns.values))
print('Number of places: ' + str(len(list(set(df_train['place_id'].values.tolist())))))
print('\n')
print('dtypes')
print('\n')
print(df_train.dtypes)
print('\n')
print('Info: ')
print('\n')
print(df_train.info)
print('Shape: ')
print('\n')
print(df_train.shape)
print('\n')
print('numerical columns statistcs')
print('\n')
print(df_train.describe())
sampler = np.random.permutation(5)
df_train.take(sampler)
randomSample = df_train.take(np.random.permutation(len(df_train))[:3])
randomSample
nb_total = df_train.place_id.count()
nb_unique = df_train.place_id.drop_duplicates().count()

print('Number place_ids: {}'.format(nb_total))
print('Unique place_ids: {}'.format(nb_unique))
print("Average number of duplicates: %.1f" % (nb_total/nb_unique))
f, axarr = plt.subplots(5, figsize=(15, 25))

sns.distplot(df_train['x'], ax=axarr[0])
sns.distplot(df_train['y'], ax=axarr[1])
sns.distplot(df_train['accuracy'], ax=axarr[2])
sns.distplot(df_train['time'], ax=axarr[3])
sns.distplot(df_train['place_id'], ax=axarr[4])


axarr[0].set_title('x')
axarr[1].set_title('y')
axarr[2].set_title('accuracy')
axarr[3].set_title('time')
axarr[4].set_title('place_id')

plt.tight_layout()
plt.show()
