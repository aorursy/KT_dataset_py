import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import os



base_dir = '../input'

print(os.listdir(base_dir ))
X_train = pd.read_csv(f'{base_dir}/X_train.csv')

X_test = pd.read_csv(f'{base_dir}/X_test.csv')

y_train = pd.read_csv(f'{base_dir}/y_train.csv')
X_train.head()
y_train.head()
print(f"There are {len(y_train['surface'].unique())} unique surfaces.")

print(f"There are {len(y_train['group_id'].unique())} recording sessions.")

print(f"There are {len(y_train['series_id'].unique())} different series to predict!")
train_set = X_train.merge(y_train, on='series_id')

train_set.head()
train_set.groupby('series_id').first()['surface'].value_counts()
train_set.describe()
import matplotlib.pyplot as plt

for col in X_train.drop(['series_id', 'measurement_number'], axis=1).select_dtypes(include=np.number).columns:

    train_set[col]

    plt.figure()

    train_col = X_train[col]

    test_col = X_test[col]

    train_col.name = f'{col}_train'

    test_col.name = f'{col}_test'

    sns.kdeplot(train_col)

    sns.kdeplot(test_col)