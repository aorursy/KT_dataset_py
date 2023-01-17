import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

df_train = pd.read_csv('../input/train.csv', nrows=500000)
target = df_train['Demanda_uni_equil'].tolist()
df_train = pd.read_csv('../input/train.csv', nrows=500000)
df_test = pd.read_csv('../input/test.csv', nrows=500000)

nl()
print('Size of training set: ' + str(df_train.shape))
print(' Size of testing set: ' + str(df_test.shape))

nl()
print('Columns in train: ' + str(df_train.columns.tolist()))
print(' Columns in test: ' + str(df_test.columns.tolist()))

nl()
print(df_train.describe())
target = df_train['Demanda_uni_equil'].tolist()

def label_plot(title, x, y):
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)

plt.hist(target, bins=200, color='blue')
label_plot('Distribution of target values', 'Demanda_uni_equil', 'Count')
plt.show()

print("Looks like we have some pretty big outliers, let's zoom in and try again")

print('Data with target values under 50: ' + str(round(len(df_train.loc[df_train['Demanda_uni_equil'] <= 50]) / 5000, 2)) + '%')

plt.hist(target, bins=50, color='blue', range=(0, 50))
label_plot('Distribution of target values under 50', 'Demanda_uni_equil', 'Count')
plt.show()

from collections import Counter
print(Counter(target).most_common(10))
print('Our most common value is 2')

sub = pd.read_csv('../input/sample_submission.csv')
sub['Demanda_uni_equil'] = 2
sub.to_csv('mostcommon.csv', index=False)
pseudo_time = df_train.loc[df_train.Demanda_uni_equil < 20].index.tolist()
target = df_train.loc[df_train.Demanda_uni_equil < 20].Demanda_uni_equil.tolist()

plt.hist2d(pseudo_time, target, bins=[50,20])
label_plot('Histogram of target value over index', 'Index', 'Target')
plt.show()
semana = df_train['Semana']
print(semana.value_counts())
print('\nIt looks like by sampling only the first 500,000 columns, we have only sampled from week 3.\nWe will have to take a larger portion of the dataset\n')

timing = pd.read_csv('../input/train.csv', usecols=['Semana','Demanda_uni_equil'])
print('Size: ' + str(timing.shape))

print(timing['Semana'].value_counts())
plt.hist(timing['Semana'].tolist(), bins=7, color='red')
label_plot('Distribution of weeks in data', 'Semana', 'Frequency')