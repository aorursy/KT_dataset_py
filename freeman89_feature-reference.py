import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
df_train =pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
get_detail = lambda values: (np.min(values), np.max(values), np.mean(values), np.median(values), np.std(values))
def visualise_cont(name, max_percent=None):

    

    plt.figure(figsize=(10,10))

    

    train_values = np.sort(df_train[name].values)

    plt.subplot(2, 1, 1)

    # Labels

    x_label = 'Min: %f, Max: %f, Mean: %f, Median: %f, Std: %f' % get_detail(train_values)

    y_label ='Train Freature ' + name

    

    plt.plot(train_values, '-b')

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    

    test_values = np.sort(df_test[name].values)

    plt.subplot(2, 1, 2)

    # Labels

    x_label = 'Min: %f, Max: %f, Mean: %f, Median: %f, Std: %f' % get_detail(test_values)

    y_label = 'Test Freature ' + name

    

    plt.plot(test_values, '-r')

    plt.xlabel(x_label)

    plt.ylabel(y_label)

    plt.show()
def visualise_cat(name):

    plt.figure(figsize=(10,10))

    plt.subplot(2, 1, 1)

    train_counts = df_train[name].value_counts()

    train_counts.plot(kind='bar', title='Train Feature '+name, color='b')

    plt.subplot(2, 1, 2)

    test_counts = df_test[name].value_counts()

    test_counts.plot(kind='bar', title='Test Feature '+name, color='r')

    plt.show()

    print('Train Feature ' + name)

    print(train_counts)

    print('-')

    print('Test Feature ' + name)

    print(test_counts)
for feature in df_train.columns.values:

    if 'cat' in feature:

        visualise_cat(feature)

    elif 'cont' in feature:

        visualise_cont(feature)
# Loss

plt.figure(figsize=(10,10))

values = np.sort(df_train['loss'].values)

x_label = 'Min: %f, Max: %f, Mean: %f, Median: %f, Std: %f' % get_detail(values)

y_label ='Loss 100% of data'

    

plt.plot(values, '-g')

plt.xlabel(x_label)

plt.ylabel(y_label)
# Loss

plt.figure(figsize=(10,10))

values = np.sort(df_train['loss'].values)[0:int(len(values) * .99)]

x_label = 'Min: %f, Max: %f, Mean: %f, Median: %f, Std: %f' % get_detail(values)

y_label ='Loss 99% of data'

    

plt.plot(values, '-g')

plt.xlabel(x_label)

plt.ylabel(y_label)