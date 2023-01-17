import pandas as pd
import numpy as np
import time

start = time.time()

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

print('Size of training data: ' + str(df_train.shape))
print('Size of testing data:  ' + str(df_test.shape))

print('\nColumns:' + str(df_train.columns.values))

print(df_train.describe())

#print(df_train['place_id'])

print('\nNumber of place ids: ' + str(len(list(set(df_train['place_id'].values.tolist())))))

end = time.time()
print('\nTime Elapsed: ' + str(end-start))
print('Target values:')
print(df_train['place_id'].value_counts())
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from matplotlib.colors import LogNorm

time = df_train['time']
x = df_train['x']
y = df_train['y']
accuracy = df_train['accuracy']

n, bins, patches = plt.hist(time, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Data density at different time dimension')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

n, bins, patches = plt.hist(accuracy, 50, normed=1, facecolor='green', alpha=0.75)
plt.title('Histogram of location accuracy')
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.show()

bins = 20
while bins <=160:
    plt.hist2d(x, y, bins=bins, norm=LogNorm())
    plt.colorbar()
    plt.title('x and y location histogram - ' + str(bins) + ' bins')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    bins = bins * 2
