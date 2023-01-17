# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print('# of training items: %d' % df_train.shape[0])
print('Columns: %s' % df_train.columns.values)
print('Range of x: %d -> %d' % (df_train['x'].min(), df_train['x'].max()))
print('Range of y: %d -> %d' % (df_train['y'].min(), df_train['y'].max()))
print('Range of accuracy: %d -> %d' % (df_train['accuracy'].min(), df_train['accuracy'].max()))
print('Range of place_id: %d -> %d' % (df_train['place_id'].min(), df_train['place_id'].max()))
print('Range of time: %d -> %d' % (df_train['time'].min(), df_train['time'].max()))
df_train.head(3)
print('# of testing items: %d' % df_test.shape[0])
print('Columns: %s' % df_test.columns.values)
print('Range of x: %d -> %d' % (df_test['x'].min(), df_test['x'].max()))
print('Range of y: %d -> %d' % (df_test['y'].min(), df_test['y'].max()))
print('Range of accuracy: %d -> %d' % (df_test['accuracy'].min(), df_test['accuracy'].max()))
print('Range of time: %d -> %d' % (df_test['time'].min(), df_test['time'].max()))
df_test.head(3)
print('# of places: %d' % df_train['place_id'].unique().size)
# Create a DataFrame to store box area for each place
def create_df_agg(df):
    gr_place = df.groupby(['place_id'])
    df_train_agg = gr_place.agg({'x': [np.min, np.max], 'y': [np.min, np.max], 'time': np.ma.count})
    df_train_agg.insert(2, 'height', df_train_agg['y']['amax'] - df_train_agg['y']['amin'])
    df_train_agg.insert(5, 'width', df_train_agg['x']['amax'] - df_train_agg['x']['amin'])
    return df_train_agg
df_train_agg = create_df_agg(df_train)
df_train_agg.head(5)
df_train_agg['width'].describe()
df_train_agg['height'].describe()
def draw_place(df, place_sets):
    plt.figure(figsize=(12, 6))
    for place_set in place_sets:
        plt.subplot(place_set[1])
        plt.title(place_set[2])
        check_ins = df[df['place_id'] == place_set[0]]
        # Use alpha to reflect accuracy
        rgba_colors = np.zeros((check_ins.shape[0],4))
        rgba_colors[:, 0] = 1.
        rgba_colors[:, 3] = 1/check_ins['accuracy']
        plt.scatter(check_ins['x'], check_ins['y'], color=rgba_colors)
    plt.show()
draw_place(df_train, [[1000474694, 221, 'Area=(0.3084, 0.0470), 77 check-ins'], 
                       [1000017288, 222, 'Area=(0.0524, 0.6727), 95 check-ins'],
                       [1000616752, 223, 'Area=(6.8033, 0.0928), 642 check-ins'],
                       [1000705331, 224, 'Area=(9.8780, 0.6727), 962 check-ins']
                      ])
df_train['accuracy'].value_counts().head(5)
plt.hist(df_train['accuracy'], bins=100, facecolor='red')
plt.show()
# Remove noisy check-ins
df_train1 = df_train[df_train['accuracy'] <= 100]
df_train_agg1 = create_df_agg(df_train1)
df_train_agg1['width'].describe()
df_train_agg['height'].describe()
plt.hist(df_train['time'], bins=218)
plt.title('Histogram of time values')
plt.xlabel('Time values')
plt.ylabel('Frequency')
plt.show()