import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import seaborn as sns

sns.set_style('whitegrid')

import sklearn

import tensorflow as tf

from tensorflow import keras
data_path = '../input/kobe-bryant-shot-selection/data.csv.zip'



df = pd.read_csv(data_path)

df.head()
cols_to_drop = ['game_id', 'game_event_id', 'lat', 'lon', 'team_id', 'team_name', 'matchup', 'game_date']



df = df.drop(cols_to_drop, axis=1)

df
test_data = df.loc[df['shot_made_flag'].isnull()]

test_data.head()
df = df[df['shot_made_flag'].notna()]

df = df.drop('shot_id', axis=1)



df.head()
corr = df.corr()

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5)
corr
cols_to_drop = ['loc_x', 'playoffs']



df = df.drop(cols_to_drop, axis=1)

df.head()
df['action_type'].value_counts()
df['combined_shot_type'].value_counts()
df = df.drop('action_type', axis=1)
df['time_remainig'] = df['minutes_remaining'] * 60 + df['seconds_remaining']

df.head()
df = df.drop(['minutes_remaining', 'seconds_remaining'], axis=1)

df.head()
def category_feature_importance(feature, target, col_wrap=4):

    print(df[feature].value_counts())

    sns.catplot(target, col=feature, col_wrap=col_wrap, data=df, 

        kind="count", height=4, aspect=.8)
category_feature_importance('shot_type', 'shot_made_flag')
category_feature_importance('shot_zone_area', 'shot_made_flag', 3)
category_feature_importance('shot_zone_basic', 'shot_made_flag')
category_feature_importance('opponent', 'shot_made_flag', 5)
category_feature_importance('season', 'shot_made_flag')
df.head()
df.groupby('shot_made_flag').mean()['loc_y'].plot(kind='bar')
sns.stripplot("shot_distance", data=df)  # Above 70 shot distance
sns.stripplot("loc_y", data=df)  # Above 600 shot distance
filt = (df['loc_y'] < 600) & (df['shot_distance'] < 70)

df = df[filt]
sns.stripplot("loc_y", data=df)
def cols_to_convert_to_int(df, cols):

    def categorical_to_int(series):

        temp = {x: i for i, x in enumerate(series.unique())}

        series = series.apply(lambda x:temp[x])

        return series

    for col in cols:

        df[col] = categorical_to_int(df[col])
cols_to_convert_to_int(df, ['combined_shot_type', 'season', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent'])

df.head()
# Scale the df

df = (df - df.mean())/df.std()

df.head()
df = df.sample(frac = 1) 
all_labels = df['shot_made_flag']

all_data = df.drop('shot_made_flag', axis=1)



assert(len(all_labels) == len(all_data))
from keras.utils import to_categorical



train_size = int((80/100) * df.shape[0])



train_data = all_data[: train_size].values

train_labels = to_categorical(all_labels[: train_size].values)



valid_data = all_data[train_size: ].values

valid_labels = to_categorical(all_labels[train_size: ].values)



# np.random.seed(42)

# np.random.shuffle(train_data)



assert(len(train_data) == len(train_labels))

assert(len(valid_data) == len(valid_labels))
print(train_labels[: 2])

print(train_data[: 2])
model = keras.models.Sequential([

    keras.layers.Dense(32),

    keras.layers.Dropout(0.5),

    keras.layers.LeakyReLU(),

    keras.layers.Dense(2, activation='softmax'),

])



model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
my_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)



history = model.fit(train_data, train_labels, epochs=200, 

              validation_data=(valid_data, valid_labels),

              callbacks=[my_cb])
print(history.history.keys())

epochs = len(history.history['loss'])

epochs
y1 = history.history['loss']

y2 = history.history['val_loss']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['loss', 'val_loss'])

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.tight_layout()
y1 = history.history['acc']

y2 = history.history['val_acc']

x = np.arange(1, epochs+1)



plt.plot(x, y1, y2)

plt.legend(['acc', 'val_acc'])

plt.xlabel('Epochs')

plt.ylabel('Acc')

plt.tight_layout()
res = model.evaluate(valid_data, valid_labels)
df.head()
test_data['time_remaining'] = test_data['minutes_remaining'] * 60 + test_data['seconds_remaining']

test_data = test_data.drop(['action_type', 'loc_x', 'playoffs', 'minutes_remaining', 'seconds_remaining'], axis=1)

cols_to_convert_to_int(test_data, ['combined_shot_type', 'season', 'shot_type', 'shot_zone_area', 'shot_zone_basic', 'shot_zone_range', 'opponent'])

test_data.head()
id_nums = test_data['shot_id']

test_data = test_data.drop('shot_id', axis=1)
test_data = test_data.drop('shot_made_flag', axis=1)

test_data.head()
test_data = (test_data - test_data.mean())/test_data.std()

test_data.head()
test = test_data.values

test[: 2]
assert(len(train_data[0]) == len(test[0]))
ans = model.predict(test)
answers = []

for entry in ans:

    answers.append(entry[1])
final = pd.DataFrame({'shot_id': id_nums, 'shot_made_flag': answers})
final.head()
kaggle_output_path = './submission.csv'

final.to_csv(kaggle_output_path, index=False)