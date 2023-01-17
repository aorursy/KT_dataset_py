import os

import random

import pandas as pd

import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

from scipy import stats

import seaborn as sns

import requests

from PIL import Image

import imageio

from io import BytesIO

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
"""from tqdm import tqdm_notebook as tqdm

import pandas as pd

import requests

from PIL import Image

import imageio



game_df = pd.read_csv('appstore_games.csv')

game_df = game_df.dropna(subset=['Icon URL', 'Average User Rating'])



imm_arr_list=[]

for url in tqdm(game_df['Icon URL'], total=len(game_df['Icon URL'])):

    res = requests.get(game_df['Icon URL'][0])

    imm_arr = imageio.imread(BytesIO(res.content))

    imm_arr_x, imm_arr_y = int(imm_arr.shape[0] / 2), int(imm_arr.shape[1] / 2)

    imm_arr = np.array(Image.fromarray(imm_arr).resize((imm_arr_x, imm_arr_y)))

    imm_arr_list.append(imm_arr)

    

imm_arr = np.stack(imm_arr_list, axis=0)

np.save('game_imm_arr', imm_arr)"""
game_df = pd.read_csv('../input/17k-apple-app-store-strategy-games/appstore_games.csv')

game_df = game_df.dropna(subset=['Icon URL', 'Average User Rating'])

game_df = game_df.drop_duplicates().reset_index(drop=True)

imm_arr = np.load('../input/game-image-array/game_imm_arr.npy')

game_df['imm_arr'] = list(imm_arr)

game_df['User Rating Count'] = game_df['User Rating Count'].replace(np.nan, 0)

print("Imported image array shape : ", imm_arr.shape)

print("game_df shape : ", game_df.shape)

game_df.head()
print('Unique values in `Average User Ratings`:\n', game_df['Average User Rating'].unique())

print(game_df['Average User Rating'].describe())



fig = plt.figure(facecolor='white')

plt.hist(game_df['Average User Rating'], bins=20)

plt.xlabel('Rating')

plt.ylabel('Frequency')



sns.despine()

plt.show()
numeric_columns = ['Average User Rating', 'User Rating Count', 'Price', 'Size']

low_rated_games = game_df[game_df['Average User Rating'] <= 2.5][numeric_columns]

high_rated_games = game_df[game_df['Average User Rating'] >= 5][numeric_columns]



print("Poorly Rated Games (Rated 2 or below):")

print(low_rated_games.describe())

print("Well Rated Games (Rated 4 or above):")

print(high_rated_games.describe())



print('\nP-Values of differences between means in `numeric_columns` : ')

for col in numeric_columns:

    p_value_col = stats.ttest_ind(low_rated_games[col],high_rated_games[col])[1]

    print(f'\t\"{col}\" : ', p_value_col)
n = 4



fig, axs = plt.subplots(n, n, figsize= (10, 10), facecolor='white')



for i in range(0, n):

    for j in range(0, n):

        if abs(i - j) < 2:

            random_test_df = game_df[game_df['Average User Rating'] < 2]

            random_test_number = random_test_df.sample().reset_index()['index'][0]

        else:

            random_test_number = random.randint(0, len(game_df))

    

        display_url = game_df.loc[random_test_number, 'Icon URL']

        display_rating = game_df.loc[random_test_number, 'Average User Rating']



        res = requests.get(display_url)

        display_imm_arr = imageio.imread(BytesIO(res.content))

        axs[i][j].imshow(display_imm_arr, interpolation='nearest')

        axs[i][j].set_title(f"Rating: {display_rating}")

        

fig.tight_layout()

plt.show()

scaler = MinMaxScaler()



### Creating a series of the average ratings

### Will become target series

rating_series = game_df['Average User Rating']



### Creating array of non-image features

### These are transformed using the `MinMaxScaler()`

feats = game_df[['User Rating Count', 'Size']].replace(np.nan, 0)

feats_array = scaler.fit_transform(feats)



### Reshape all the image arrays into the same 4D shape for the Conv2D layer

imm_arr = imm_arr.reshape(imm_arr.shape[0], 128, 128, 3) / 255

print("Features Array shape : \t", imm_arr.shape)

input_shape = (128, 128, 3)
x_imm_train, x_imm_test, y_train, y_test = train_test_split(imm_arr, rating_series, test_size = .2)



train_ind = y_train.index

test_ind = y_test.index



### Convert y_* to arrays as we've extracted the useful indexes already

y_train = np.asarray(y_train)

y_train = y_train.reshape(-1,1)

y_test = np.asarray(y_test)

y_test = y_test.reshape(-1,1)



### MinMax scale the y_train and y_test arrays to get the Ratings within the range a NN can predict

y_train = scaler.fit_transform(y_train)

y_test = scaler.fit_transform(y_test)



### Create arrays of the non-image features using the derived index lists

x_feats_train = np.asarray(feats_array[train_ind])

x_feats_test = np.asarray(feats_array[test_ind])



x_feats_train = scaler.fit_transform(x_feats_train)

x_feats_test = scaler.fit_transform(x_feats_test)



print("x_feats_train shape : \t", x_feats_train.shape)

print("x_feats_test shape : \t", x_feats_test.shape)

print("x_imm_train shape : \t", x_imm_train.shape)

print("x_imm_test shape : \t", x_imm_test.shape)

print("y_train shape : \t", y_train.shape)

print("y_test shape : \t\t", y_test.shape)
def build_image_model(filters, input_shape, initializer = tf.keras.initializers.he_normal()):

    ### One input layer for each stream of data

    input_feats = tf.keras.layers.Input(shape=(2,), name='feat_input')

    input_image = tf.keras.layers.Input(shape=input_shape, name='image_input')

    ### Reduce the images

    conv_i_1 = tf.keras.layers.Conv2D(filters, 2, 2, activation='relu', padding="same", kernel_initializer = initializer)(input_image)

    max_i_1 = tf.keras.layers.MaxPool2D(pool_size=(2,2))(conv_i_1)

    flatten_i = tf.keras.layers.Flatten()(max_i_1)

    ### Merged the reduced image with the numeric features

    merged = tf.keras.layers.concatenate([flatten_i, input_feats], axis=-1)

    flatten1 = tf.keras.layers.Flatten()(merged)

    dropout2 = tf.keras.layers.Dropout(.4)(flatten1)

    dense2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer = initializer)(dropout2)

    output = tf.keras.layers.Dense(1, activation='linear')(dense2)

    model = tf.keras.models.Model(inputs=[input_feats, input_image], outputs=output)

    return model
model = build_image_model(filters=64, input_shape = input_shape)



model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='mean_squared_error', metrics=['mean_absolute_error'])

model.summary()



earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)



callbacks_list = [earlystop_callback]



history = model.fit([x_feats_train, x_imm_train], y_train, epochs=150, batch_size=15, validation_split = .2, callbacks=callbacks_list)
n = 4



fig, axs = plt.subplots(n, n, figsize= (10, 10), facecolor='white')



for i in range(0, n):

    for j in range(0, n):

        if abs(i - j) < 2:

            random_test_df = game_df[game_df['Average User Rating'] < 2]

            random_test_number = random_test_df.sample().reset_index()['index'][0]

        else:

            random_test_number = random.randint(0, len(game_df))

    

        display_arr = imm_arr[random_test_number]

        display_title = game_df.loc[random_test_number, 'Name']

        display_rating = game_df.loc[random_test_number, 'Average User Rating']

        display_rating_count = game_df.loc[random_test_number, 'User Rating Count']

        display_size = game_df.loc[random_test_number, 'Size']

        dis_feats_arr = np.asarray((display_rating_count, display_size)).reshape(1,2)

        

        dis_feats_arr = scaler.fit_transform(dis_feats_arr)

        

        pred = round(model.predict([dis_feats_arr, display_arr.reshape(1, 128, 128, 3)])[0][0] * 5, 3)

        print(f"Game title: {display_title}, \nUser Rating Count: {display_rating_count}, \nSize: {display_size}, \nActual Rating: {display_rating}")

        print(f"Predicted Rating: {pred}\n\n")

        

        axs[i][j].imshow(display_arr, interpolation='nearest')

        axs[i][j].set_title(f"Rating: {display_rating}\nPredicted Rating: {pred}")

        

fig.tight_layout()

plt.show()

n=50



index_org = game_df.loc[:, 'Average User Rating'].sort_values().index[:n]



check_df = game_df.loc[index_org, ['User Rating Count', 'Size', 'imm_arr', 'Average User Rating']]

pred_list = []

for index, row in check_df.iterrows():

    row_feats = np.asarray([(row['User Rating Count'], row['Size'])]).reshape(-1,2)

    reshaped_imm = row['imm_arr'].reshape(1, 128, 128, 3) / 255

    row_feats = scaler.fit_transform(row_feats)

    pred_list.append((model.predict([row_feats, reshaped_imm]) * 5)[0][0])

check_df['pred'] = pred_list

check_df = check_df.drop('imm_arr', axis=1)



check_df
