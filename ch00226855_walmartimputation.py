# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Make numpy values easier to read.

np.set_printoptions(precision=3, suppress=True)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



import time
path = "/kaggle/input/walmart-recruiting-store-sales-forecasting/"

dataset = pd.read_csv(path + "train.csv.zip", names=['Store','Dept','Date','weeklySales','isHoliday'],sep=',', header=0)

features = pd.read_csv(path + "features.csv.zip",sep=',', header=0,

                       names=['Store','Date','Temperature','Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4',

                              'MarkDown5','CPI','Unemployment','IsHoliday']).drop(columns=['IsHoliday'])

stores = pd.read_csv(path + "stores.csv", names=['Store','Type','Size'],sep=',', header=0)

dataset = dataset.merge(stores, how='left').merge(features, how='left')



dataset.head()
sales = dataset.groupby(['Store', 'Dept', 'Date'])['weeklySales'].sum().unstack()

print(sales.shape)

sales.head()
sales.isna().sum().hist()
sales.isna().sum(axis=1).hist()
# normalize values

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

data = scaler.fit_transform(sales).astype(np.float32)

sales_scaled = pd.DataFrame(data=data, columns=sales.columns, index=sales.index)

sales_scaled.head()
# Scale it back

data_inv = scaler.inverse_transform(sales_scaled)

sales_inv = pd.DataFrame(data=data_inv, columns=sales.columns, index=sales.index)

sales_inv.head()
# Extract the complete rows

sales_complete = sales_scaled[sales_scaled.isna().sum(axis=1) == 0]

sales_complete.isna().sum(axis=1).hist()
import tensorflow as tf

tf.__version__
no = sales_complete.shape[0] # No

dim = sales_complete.shape[1]

h_dim = dim



# System parameters

batch_size = 128 # mb_size

hint_rate = 0.9 # p_hint

alpha = 10 # loss hyperparameter

train_rate = 0.8

# Generator

def make_generator_model(dim, h_dim):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(h_dim, input_shape=(dim*2,), activation='relu'))

    model.add(tf.keras.layers.Dense(h_dim, activation='relu'))

    model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))

    return model
# Discriminator

def make_discriminator(dim, h_dim):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(h_dim, input_shape=(dim*2,), activation='relu'))

    model.add(tf.keras.layers.Dense(h_dim, activation='relu'))

    model.add(tf.keras.layers.Dense(dim, activation='sigmoid'))

    return model
# Use generator to create an instance

generator = make_generator_model(dim, h_dim)



ind = 123

X_mb_original = sales_complete.iloc[ind, :].to_numpy()

M_mb = np.random.choice([0.0, 1.0], size=dim).reshape([1, -1]).astype(np.float32)

Z_mb = np.random.rand(1, dim) * 0.01

X_mb = X_mb_original * M_mb + Z_mb * (1 - M_mb)

input_gen = np.hstack([X_mb, M_mb])



G_sample = generator(input_gen, training=False)

print(input_gen)
print(X_mb_original)
print(M_mb)
print(G_sample)
# Use the discriminator to detect mask

def binary_sampler(p, rows, cols):

    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])

    binary_random_matrix = (unif_random_matrix < p).astype(np.float32)

    return binary_random_matrix



# Sample hint vectors

H_mb_temp = binary_sampler(hint_rate, 1, dim)

H_mb = M_mb * H_mb_temp

input_disc = np.hstack([G_sample, H_mb])



discriminator = make_discriminator(dim, h_dim)

D_prob = discriminator(input_disc)
print(D_prob)
# GAIN loss

def G_loss(X, M, G_sample, D_prob):

#     X = real_output[:dim]

#     M = real_output[dim:]

    G_loss_temp = -tf.reduce_mean((1-M) * tf.math.log(D_prob + 1e-8))

    MSE_loss = tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

    total_loss = G_loss_temp + alpha * MSE_loss 

    return total_loss



def D_loss(M, D_prob):

    D_loss_temp = -tf.reduce_mean(M * tf.math.log(D_prob + 1e-8) \

                                + (1-M) * tf.math.log(1. - D_prob + 1e-8))

    total_loss = D_loss_temp

    return total_loss
G_loss_curr = G_loss(X_mb, M_mb, G_sample, D_prob)

D_loss_curr = D_loss(M_mb, D_prob)
print(G_loss_curr, D_loss_curr)
# Optimizers

generator_optimizer = tf.keras.optimizers.Adam(1e-4)

discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
@tf.function

def train_step(X, M):

    Z = tf.random.uniform([batch_size, dim])

    

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        X = M * X + (1 - M) * Z

        input_gen = tf.concat([X, M], axis=1)

        G_sample = generator(input_gen, training=True)



        # Combine with observed data

        Hat_X = X * M + G_sample * (1 - M)

        input_disc = tf.concat([Hat_X, M], axis=1)



        D_prob = discriminator(input_disc, training=True)

        

        G_loss_curr = G_loss(X, M, G_sample, D_prob)

        D_loss_curr = D_loss(M, D_prob)

        print("Losses:", G_loss_curr, D_loss_curr)

        

    gradients_gen = gen_tape.gradient(G_loss_curr, generator.trainable_variables)

    gradients_disc = disc_tape.gradient(D_loss_curr, discriminator.trainable_variables)

    

    generator_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

    return G_loss_curr, D_loss_curr
def train(dataset, maskset, epochs):

    for epoch in range(epochs):

        start = time.time()

        print("Epoch:", epoch)

        

        for X, M in zip(dataset, maskset):

#             print(M)

#             print(X * M)

#             print("Shapes:", X.shape, M.shape)

#             G_loss_curr, D_loss_curr = train_step(X, M)

#             print("Losses:", G_loss_curr, D_loss_curr)

            train_step(X, M)
miss_rate = 0.2



data_m = binary_sampler(1-miss_rate, no, dim)

miss_data_x = sales_complete.copy().to_numpy()

miss_data_x[data_m == 0] = 12.34 # np.nan will create error in X*M

train_dataset = tf.data.Dataset.from_tensor_slices(miss_data_x).batch(batch_size, drop_remainder=True)

maskset = tf.data.Dataset.from_tensor_slices(data_m).batch(batch_size, drop_remainder=True)
EPOCHS = 10



train(train_dataset, maskset, EPOCHS)
complete_generation = generator(np.hstack([miss_data_x, data_m])).numpy()

# Remove unnecessary predictions

# complete_generation[data_m == 1] = 0

complete_generation
data_m.shape

complete_generation.shape