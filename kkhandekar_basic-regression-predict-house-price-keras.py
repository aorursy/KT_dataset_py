# tensorflow_docs library installation from github

!pip install -q git+https://github.com/tensorflow/docs
# Generic Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pathlib



#SKLearn Library

from sklearn import preprocessing



#Plotting Libraries

import matplotlib.pyplot as plt

import seaborn as sns



#TensorFlow/Keras Libraries

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import layers



import tensorflow_docs as tfdocs

import tensorflow_docs.plots

import tensorflow_docs.modeling
#Data

url =  '../input/house-price/Housing_Modified.csv'

data = pd.read_csv(url, header='infer')
print("Total Records: ",data.shape[0])
#Inspect

data.head()
#Check for missing value

data.isna().sum()
#Stat Summary of numeric columns

data[['price','lotsize','bedrooms','bathrms']].describe()
# Function to plot histogram

def plot_sumry(dataframe,col1,col2):

    #plt.figure(figsize=(10,5))

    sns.set_palette('pastel')

    sns.set_color_codes()

    fig = plt.figure(figsize=(15, 15))

    plt.subplots_adjust(hspace = 0.9)

    

    plt.subplot(221)

    ax1 = sns.distplot(data[col1], color = 'midnightblue')

    plt.title(f'{col1.capitalize()} Distribution', fontsize=15)

    

    plt.subplot(222)

    ax1 = sns.distplot(data[col2], color = 'midnightblue')

    plt.title(f'{col2.capitalize()} Distribution', fontsize=15)

    

   
#Distribution Summary of House Price

plot_sumry(data, 'price','lotsize')
# Function to plot Count Plot

def count_plot(dataframe,col1,col2):

    sns.set_palette('pastel')

    sns.set_color_codes()

    fig = plt.figure(figsize=(15, 15))

    plt.subplots_adjust(hspace = 0.9)

    

    plt.subplot(221)

    ax1 = sns.countplot(x=col1, color = 'cornflowerblue',data=dataframe)

    plt.title(f'Number of Houses per {col1.capitalize()}', fontsize=15)

    

    plt.subplot(222)

    ax1 = sns.countplot(x=col2, color = 'cornflowerblue',data=dataframe)

    plt.title(f'Number of Houses per {col2.capitalize()}', fontsize=15)

    

   
#Plotting a count plot

count_plot(data,'bedrooms','bathrms')
#Creating a list of categorical columns

cat_col = data.select_dtypes(include=['object']).columns



#Instantiating LabelEncoder Object

lb_encode = preprocessing.LabelEncoder()



#Iterating over the above list & encoding the columns

for i in cat_col:

    data[i] = lb_encode.fit_transform(data[i])

#Inspect

data.head()
train_ds = data.sample(frac=0.9,random_state=0)

test_ds = data.drop(train_ds.index)
print("Records in Training: ",train_ds.shape[0], " ", "Records in Testing: ", test_ds.shape[0])
#Split features from labels aka target

train_label = train_ds.pop('price')

test_label = test_ds.pop('price')
def build_model():

    model = keras.Sequential([

    layers.Dense(64, activation='relu', input_shape=[len(train_ds.keys())]),

    layers.Dense(64, activation='relu'),

    layers.Dense(1)

    ])

    

    optimizer = tf.keras.optimizers.RMSprop(0.05)

    

    model.compile(loss='mse',

                optimizer=optimizer,

                metrics=['mae', 'mse'])

    

    return model





#Defining Model

model = build_model()
#Inspect Model

model.summary()
EPOCHS = 1 



history = model.fit(train_ds, train_label,epochs=EPOCHS, validation_split = 0.1, verbose=0, callbacks=[tfdocs.modeling.EpochDots()])
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.head()
loss, mae, mse = model.evaluate(test_ds, test_label, verbose=0)



print("Testing Data Mean Abs Error: {:.2f}".format(mae))
test_pred = model.predict(test_ds).flatten()
error = test_pred - test_label

plt.hist(error, bins = 25)

plt.xlabel("Prediction Error [HousePrice]")

_ = plt.ylabel("Count")
#Add a new column to test dataset

test_ds['HousePrice_Pred'] = test_pred



#Inspect

test_ds.head()
#Saving the test dataset to Output

test_ds.to_csv('PredictedHousePrice.csv',index=False)