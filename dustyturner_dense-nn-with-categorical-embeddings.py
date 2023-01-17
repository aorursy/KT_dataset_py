# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



from tensorflow import keras

from tensorflow.keras import regularizers

from tensorflow.keras.layers import Concatenate, Dense, Dropout, Embedding, Input, Reshape

import matplotlib.pyplot as plt



%matplotlib inline



pd.options.display.float_format = '{:.4f}'.format

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data = pd.read_pickle('../input/nn-data/train_data_nn.pkl')

test_data = pd.read_pickle('../input/nn-data/test_data_nn.pkl')

train_data.info()
def build_basic_model():

    # Specify the number of different elements in the categorical feature and 

    # the length of the embedding vector these elements are going to be converted to.

    # This information is required when defining an embedding layer.

    elements_in_category = 10

    embedding_size = int(min(elements_in_category / 2, 50))



    # create an input for the categorical feature

    categorical_input = Input(shape=(1,))

    

    # create an input for the remaining numerical data

    numerical_input = Input(shape=(9,))    

    

    # crate an embedding layer for the categorical feature

    category_embedding = Embedding(elements_in_category, 

                                   embedding_size, 

                                   input_length=1)(categorical_input)

    category_embedding = Reshape(target_shape=(embedding_size,))(category_embedding)



    # concatenate the embedding values from the categorical input 

    # and the numerical inputs together

    inputs = Concatenate(axis=-1)([category_embedding, numerical_input])



    # create a basic 100 node MLP with a single node regression output

    dense_layer = Dense(100, activation='relu', 

                        kernel_regularizer=regularizers.l2(0.01))(inputs)

    output_layer = Dense(1, kernel_regularizer=regularizers.l2(0.01))(dense_layer)   



    # build the model

    model = keras.Model(inputs=[categorical_input, numerical_input], outputs=output_layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    model.compile(loss='mse',

                  optimizer=optimizer,

                  metrics=['mae', 'mse'])

    

    return model
from sklearn.preprocessing import OrdinalEncoder



cat_features = ['date_block_num', 'item_id', 'shop_id', 'item_category_id', 'months_since_item_first_sale',

                'months_since_last_sale', 'month', 'month_length']



train_test_cat_features = pd.concat([train_data[cat_features], test_data[cat_features]])



enc = OrdinalEncoder().fit(train_test_cat_features)



train_data_cat_features = pd.DataFrame(enc.transform(train_data[cat_features]),

                                       columns=cat_features)

test_data_cat_features = pd.DataFrame(enc.transform(test_data[cat_features]),

                                      columns=cat_features)
# we will be using the last month of training data as a validation set to train the NN

val_mask = train_data_cat_features.date_block_num == (train_data_cat_features.date_block_num.max())



# Features that aren't to be included in the training data

# 'item_cnt_month' is the target and ID is not a descriptive feature

drop_features = ['item_cnt_month', 'ID']



# inputs to model need to come in the form of a list containing a (1,n) vector

# for each categorical variable in order, and a (n, n_continuous_features) matrix

# of the remaining continuous features

X_train = []



# this loops over the categorical features and creates an individual vector for

# each one, appending it to the input list

for cat in cat_features:

  X_train.append(np.array(train_data_cat_features[~val_mask][cat]).reshape(-1,1))



# the remaining continuous features are appended to the end of the list as a matrix

X_train.append(train_data[~val_mask].drop(drop_features + cat_features, axis=1).values)

y_train = train_data[~val_mask].item_cnt_month



X_val = []



for cat in cat_features:

  X_val.append(np.array(train_data_cat_features[val_mask][cat]).reshape(-1,1))



X_val.append(train_data[val_mask].drop(drop_features + cat_features, axis=1).values)

y_val = train_data[val_mask].item_cnt_month



X_test=[]



for cat in cat_features:

    X_test.append(np.array(test_data_cat_features[cat]).reshape(-1,1))



X_test.append(test_data.drop(cat_features, axis=1).values)
train_examples_2 = []

for data in X_train:

  train_examples_2.append(data[:2])



train_examples_2
# prevents any conflicts with previously loaded models

tf.keras.backend.clear_session() 
def build_categorical_inputs(features):



    initial_inputs = {}

    cat_input_layers={}



    for feature in features:

        no_of_unique_cats  = train_test_cat_features[feature].nunique()

        embedding_size = int(min(np.ceil((no_of_unique_cats)/2), 50))

        categories  = no_of_unique_cats + 1



        initial_inputs[feature] = Input(shape=(1,))

        embedding_layer = Embedding(categories, 

                                    embedding_size,

                                    embeddings_regularizer=regularizers.l2(0.01),

                                    input_length=1)(initial_inputs[feature])

        cat_input_layers[feature] = Reshape(target_shape=(embedding_size,))(embedding_layer)



    return initial_inputs, cat_input_layers
def build_model():

  

    initial_inputs, input_layers = build_categorical_inputs(cat_features)



    no_of_num_features = len(train_data.columns) - len(cat_features) - len(drop_features)

    

    initial_inputs['numerical_features'] = Input(shape=(no_of_num_features,))

    input_layers['numerical_features'] = initial_inputs['numerical_features']



    inputs = Concatenate(axis=-1)([layer for layer in input_layers.values()])



    drop_1_out = Dropout(0.1)(inputs)

    dense_1_out = Dense(256, activation='relu', 

                        kernel_regularizer=regularizers.l2(0.01))(drop_1_out)

    drop_2_out = Dropout(0.1)(dense_1_out)

    dense_2_out = Dense(125, activation='relu', 

                        kernel_regularizer=regularizers.l2(0.01))(drop_2_out)

    drop_3_out = Dropout(0.1)(dense_2_out)

    dense_3_out = Dense(64, activation='relu', 

                        kernel_regularizer=regularizers.l2(0.01))(drop_3_out)

    drop_4_out = Dropout(0.1)(dense_3_out)

    dense_4_out = Dense(32, activation='relu', 

                        kernel_regularizer=regularizers.l2(0.01))(drop_4_out)

    final_out = Dense(1, kernel_regularizer=regularizers.l2(0.01))(dense_4_out)   



    model = keras.Model(inputs=[input for input in initial_inputs.values()], 

                        outputs=final_out)



    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)



    model.compile(loss='mse',

                  optimizer=optimizer,

                  metrics=['mae', 'mse'])

    

    return model



model = build_model()
model.summary()
train_examples_10 = []

for data in X_train:

  train_examples_10.append(data[:10])



example_result = model.predict(train_examples_10)

example_result
checkpoint_path = os.getcwd()

checkpoint_dir = os.path.dirname(checkpoint_path)



# Create a callback that saves the model's weights

# not particularly useful in the Kaggle Notebook environment 

# but can be really useful for training in other environments

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,

                                                 save_weights_only=True,

                                                 verbose=1)



# create an early stopping callback 

# will stop training if the validation MSE hasn't improved in (patience) epochs

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)



# train the model

history = model.fit(X_train, y_train,

                    batch_size=1000,

                    epochs=30, 

                    validation_data =(X_val, y_val),

                    callbacks=[checkpoint, early_stop],

                    verbose=1)
hist = pd.DataFrame(history.history)

hist['epoch'] = history.epoch

hist.to_csv('model_history.csv', index=False)

hist
def plot_history(hist):

    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Abs Error')

    plt.plot(hist['epoch'], hist['mae'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_mae'],

           label='Val Error')  

    plt.legend()



    plt.figure()

    plt.xlabel('Epoch')

    plt.ylabel('Mean Square Error')

    plt.plot(hist['epoch'], hist['mse'],

           label='Train Error')

    plt.plot(hist['epoch'], hist['val_mse'],

           label='Val Error')

    plt.legend()

    plt.show()





plot_history(hist.iloc[5:])
model.evaluate(X_train, y_train)
predictions = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({

    "ID": test_data.index,

    'item_cnt_month': predictions.flatten()

})

submission.to_csv('model_output.csv', index=False)
import pandas as pd

item_categories = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")

items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")

shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")