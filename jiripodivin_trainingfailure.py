# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tensorflow.keras import Sequential, Input, Model

from tensorflow.keras.layers import Dense, Concatenate, Flatten, Dropout, Add

from tensorflow.keras import optimizers

import tensorflow as tf

import matplotlib.pyplot as plt

import re

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_df = pd.read_csv('/kaggle/input/titanic/train.csv')

test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
def title_extraction(name):

    #Name column values take form of FirstName, Title. SecondName

    title = name.split(',')[1].split('.')[0].lstrip()

    return title
train_df['Title'] = train_df['Name'].apply(title_extraction)

train_df['NameL'] = train_df['Name'].apply(lambda x: len(x))



test_df['Title'] = test_df['Name'].apply(title_extraction)

test_df['NameL'] = test_df['Name'].apply(lambda x: len(x))
train_df['HadCabin'] = train_df['Cabin'].isna().astype(float)

train_df['Cabin'] = train_df['Cabin'].fillna('X')

train_df['Cabin'] = train_df['Cabin'].apply(lambda x: x[0])



test_df['HadCabin'] = test_df['Cabin'].isna().astype(float)

test_df['Cabin'] = test_df['Cabin'].fillna('X')

test_df['Cabin'] = test_df['Cabin'].apply(lambda x: x[0])
categorical_vars = ['Pclass', 'Embarked', 'SibSp', 'Parch', 'Title', 'Cabin']



train_df['Sex'] = train_df['Sex'].apply(lambda x: 1 if x=='male' else 0)



for variable in categorical_vars:

    train_df = pd.concat([train_df, pd.get_dummies(train_df[variable], dummy_na=True, prefix=variable)], axis=1)





test_df['Sex'] = test_df['Sex'].apply(lambda x: 1 if x=='male' else 0)



for variable in categorical_vars:

    test_df = pd.concat([test_df, pd.get_dummies(test_df[variable], dummy_na=True, prefix=variable)], axis=1)
train_df['NanAge'] = train_df['Age'].isna().astype(float)

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].median())



test_df['NanAge'] = test_df['Age'].isna().astype(float)

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].median())
labels = train_df['Survived'].values



train_df = train_df.drop(columns=['Name', 'Embarked', 'PassengerId', 'Survived', 'Cabin', 'Ticket']+categorical_vars)

test_df = test_df.drop(columns=['Name', 'Embarked', 'PassengerId', 'Cabin', 'Ticket']+categorical_vars)



train_df_old = train_df.copy()

test_df_old = test_df.copy()



train_df = train_df.reindex(columns=train_df.columns.join(test_df.columns, 'outer').array, fill_value=0)

test_df = test_df.reindex(columns=test_df.columns.join(train_df.columns, 'outer').array, fill_value=0)





extra_train_cols = train_df.columns.difference(test_df.columns)

extra_test_cols = test_df.columns.difference(train_df.columns)
for column in train_df.columns:

    if train_df[column].min() != train_df[column].max():

        train_df[column] = (train_df[column]-train_df[column].min())/(train_df[column].max()-train_df[column].min())

        

for column in test_df.columns:

    if test_df[column].min() != test_df[column].max():

        test_df[column] = (test_df[column]-test_df[column].min())/(test_df[column].max()-test_df[column].min())       
validation_split = 5/6

batch_size = 32



training_data = train_df.values[:int(len(train_df.values)*validation_split)]

validation_data = train_df.values[int(len(train_df.values)*validation_split):]
train_labels = labels[:int(train_df.values.shape[0]*validation_split)]

training_data = tf.data.Dataset.from_tensor_slices((training_data, train_labels/1.0)).shuffle(training_data.shape[0]).repeat().batch(batch_size)



validation_labels = labels[int(train_df.values.shape[0]*validation_split):]

validation_data = tf.data.Dataset.from_tensor_slices((validation_data, validation_labels/1.0)).shuffle(validation_data.shape[0]).repeat().batch(batch_size)
models_performance = []
def plot_metrics(records, metric_keys, loss=True):

    fig, ax = plt.subplots()

    fig.set_size_inches((20, 8))



    for name, record in records:

        n_metric_keys = [key for key in record.keys() if any([m_key in str(key) for m_key in metric_keys])]



        for metric in n_metric_keys:

            ax.plot([i for i in range(len(record[metric]))], 

                    record[metric],

                    label=name + '_' + metric, 

                    ls='-.')



    ax.set_ylabel('Validation - '+' / '.join(metric_keys))

    ax.set_xlabel('Epoch')

    ax.legend()

    plt.show()

    

    if loss:

        fig, ax = plt.subplots()

        fig.set_size_inches((20, 8))



        for name, record in records:    

            ax.plot([i for i in range(len(record['val_loss']))], 

                    record['val_loss'],

                    label=name, 

                    ls='--')



        ax.set_ylabel('Validation Loss')

        ax.set_xlabel('Epoch')

        ax.legend()

        plt.show()
def simple_model(depth,

                 layer_size,

                 loss=tf.keras.losses.BinaryCrossentropy(),

                 kernel_regularizer=None):

    

    model_input= Input(shape=(train_df.values.shape[1],))

    

    model_hidden = Dense(layer_size, activation='relu')(model_input)



    for i in range(0, depth):

        model_hidden = Dense(

            layer_size, 

            activation='relu',

            kernel_regularizer=kernel_regularizer)(model_hidden)

        model_hidden = Dropout(0.3)(model_hidden)



    model_output = Dense(1, activation='sigmoid')(model_hidden)



    model = Model(model_input, model_output, name='Simple'+str(depth)+'_'+str(layer_size))



    model.compile(loss=loss,

                  optimizer='adam', 

                  metrics=[tf.keras.metrics.Accuracy(),

                           tf.keras.metrics.Precision(), 

                           tf.keras.metrics.Recall()])

    model.summary()

    tf.keras.utils.plot_model(model, to_file='simple_model.png', expand_nested=True, show_shapes=True)

    

    return model
def residual_model(depth,

                   layer_size,

                   loss=tf.keras.losses.BinaryCrossentropy(),

                   kernel_regularizer=None):

    

    model_input= Input(shape=(train_df.values.shape[1],))



    model_hidden = Dense(layer_size, activation='relu', name='FirstDeep')(model_input)



    line = [model_hidden]

    for i in range(0, depth):

        if len(line) > 1:

            #layer block    

            model_hidden = Dense(

                layer_size, 

                activation='relu',

                kernel_regularizer=kernel_regularizer)(concat_layer)

        else:

            #layer block    

            model_hidden = Dense(

                layer_size, 

                activation='relu',

                kernel_regularizer=kernel_regularizer)(model_hidden)

        model_hidden = Dropout(0.3)(model_hidden)

        line = line+[model_hidden]

        concat_layer = Add()(line)



    model_output = Dense(1, activation='sigmoid')(concat_layer) 



    model_residual = Model(model_input, model_output, name='Residual_'+str(depth)+'_'+str(layer_size))



    model_residual.compile(loss=loss,

                  optimizer='adam', 

                  metrics=[tf.keras.metrics.Accuracy(),

                           tf.keras.metrics.Precision(), 

                           tf.keras.metrics.Recall()])

    model_residual.summary()



    tf.keras.utils.plot_model(model_residual, to_file='residual_model.png', expand_nested=True, show_shapes=True)

    

    return model_residual
training_epochs = 50

training_steps = 20

layer_size = 32

layers = 3



models_performance = []

models_performance = dict()



models_performance['Residual'+str(layers)+'_'+str(layer_size)] = residual_model(layers,

                                          layer_size,

                                          loss=tf.keras.losses.BinaryCrossentropy(

                                            from_logits=True)).fit(

                                               training_data,

                                               epochs=training_epochs,

                                               steps_per_epoch=training_steps,

                                               validation_data=validation_data,

                                               validation_steps=32,

                                               verbose=1).history



models_performance['Residual'+str(layers*2)+'_'+str(layer_size)] = residual_model(layers*2,

                                          layer_size,

                                          loss=tf.keras.losses.BinaryCrossentropy(

                                            from_logits=True)).fit(

                                               training_data,

                                               epochs=training_epochs,

                                               steps_per_epoch=training_steps,

                                               validation_data=validation_data,

                                               validation_steps=32,

                                               verbose=1).history



models_performance['Residual'+str(layers*4)+'_'+str(layer_size)] = residual_model(layers*4,

                                          layer_size,

                                          loss=tf.keras.losses.BinaryCrossentropy(

                                            from_logits=True)).fit(

                                               training_data,

                                               epochs=training_epochs,

                                               steps_per_epoch=training_steps,

                                               validation_data=validation_data,

                                               validation_steps=32,

                                               verbose=1).history

metric_keys = ['val_accuracy']



plot_metrics(

    [(model_name, models_performance[model_name]) for model_name in models_performance.keys()],

    metric_keys,

    loss=True)