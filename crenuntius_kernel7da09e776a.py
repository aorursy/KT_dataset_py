import os

import sys

import warnings

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adamax

from tensorflow.keras import Input

import tensorflow as tf

import pandas as pd

import numpy as np

import decimal

from decimal import Decimal, ROUND_HALF_UP

from sklearn.preprocessing import OneHotEncoder



%matplotlib inline 
def rounding_func(inval):

    return int(Decimal(str(inval)).quantize(Decimal('1E1'), rounding=ROUND_HALF_UP))



def preprocessing_func(inputs_csv, train_flag=True):

    data = inputs_csv.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

    

    pc_mode = int(data['Pclass'].mode())

    se_mode = data['Sex'].mode()

    ag_mode = int(data['Age'].median())

    si_mode = int(data['SibSp'].mode())

    pa_mode = int(data['Parch'].mode())

    em_mode = data['Embarked'].mode()

    

    if train_flag:

        coll = inputs_csv.loc[:, ['Survived']]

        

    pd.isnull(data.at[pc_mode, 'Pclass'])

    pd.isnull(data.at[si_mode, 'SibSp'])

    pd.isnull(data.at[pa_mode, 'Parch'])

    

    for ind, val in enumerate(data['Pclass']):

        data.at[ind, 'Pclass'] = val - 1

        

    for ind, val in enumerate(data['Sex']):

        if val == 'male':

            data.at[ind, 'Sex'] = 0

        elif val == 'female':

            data.at[ind, 'Sex'] = 1

        else:

            data.at[ind, 'Sex'] = se_mode



    for ind, val in enumerate(data['Embarked']):

        if val != 'S' and val != 'Q' and val != 'C':

            val = em_mode[0]

        

        if val == 'S':

            data.at[ind, 'Embarked'] = 0

        elif val == 'Q':

            data.at[ind, 'Embarked'] = 1

        elif val == 'C':

            data.at[ind, 'Embarked'] = 2

        else:

            data.at[ind, 'Embarked'] = -1



    for ind, val in enumerate(data['Age']):

        try:

            if val < 1.0:

                data.at[ind, 'Age'] = 7

            elif val > 60.0:

                data.at[ind, 'Age'] = 6

            else:

                data.at[ind, 'Age'] = rounding_func(val) / 10

        except:

            data.at[ind, 'Age'] = rounding_func(ag_mode) / 10

            

    data['Age'] = data['Age'].astype(np.int8)

    

    for ind, val in enumerate(data['SibSp']):

        if val > 0:

            data.at[ind, 'SibSp'] = 1

            

    for ind, val in enumerate(data['Parch']):

        if val > 0:

            data.at[ind, 'Parch'] = 1



    

    oneenc = OneHotEncoder(categories='auto', sparse=False, dtype=np.int32)

    data = oneenc.fit_transform(data.values)

                

    if train_flag:

        coll = oneenc.fit_transform(coll.values)

    

        return data, coll

    

    return data
train_data = pd.read_csv('../input/titanic/train.csv')

test_data = pd.read_csv('../input/titanic/test.csv')

sub_data = pd.read_csv('../input/titanic/gender_submission.csv')

output_name = 'submission.csv'
train, coll = preprocessing_func(train_data)

test = preprocessing_func(test_data, train_flag=False)
hidden_dim = 64

output_dim = 2

epochs = 100

inputs = Input(shape=(train.shape[1],))

x = Dense(hidden_dim, activation='relu')(inputs)

x = Dense(hidden_dim, activation='relu')(x)

outputs = Dense(output_dim, activation='softmax')(x)



model = Model(inputs=inputs, outputs=outputs)

model.compile(loss='categorical_crossentropy',

                 optimizer=Adamax(lr=0.01),

                 metrics=['accuracy'])

model.fit(train, coll, batch_size=128, epochs=epochs, validation_split=0.3)

pred = model.predict(test)
sub_data['Survived'] = [int(np.argmax(_)) for _ in pred]

sub_data.to_csv(output_name, index=False)