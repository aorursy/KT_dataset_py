# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
import math

from tensorflow import keras
from tensorflow.keras import layers

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test= pd.read_csv("/kaggle/input/titanic/test.csv")
df_train.head(30)
print(df_train.shape)
print(df_test.shape)
df_train.info()
df_test.info()
df_train.isnull().sum()
df_test.isnull().sum()
df_val = df_train.sample(frac=0.2, random_state=1337)
df_train = df_train.drop(df_val.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(df_train), len(df_val))
)
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.drop(['PassengerId','Fare','Name', 'Cabin','Ticket','Embarked'], axis=1)
    dataframe = dataframe.copy()
    labels = dataframe.pop("Survived")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


ds_train = dataframe_to_dataset(df_train)
ds_val = dataframe_to_dataset(df_val)

def dataframe_to_dataset_test(dataframe):
    dataframe = dataframe.drop(['Fare','Cabin','Ticket','Embarked','Name'], axis=1)
    dataframe = dataframe.copy()
    labels = dataframe.pop("PassengerId")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


ds_test = dataframe_to_dataset_test(df_test)


for x, y in ds_train.take(1):
    print("Input:", x)
    print("Survived:", y)
for x, y in ds_val.take(1):
    print("Input:", x)
    print("Survived:", y)
ds_train = ds_train.batch(32)
ds_val = ds_val.batch(32)
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding
from tensorflow.keras.layers.experimental.preprocessing import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_string_categorical_feature(feature, name, dataset):
    # Create a StringLookup layer which will turn strings into integer indices
    index = StringLookup()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    index.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = index(feature)

    # Create a CategoryEncoding for our integer indicesfb
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a dataset of indices
    feature_ds = feature_ds.map(index)

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(encoded_feature)
    return encoded_feature


def encode_integer_categorical_feature(feature, name, dataset):
    # Create a CategoryEncoding for our integer indices
    encoder = CategoryEncoding(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the space of possible indices
    encoder.adapt(feature_ds)

    # Apply one-hot encoding to our indices
    encoded_feature = encoder(feature)
    return encoded_feature


pclass = keras.Input(shape=(1,), name="Pclass", dtype="int64")
parch = keras.Input(shape=(1,), name="Parch")
sibsp = keras.Input(shape=(1,), name="SibSp")

sex = keras.Input(shape=(1,), name="Sex", dtype="string")

all_inputs = [
    pclass,
    parch,
    sibsp,
    sex,
  
]

pclass_encoded = encode_integer_categorical_feature(pclass, "Pclass", ds_train)
sex_encoded = encode_string_categorical_feature(sex, "Sex", ds_train)

parch_encoded = encode_numerical_feature(parch, "Parch", ds_train)
sibsp_encoded = encode_numerical_feature(sibsp, "SibSp", ds_train)

pclass_encoded_test = encode_integer_categorical_feature(pclass, "Pclass", ds_test)
sex_encoded_test = encode_string_categorical_feature(sex, "Sex", ds_test)

parch_encoded_test = encode_numerical_feature(parch, "Parch", ds_test)
sibsp_encoded_test = encode_numerical_feature(sibsp, "SibSp", ds_test)


all_features = layers.concatenate(
    [
        pclass_encoded,
        parch_encoded,
        sibsp_encoded,
        sex_encoded,
    
    ]
)


x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])


keras.utils.plot_model(model, show_shapes=True, rankdir="")

model.fit(ds_train, epochs=60, validation_data=ds_val)


#diccionario_test2 = {"Pclass":[1,1], "Parch":[0,0], "SibSp":[0,0],"Sex":["female","female"] }

predicciones = []

for index,fila in df_test.iterrows():
    diccionario_test = {"Pclass":fila["Pclass"], "Parch":fila["Parch"], "SibSp":fila["SibSp"],"Sex":fila["Sex"] }
    entrada_red = {name: tf.convert_to_tensor([value]) for name, value in diccionario_test.items()}
    predicciones.append(model.predict(entrada_red))


              
array_vivos_muertos = np.array([0 if e <= 0.50 else 1 for e in predicciones])
diccionario = {"PassengerId":df_test['PassengerId'] , "Survived":array_vivos_muertos}

dataframe_solution = pd.DataFrame(diccionario)

dataframe_solution

#entrada_red2 = {name: tf.convert_to_tensor([value]) for name, value in diccionario_test2.items()}

#print(entrada_red2)








dataframe_solution.to_csv("titanic_solution.csv", index=False)