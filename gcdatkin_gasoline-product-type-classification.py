import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.utils import class_weight



import tensorflow as tf
data = pd.read_csv('../input/gas-prices-in-brazil/2004-2019.tsv', delimiter='\t')
data
data.info()
unneeded_columns = ['Unnamed: 0', 'DATA INICIAL', 'DATA FINAL']



data = data.drop(unneeded_columns, axis=1)
data
data.isna().sum()
data['PRODUTO'].value_counts()
plt.figure(figsize=(12, 12))

plt.pie(

    x=data['PRODUTO'].value_counts(),

    labels=data['PRODUTO'].value_counts().index,

    autopct='%.1f%%',

    colors=sns.color_palette('rocket')

)

plt.show()
data
label_encoder = LabelEncoder()



data['PRODUTO'] = label_encoder.fit_transform(data['PRODUTO'])
dict(enumerate(label_encoder.classes_))
{column: list(data[column].unique()) for column in ['REGIÃO', 'ESTADO', 'UNIDADE DE MEDIDA']}
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['REGIÃO', 'ESTADO', 'UNIDADE DE MEDIDA'],

    ['R', 'E', 'U']

)
data
data.isin(['-']).sum()
data = data.replace('-', np.NaN)



for column in data.columns:

    data[column] = data[column].fillna(data[column].astype(np.float).mean())
data.isna().sum().sum()
data
y = data.loc[:, 'PRODUTO']

X = data.drop('PRODUTO', axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=34)
X.shape
num_classes = len(y.unique())
class_weights = dict(

    enumerate(

        class_weight.compute_class_weight(

            'balanced',

            y_train.unique(),

            y_train

        )

    )

)



class_weights
inputs = tf.keras.Input(shape=(49,))

x = tf.keras.layers.Dense(64, activation='relu')(inputs)

x = tf.keras.layers.Dense(64, activation='relu')(x)

outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)



model = tf.keras.Model(inputs, outputs)





model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics=['accuracy']

)





batch_size = 32

epochs = 100



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    class_weight=class_weights,

    batch_size=batch_size,

    epochs=epochs,

    callbacks=[

        tf.keras.callbacks.EarlyStopping(

            monitor='val_loss',

            patience=3,

            restore_best_weights=True,

            verbose=1

        )

    ]

)
model.evaluate(X_test, y_test)