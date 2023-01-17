import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/adult-census-income/adult.csv')
data
data.drop('education', axis=1, inplace=True)
data.isna().sum()
data.isin(['?']).sum()
data = data.replace('?', np.NaN)
data.isna().sum()
data
categorical_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
def get_uniques(df, columns):

    uniques = dict()

    for column in columns:

        uniques[column] = list(df[column].unique())

    return uniques
get_uniques(data, categorical_features)
binary_features = ['sex']



nominal_features = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
def binary_encode(df, columns):

    label_encoder = LabelEncoder()

    for column in columns:

        df[column] = label_encoder.fit_transform(df[column])

    return df



def onehot_encode(df, columns):

    for column in columns:

        dummies = pd.get_dummies(df[column])

        df = pd.concat([df, dummies], axis=1)

        df.drop(column, axis=1, inplace=True)

    return df
data = binary_encode(data, binary_features)

data = onehot_encode(data, nominal_features)
(data.dtypes == 'object').sum()
data
y = data['income']

X = data.drop('income', axis=1)
label_encoder = LabelEncoder()

y = label_encoder.fit_transform(y)

y_mappings = {index: label for index, label in enumerate(label_encoder.classes_)}

y_mappings
y
scaler = MinMaxScaler()

X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
inputs = tf.keras.Input(shape=(88,))

x = tf.keras.layers.Dense(16, activation='relu')(inputs)

x = tf.keras.layers.Dense(16, activation='relu')(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs=inputs, outputs=outputs)





optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)



metrics = [

    tf.keras.metrics.BinaryAccuracy(name='acc'),

    tf.keras.metrics.AUC(name='auc')

]



model.compile(

    optimizer=optimizer,

    loss='binary_crossentropy',

    metrics=metrics

)





batch_size = 32

epochs = 26



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
plt.figure(figsize=(14, 10))



epochs_range = range(1, epochs + 1)

train_loss = history.history['loss']

val_loss = history.history['val_loss']



plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")



plt.xlabel("Epochs")

plt.ylabel("Loss")

plt.legend()



plt.show()
np.argmin(val_loss)
model.evaluate(X_test, y_test)
y.sum() / len(y)