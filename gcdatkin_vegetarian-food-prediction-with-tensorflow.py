import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/indian-food-101/indian_food.csv')
data
data.isna().sum()
food_vocab = set()



for ingredients in data['ingredients']:

    for food in ingredients.split(','):

        if food.strip().lower() not in food_vocab:

            food_vocab.add(food.strip().lower())
food_vocab
len(food_vocab)
food_columns = pd.DataFrame()



for i, ingredients in enumerate(data['ingredients']):

    for food in ingredients.split(','):

        if food.strip().lower() in food_vocab:

            food_columns.loc[i, food.strip().lower()] = 1



food_columns = food_columns.fillna(0)
food_columns
data = data.drop(['name', 'ingredients'], axis=1)
data
{column: list(data[column].unique()) for column in data.columns if data.dtypes[column] == 'object'}
data[['flavor_profile', 'state', 'region']] = data[['flavor_profile', 'state', 'region']].replace('-1', np.NaN)
def onehot_encode(df, columns, prefixes):

    df = df.copy()

    for column, prefix in zip(columns, prefixes):

        dummies = pd.get_dummies(df[column], prefix=prefix)

        df = pd.concat([df, dummies], axis=1)

        df = df.drop(column, axis=1)

    return df
data = onehot_encode(

    data,

    ['flavor_profile', 'course', 'state', 'region'],

    ['f', 'c', 's', 'r']

)
data
data[['prep_time', 'cook_time']] = data[['prep_time', 'cook_time']].replace(-1, np.NaN)
data['prep_time'] = data['prep_time'].fillna(data['prep_time'].mean())

data['cook_time'] = data['cook_time'].fillna(data['cook_time'].mean())
label_encoder = LabelEncoder()



data['diet'] = label_encoder.fit_transform(data['diet'])
{index: label for index, label in enumerate(label_encoder.classes_)}
data
y = data['diet']



X = data.drop('diet', axis=1)

X_food = pd.concat([X, food_columns], axis=1)
scaler = StandardScaler()



X = scaler.fit_transform(X)

X_food = scaler.fit_transform(X_food)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

X_food_train, X_food_test, y_food_train, y_food_test = train_test_split(X_food, y, train_size=0.7, random_state=42)
def build_model(num_features, hidden_layer_sizes=(64, 64)):

    inputs = tf.keras.Input(shape=(num_features,))

    x = tf.keras.layers.Dense(hidden_layer_sizes[0], activation='relu')(inputs)

    x = tf.keras.layers.Dense(hidden_layer_sizes[1], activation='relu')(x)

    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    

    

    model.compile(

        optimizer='adam',

        loss='binary_crossentropy',

        metrics=[

            'accuracy',

            tf.keras.metrics.AUC(name='auc')

        ]

    )

    

    return model
X.shape
model = build_model(40)



batch_size = 64

epochs = 41



history = model.fit(

    X_train,

    y_train,

    validation_split=0.2,

    batch_size=batch_size,

    epochs=epochs,

    verbose=0

)
plt.figure(figsize=(20, 10))



epochs_range = range(1, epochs + 1)

train_loss, val_loss = history.history['loss'], history.history['val_loss']

train_auc, val_auc = history.history['auc'], history.history['val_auc']



plt.subplot(1, 2, 1)

plt.plot(epochs_range, train_loss, label="Training Loss")

plt.plot(epochs_range, val_loss, label="Validation Loss")

plt.title("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(epochs_range, train_auc, label="Training AUC")

plt.plot(epochs_range, val_auc, label="Validation AUC")

plt.title("AUC")

plt.legend()



plt.show()
print(np.argmin(val_loss), np.argmax(val_auc))
model.evaluate(X_test, y_test)
len(y_test)
X_food.shape
food_model = build_model(405, hidden_layer_sizes=(128, 128))



food_batch_size = 64

food_epochs = 200



food_history = food_model.fit(

    X_food_train,

    y_food_train,

    validation_split=0.2,

    batch_size=food_batch_size,

    epochs=food_epochs,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
plt.figure(figsize=(20, 10))



food_epochs_range = range(1, food_epochs + 1)

food_train_loss, food_val_loss = food_history.history['loss'], food_history.history['val_loss']

food_train_auc, food_val_auc = food_history.history['auc'], food_history.history['val_auc']



plt.subplot(1, 2, 1)

plt.plot(food_epochs_range, food_train_loss, label="Training Loss")

plt.plot(food_epochs_range, food_val_loss, label="Validation Loss")

plt.title("Loss")

plt.legend()



plt.subplot(1, 2, 2)

plt.plot(food_epochs_range, food_train_auc, label="Training AUC")

plt.plot(food_epochs_range, food_val_auc, label="Validation AUC")

plt.title("AUC")

plt.legend()



plt.show()
print(np.argmin(food_val_loss), np.argmax(food_val_auc))
food_model.evaluate(X_food_test, y_food_test)