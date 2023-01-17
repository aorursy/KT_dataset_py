import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.express as px



from sklearn.model_selection import train_test_split



import tensorflow as tf
data = pd.read_csv('../input/age-gender-and-ethnicity-face-data-csv/age_gender.csv')
data
data.isnull().sum()
data = data.drop('img_name', axis=1)
{column: list(data[column].unique()) for column in ['gender', 'ethnicity', 'age']}
data['age'] = pd.qcut(data['age'], q=4, labels=[0, 1, 2, 3])
data
print(len(data['pixels'][0].split(' ')))

print(np.sqrt(2304))
num_pixels = 2304

img_height = 48

img_width = 48
target_columns = ['gender', 'ethnicity', 'age']



y = data[target_columns]

X = data.drop(target_columns, axis=1)
y
X
X = pd.Series(X['pixels'])

X = X.apply(lambda x: x.split(' '))

X = X.apply(lambda x: np.array(list(map(lambda z: np.int(z), x))))

X = np.array(X)

X = np.stack(np.array(X), axis=0)

X = np.reshape(X, (-1, 48, 48))



X.shape
plt.figure(figsize=(10, 10))



for index, image in enumerate(np.random.randint(0, 1000, 9)):

    plt.subplot(3, 3, index + 1)

    plt.imshow(X[image])

    plt.axis('off')



plt.show()
y
y_gender = np.array(y['gender'])

y_ethnicity = np.array(y['ethnicity'])

y_age = np.array(y['age'])
X.shape
def build_model(num_classes, activation='softmax', loss='sparse_categorical_crossentropy'):

    

    inputs = tf.keras.Input(shape=(img_height, img_width, 1))

    x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(inputs)

    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(128, activation='relu')(x)

    outputs = tf.keras.layers.Dense(num_classes, activation=activation)(x)

    

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    

    

    model.compile(

        optimizer='adam',

        loss=loss,

        metrics=['accuracy']

    )

    

    return model
{column: list(data[column].unique()) for column in ['gender', 'ethnicity', 'age']}
X_gender_train, X_gender_test, y_gender_train, y_gender_test = train_test_split(X, y_gender, train_size=0.7)

X_ethnicity_train, X_ethnicity_test, y_ethnicity_train, y_ethnicity_test = train_test_split(X, y_ethnicity, train_size=0.7)

X_age_train, X_age_test, y_age_train, y_age_test = train_test_split(X, y_age, train_size=0.7)
gender_model = build_model(1, activation='sigmoid', loss='binary_crossentropy')



gender_history = gender_model.fit(

    X_gender_train,

    y_gender_train,

    validation_split=0.2,

    batch_size=64,

    epochs=7,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig = px.line(

    gender_history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Gender Model"

)



fig.show()
gender_acc = gender_model.evaluate(X_gender_test, y_gender_test)[1]
ethnicity_model = build_model(5, activation='softmax', loss='sparse_categorical_crossentropy')



ethnicity_history = ethnicity_model.fit(

    X_ethnicity_train,

    y_ethnicity_train,

    validation_split=0.2,

    batch_size=64,

    epochs=8,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig = px.line(

    ethnicity_history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Ethnicity Model"

)



fig.show()
ethnicity_acc = ethnicity_model.evaluate(X_ethnicity_test, y_ethnicity_test)[1]
age_model = build_model(4, activation='softmax', loss='sparse_categorical_crossentropy')



age_history = age_model.fit(

    X_age_train,

    y_age_train,

    validation_split=0.2,

    batch_size=64,

    epochs=7,

    callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],

    verbose=0

)
fig = px.line(

    age_history.history,

    y=['loss', 'val_loss'],

    labels={'index': "Epoch", 'value': "Loss"},

    title="Age Model"

)



fig.show()
age_acc = age_model.evaluate(X_age_test, y_age_test)[1]
fig = px.bar(

    x=["Gender", "Ethnicity", "Age"],

    y=[gender_acc, ethnicity_acc, age_acc],

    labels={'x': "", 'y': "Accuracy"},

    color=["Gender", "Ethnicity", "Age"],

    title="Model Performance"

)



fig.show()