import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from tensorflow.keras import layers

from tensorflow.keras import activations

from tensorflow.keras import regularizers

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import seaborn as sns

import altair as alt

pd.set_option('max_columns', None)

plt.style.use('seaborn')



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

dataset = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

testset = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
X = dataset.iloc[:, 1:].values

y = dataset.iloc[:, 0].values

X_submission = testset.values



X = X.reshape(X.shape[0], 28, 28, 1)/255.0

X_submission = X_submission.reshape(X_submission.shape[0], 28, 28, 1)/255.0

y = to_categorical(y, num_classes=10)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from itertools import product

import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    # shear_range=0.1,

    zoom_range=0.1,

    # horizontal_flip=True,

    # vertical_flip=True,

)

datagen.fit(X_train)



def create_model(units=[256,256,10], l2=[0.01,0.01], dropout=[0.1,0.1], conv_loops=0):

    model = tf.keras.models.Sequential()



    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu', input_shape=(28,28,1)))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(32, kernel_size=5, padding='same', activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.4))



    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Conv2D(64, kernel_size=5, padding='same', activation='relu'))

    model.add(tf.keras.layers.BatchNormalization())

    model.add(tf.keras.layers.Dropout(0.4))



    model.add(tf.keras.layers.Flatten())



    for i in range(len(units)):

        if i == len(units) - 1:

            model.add(layers.Dense(units=units[i], activation=activations.softmax))

        else:

            model.add(layers.Dense(

                units=units[i],

                activation=activations.relu,

            ))

            model.add(tf.keras.layers.BatchNormalization())

            if dropout[i] > 0:

                model.add(layers.Dropout(dropout[i]))

    

    model.compile(

        optimizer='adam', 

        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 

        metrics=['accuracy', 'categorical_crossentropy']

    )

    return model



def create_models():

    models = dict()

    all_permutations = [dict(zip(models_params, v)) for v in product(*models_params.values())]

    for x in all_permutations:

        name = json.dumps(x).translate({ord(i): None for i in '{}[]" :,.'})

        models[name] = dict(

            params=x,

            classifier=create_model(**x)

        )

    return models



def fit_model(model, name):

    model.summary()

    tf.keras.backend.clear_session()

    fit_stats = model.fit(

        datagen.flow(

            X_train,

            y_train,

            batch_size=batch_size,

        ),

        # batch_size=batch_size,

        epochs=epochs,

        callbacks=[

            tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=8, verbose=1),

            tf.keras.callbacks.ModelCheckpoint(

                filepath='./models/{accuracy:.5f}'+name+'.h5',

                monitor='accuracy',

                verbose=1,

                save_best_only=True,

                mode='max',

            ),

        ],

        validation_data= (X_test, y_test),

        workers=4,

    )

    return dict(

        name=name,

        # history=fit_stats.history,

        epochs=len(fit_stats.history['accuracy']),

        accuracy=np.max(fit_stats.history['accuracy'])

    )



models_params = {

    'units': [[128,10]],

    'dropout': [[0.4]],

}

batch_size = 128

epochs = 75





models = create_models()

models
# For each model, train and save models

history = dict()

for i in models:

    print(i)

    stats = fit_model(models[i]['classifier'], i)

    history[stats['name']] = dict(

        epochs=stats['epochs'],

        accuracy=stats['accuracy']*100

    )



history = pd.DataFrame(history).T

history
final = models['units12810dropout04']['classifier']

y_pred = final.predict(X_submission)
from sklearn.metrics import confusion_matrix, accuracy_score



y_final = final.predict(X_test)

confusion_matrix(np.argmax(y_final, axis=1), np.argmax(y_test, axis=1))
accuracy_score(np.argmax(y_final, axis=1), np.argmax(y_test, axis=1))
y_pred = np.argmax(y_pred, axis=1)

y_pred
submission = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')

submission['Label'] = y_pred

submission.to_csv('submission.csv', index=False)

final.save('final_model')