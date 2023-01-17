import tensorflow

tensorflow.__version__



import tensorflow.keras as keras

keras.__version__
import pandas as pd

from keras.layers import *

from keras.models import *

from keras.callbacks import CSVLogger, ModelCheckpoint

from keras import losses

from keras import optimizers

from sklearn.model_selection import train_test_split
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

Dig_MNIST = pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
import matplotlib.pyplot as plt



def vizualize_data():

    _train = train.ix[:,1:].values.astype(float)

    y_train = train.ix[:,0].values



    X_test = test.values.astype(float)



    fig = plt.figure()



    Nsample = 4

    labels = list(range(10))

    for y in labels:

        d = train.ix[train['label']==y, 1:].sample(n=Nsample).values

        for idx in range(Nsample):

            a = fig.add_subplot(Nsample, len(labels), idx * len(labels) + y + 1)

            plt.imshow(d[idx].reshape((28,28)))

            plt.axis('off')



    plt.show()

    

vizualize_data()
import numpy as np



x = train.iloc[:, 1:].values



y = []

for label in train.iloc[:, :1].values:

  temp = []

  for i in range(10):

    if i == label:

      temp.append(1)

    else:

      temp.append(0)

  y.append(temp)

y = np.array(y)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=10000, random_state=42)



x_input = Input(shape=(784,))

y_prediction = BatchNormalization()(x_input)

y_prediction = Dense(20, activation=None)(y_prediction)

y_prediction = Activation('elu')(y_prediction)

y_prediction = Dropout(rate=0.3)(y_prediction)

y_prediction = Dense(20, activation=None)(y_prediction)

y_prediction = Activation('elu')(y_prediction)

prediction = Dense(10, activation='softmax')(y_prediction)



model = Model(inputs=[x_input], output=[prediction])

model.compile('RMSprop', losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()
model.fit(

    x_train, y_train,

    batch_size=16,

    epochs=20,

    verbose=1,

    validation_data=(x_valid, y_valid),

    callbacks=[

        CSVLogger('log.csv'),

        ModelCheckpoint('model.h5', save_best_only=True),

    ]

)
x_test=test.drop('id', axis=1).iloc[:,:].values

pred_probas = model.predict(x_test, batch_size=16)

prediction = pred_probas.argmax(axis=1)



submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = prediction

submission.to_csv("submission.csv", index=False)

submission.tail()