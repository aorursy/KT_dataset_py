import tensorflow as tf

from tensorflow.keras.layers import Input, Dense

from tensorflow.keras.models import Model

import tensorflow.keras.backend as k

from sklearn.metrics import classification_report

import numpy as np

import pandas as pd
csvTrain = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

csvTest = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

csvSample = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

csvTrainY, csvTrainX = csvTrain.iloc[:,:1], csvTrain.iloc[:,1:]

csvSample.head(), csvSample.tail()
k.clear_session()

batchSize = 100

inputData = Input(shape = [784], dtype = tf.float32)

hiddenLayer = Dense(300, activation = 'relu')(inputData)

output = tf.keras.layers.Dense(10, activation='softmax')(hiddenLayer)

model = Model(inputs = inputData, outputs = output)

model.compile(

    optimizer='adam',

    loss='sparse_categorical_crossentropy',

    metrics='sparse_categorical_accuracy'

)

model.fit(csvTrainX, csvTrainY, epochs = 10, batch_size=100)

report = classification_report(np.argmax(model.predict(csvTrainX), axis = 1), csvTrainY)

print(report)
testPrediction = np.argmax(model.predict(csvTest), axis=1)

d = {

        "ImageId": list(range(1, len(testPrediction)+1)),

        "Label": testPrediction.tolist()

    }

submission = pd.DataFrame(d)

submission.to_csv("/kaggle/working/submit.csv", index = False)