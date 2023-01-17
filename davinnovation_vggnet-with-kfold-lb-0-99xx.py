import pandas as pd

import numpy as np

import keras.layers.core as core

import keras.layers.convolutional as conv

import keras.models as models

import keras.utils.np_utils as kutils



from sklearn.model_selection import StratifiedKFold
train = pd.read_csv("../input/train.csv").values

test = pd.read_csv("../input/test.csv").values
np_epoch = 10



batch_size = 64

img_rows, img_cols = 28, 28



nb_filters_1 = 64

nb_filters_2 = 128

nb_filters_3 = 256

nb_conv = 3



n_folds = 3
trainX = train[:,1:].reshape(train.shape[0],img_rows,img_cols,1)

trainX = trainX.astype(float)

trainX /= 255



trainY = kutils.to_categorical(train[:,0])

nb_classes = trainY.shape[1]
def create_model():

    model = models.Sequential()

    

    model.add(conv.Conv2D(nb_filters_1, nb_conv, nb_conv,  activation="relu", input_shape=(28, 28, 1), border_mode='same'))

    model.add(conv.Conv2D(nb_filters_1, nb_conv, nb_conv, activation="relu", border_mode='same'))

    model.add(conv.MaxPooling2D(strides=(2,2)))



    model.add(conv.Conv2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))

    model.add(conv.Conv2D(nb_filters_2, nb_conv, nb_conv, activation="relu", border_mode='same'))

    model.add(conv.MaxPooling2D(strides=(2,2)))

    

    model.add(core.Flatten())

    model.add(core.Dropout(0.3))

    model.add(core.Dense(128,activation='relu'))

    model.add(core.Dense(nb_classes, activation="softmax"))

    

    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

    

    return model
predict = []
folding = StratifiedKFold(n_splits=n_folds,random_state=7,shuffle=True)
for train_index, test_index in folding.split(trainX, [ i.argmax() for i in trainY]):

    model = create_model()

    model.fit(trainX[train_index], trainY[train_index], batch_size=batch_size, epochs=np_epoch,validation_data=(trainX[test_index],trainY[test_index]))



    testX = test.reshape(test.shape[0], 28, 28, 1)

    testX = testX.astype(float)

    testX /= 255.0



    predict.append(model.predict(testX))
predict_ = predict
predict = [i.argmax() for i in (predict[0]+predict[1]+predict[2])]
np.savetxt('mnist-vggnet_aa.csv', np.c_[range(1,len(predict)+1),predict], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')






