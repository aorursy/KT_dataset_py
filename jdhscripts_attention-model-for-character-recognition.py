import numpy as np

import pandas as pd



from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential

from keras.layers import Dense, Activation, Flatten, LSTM, Convolution2D

from keras.layers import advanced_activations

from keras.layers.normalization import BatchNormalization

from keras.layers.core import Reshape



import matplotlib as plt



%matplotlib inline



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainRaw = pd.read_csv('../input/train.csv');

testRaw = pd.read_csv('../input/test.csv');
trainxraw = trainRaw.iloc[:,1:].astype(np.float32);

trx = trainxraw.as_matrix()

trx2 = []

for i in range(trx.shape[0]):

    trx2.append(np.reshape(trx[i],(28,28,1)))

trx3 = np.asarray(trx2)

trx3.shape
trainyraw = pd.get_dummies(trainRaw.iloc[:,0]).astype(np.float32);

finalx = testRaw;

trainxraw.shape
trainx = trx3[:40000,:]

trainy = trainyraw.iloc[:40000,:]

testx = trx3[40000:,:]

testy = trainyraw.iloc[40000:,:]

trainx = np.asarray(trainx)

trainy = np.asarray(trainy)

testx = np.asarray(testx)

testy = np.asarray(testy)
model = Sequential()



model.add(BatchNormalization(input_shape = trainx.shape[1:]))



print(model.output_shape)

model.add(Convolution2D(16, 3, 3))

model.add(advanced_activations.ELU())

print(model.output_shape)



print(model.output_shape)

model.add(Reshape((model.output_shape[1],-1)))

print(model.output_shape)



model.add(LSTM(16, return_sequences=False))

model.add(advanced_activations.LeakyReLU());



model.add(Dense(32))

model.add(advanced_activations.LeakyReLU());

model.add(BatchNormalization())



model.add(Dense(trainy.shape[1]))

model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])
batch_size = 5000

nb_epoch = 300

verb=1

data_augmentation = True
X_train = trainx;

Y_train = trainy;

X_test = testx;

Y_test = testy;



if not data_augmentation:

    print('Not using data augmentation.')

    model.fit(X_train, Y_train,

              batch_size=batch_size,

              nb_epoch=nb_epoch,

              validation_data=(X_test, Y_test),

              verbose=verb,

              shuffle=True)

else:

    print('Using real-time data augmentation.')

    datagen = ImageDataGenerator(

        zoom_range = 0.1,

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images



    datagen.fit(trainx)

    dataflow = datagen.flow(trainx, Y_train,

                                     batch_size=batch_size)

    model.fit_generator(dataflow,

                        samples_per_epoch=X_train.shape[0],

                        nb_epoch=nb_epoch,

                        verbose=verb,

                        validation_data=(X_test, Y_test))
pred = model.predict_classes(np.asarray(finalx).reshape(-1,28,28,1), batch_size=1000);
print(pred[0:5])

#should be 2, 0, 9, 0, 3

for i in range(5):

    img = np.asarray(finalx).reshape(-1,28,28)[i].reshape(28,28);

    plt.pyplot.imshow(img, cmap='Greys_r')

    plt.pyplot.show()
finaly = pd.DataFrame([]);

finaly['ImageId'] = np.arange(1,pred.shape[0] + 1);

finaly['Label'] = pred[:];

finaly.to_csv('output.csv', columns = ['ImageId','Label'], index = False);
