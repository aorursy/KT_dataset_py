import pandas as pd

from keras.datasets import mnist

from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D, Flatten, Dense, Dropout, Activation , Concatenate, Input , BatchNormalization

from keras.optimizers import SGD

from keras.utils import plot_model

from matplotlib import pyplot as plt

from keras.callbacks import EarlyStopping

from keras import Model

from keras.preprocessing.image import ImageDataGenerator
# load data

df_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

df_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
df_train.columns
df_test.columns
# split our data into features & target

trainX = df_train.drop('label', axis=1).values

trainy = df_train['label'].values.reshape(-1,1)



testX = df_test.values
trainX[:5]
trainy[:5]
testX[:5]
# summarize loaded dataset

print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))

print('Test: X=%s, y=%s' % (testX.shape))
# plot first few images

for i in range(9):

    img = trainX[i].reshape(28,28)

    # define subplot

    plt.subplot(330 + 1 + i)

    # plot raw pixel data

    plt.imshow(img)

    

# show the figure

plt.show()
# plot first few images

for i in range(9):

    img = trainX[i].reshape(28,28)

    # define subplot

    plt.subplot(330 + 1 + i)

    # plot raw pixel data

    plt.imshow(img,cmap=plt.get_cmap('gray') )

    

# show the figure

plt.show()
# reshape dataset to have a single channel

trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))

testX = testX.reshape((testX.shape[0], 28, 28, 1))

# one hot encode target values

trainy = to_categorical(trainy)
# convert from integers to floats

trainX = trainX.astype('float32')

testX = testX.astype('float32')

# normalize to range 0-1

trainX = trainX / 255.0

testX = testX / 255.0
print(trainX.shape)
datagen = ImageDataGenerator(

        validation_split = 0.2,

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images
batch_size = 128

Training_data = datagen.flow(trainX,

                             y=trainy,

                            batch_size = batch_size,

                            subset = 'training')



Validation_data = datagen.flow(trainX,

                             y=trainy,

                            batch_size = batch_size,

                            subset = 'validation')
es = EarlyStopping(monitor='val_loss', mode='auto', verbose=1, patience=10)
input_model = Input((trainX.shape[1],trainX.shape[2],trainX.shape[3]))





model1 = Conv2D(32,(5,5), activation='relu')(input_model)

model1 = BatchNormalization()(model1)

model1 = Conv2D(32,(5,5), activation='relu', padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = MaxPooling2D((2, 2))(model1)

model1 = Conv2D(64,(3,3), activation='relu' ,padding='same')(model1)

model1 = BatchNormalization()(model1)

model1 = Conv2D(64,(3,3), activation='relu' ,padding='valid')(model1)

model1 = BatchNormalization()(model1)

model1 = AveragePooling2D((2, 2))(model1)

model1 = Flatten()(model1)

#########################################################                          

model2 = Conv2D(32,(4,4), activation='relu')(input_model)  

model2 = BatchNormalization()(model2)

model2 = Conv2D(32,(4,4), activation='relu', padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = MaxPooling2D((2, 2))(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2) 

model2 = BatchNormalization()(model2)

model2 = Conv2D(32,(3,3), activation='relu', padding='same')(model2) 

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Conv2D(32,(2,2), activation='relu' ,padding='same')(model2)

model2 = BatchNormalization()(model2)

model2 = Conv2D(32,(2,2), activation='relu' ,padding='valid')(model2)

model2 = BatchNormalization()(model2)

model2 = AveragePooling2D((2, 2))(model2)

model2 = Flatten()(model2)

########################################################

model3 = Conv2D(32,(3,3), activation='relu')(input_model)  

model3 = BatchNormalization()(model3)

model3 = Conv2D(32,(3,3), activation='relu', padding='same')(model3)

model3 = BatchNormalization()(model3)

model3 = MaxPooling2D((2, 2))(model3)

model3 = Conv2D(32,(3,3), activation='relu', padding='same')(model3) 

model3 = BatchNormalization()(model3)

model3 = Conv2D(32,(3,3), activation='relu', padding='same')(model3)

model3 = BatchNormalization()(model3)

model3 = Conv2D(64,(2,2), activation='relu' ,padding='valid')(model3)

model3 = BatchNormalization()(model3)

model3 = Conv2D(128,(2,2), activation='relu' ,padding='valid')(model3)

model3 = BatchNormalization()(model3)

model3 = MaxPooling2D((2, 2))(model3)

model3 = Flatten()(model3)

########################################################

merged = Concatenate()([model1, model2 , model3])

merged = Dense(units = 512, activation = 'relu')(merged)

merged = Dropout(rate = 0.2)(merged)

merged = BatchNormalization()(merged)

merged = Dense(units = 20, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

merged = Dense(units = 15, activation = 'relu')(merged)

merged = BatchNormalization()(merged)

output = Dense(units = 10, activation = 'softmax')(merged)



model = Model(inputs= [input_model], outputs=[output])
model.summary()
plot_model(model, show_shapes=True)
sgd = SGD(lr=0.01, momentum=0.9)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit_generator(Training_data, 

                              epochs= 150,

                              validation_data= Validation_data,

                              verbose=1,

                              callbacks=[es])
model.save_weights("MNIST_weights.h5")
val_loss = history.history['val_loss']

loss = history.history['loss']



plt.plot(val_loss)

plt.plot(loss)

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['Val error','Train error'], loc='upper right')

plt.savefig('plot_error.png')

plt.show()
val_accuracy = history.history['val_accuracy']

accuracy = history.history['accuracy']



plt.plot(val_accuracy)

plt.plot(accuracy)

plt.xlabel('Epochs')

plt.ylabel('accuracy')

plt.legend(['Val accuracy','Train accuracy'], loc='upper right')

plt.savefig( 'plot_accuracy.png')

plt.show()
pred = model.predict(testX)

pred = pd.DataFrame(pred)

pred['Label'] = pred.idxmax(axis=1)

pred.head(5)
pred['index'] = list(range(1,len(pred)+1))

pred.head()
submission = pred[['index','Label']]

submission.head()
submission.rename(columns={'index':'ImageId'},inplace = True)

submission.head()
submission.to_csv('submission.csv',index=False)