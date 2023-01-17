import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



from tensorflow import keras

from tensorflow.keras import Sequential

from tensorflow.keras.layers import MaxPool2D, Conv2D, Dense, Flatten, Dropout, Activation

from keras.callbacks import ReduceLROnPlateau



from matplotlib import pyplot as plt

from PIL import Image
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

submission_df = pd.read_csv('/kaggle/input/digit-recognizer/sample_submission.csv')
img_rows, img_cols = 28, 28

num_classes = 10
X_train = train_df.iloc[:, 1:].values.reshape(-1,img_cols, img_rows,1)

Y_train = keras.utils.to_categorical(train_df.iloc[:, :1].values, num_classes=10)



X_test = test_df.values.reshape(-1,img_cols, img_rows,1)



fig=plt.figure(figsize=(14, 3))

columns = 10

rows = 2

for i in range(1, columns*rows +1):

    img = np.random.randint(15, size=(14,14))

    fig.add_subplot(rows, columns, i)

    plt.imshow(X_train[i][:,:,0])

plt.show()
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=2)
data_gen = keras.preprocessing.image.ImageDataGenerator(

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

        vertical_flip=False)



data_gen.fit(X_train)



# learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

#                                             patience=3, 

#                                             verbose=1, 

#                                             factor=0.5, 

#                                             min_lr=0.00001)

# train_gen = data_gen.flow(X_train, Y_train, batch_size=32)

# val_gen = data_gen.flow(X_val, Y_val, batch_size=32)
model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))
# optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'mse'])
# history = model.fit_generator(

#     data_gen.flow(X_train,Y_train, batch_size=32),

#     epochs = 3,

#     validation_data = (X_val,Y_val),

#     steps_per_epoch=X_train.shape[0] // 32

# )

epochs = 30

batch_size=86

history = model.fit_generator(data_gen.flow(X_train,Y_train, batch_size=batch_size),

                              epochs = epochs, validation_data = (X_val,Y_val),

                              verbose = 2, steps_per_epoch=X_train.shape[0] // batch_size

#                               , callbacks=[learning_rate_reduction]

                             )
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

Y_pred = model.predict(X_val.astype(np.float64))

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

# Convert validation observations to one hot vectors

Y_true = np.argmax(Y_val,axis = 1)

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
Y_test_predict = model.predict(X_test.astype(np.float64))
results = np.argmax(Y_test_predict, axis=1)

results = pd.Series(results,name="Label")
submission = pd.concat([submission_df['ImageId'], results], axis=1)

submission.to_csv('submission_csv.csv', index=False)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()