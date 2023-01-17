import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



from sklearn.model_selection import train_test_split



from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop, Adam, Adagrad, Adamax, Adadelta



from sklearn.metrics import classification_report, confusion_matrix


import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train = pd.read_csv("../input/Kannada-MNIST/train.csv")

test = pd.read_csv("../input/Kannada-MNIST/test.csv")
df_digit_counts =  train.label.value_counts().reset_index()



plt.figure(figsize=(20,8))

ax = sns.barplot(x='index', y='label', data=df_digit_counts)



for i in ax.patches:

    v1 = round((i.get_height()/len(train))*100, 2)

    ax.annotate(f'{int(i.get_height())} ({v1}%)', (i.get_x()+0.4, i.get_height()),

                    ha='center', va='bottom',

                    color= 'black')



plt.title("Digit Count")

plt.ylabel("Counts")

plt.xlabel("Digits")

plt.show()
train_X, train_y = train.drop(columns=['label']), train["label"]
train_X = np.array(train_X)

train_y = np.array(train_y)
testx = test.drop(columns=['id'])
testx = np.array(testx)
train_X = train_X.reshape(-1,28,28,1)

testx = testx.reshape(-1,28,28,1)
train_X = train_X / 255.0

testx = testx / 255.0
train_y = to_categorical(train_y, num_classes = len(np.unique(train["label"])))
X_train, X_test, y_train, y_test = train_test_split(train_X, train_y, test_size=0.25, random_state=42, stratify= train_y, shuffle=True)
epochs = 20

batch_size = 16
def create_model():

    model = Sequential()



    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu', input_shape = (28,28,1)))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 128, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(Conv2D(filters = 256, kernel_size = (3,3),padding = 'Same', activation ='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

    model.add(Dropout(0.25))



    model.add(Flatten())

    model.add(Dense(512, activation = "relu"))

    model.add(BatchNormalization())

    model.add(Dropout(0.3))



    model.add(Dense(10, activation = "softmax"))

    

    return model
model = create_model()

model.summary()
optimizer = Adamax()

model.compile(optimizer=optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])



# Set a learning rate annealer

lr_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)

es = EarlyStopping(monitor='val_loss',

                              min_delta=0,

                              patience=5,

                              verbose=0, mode='auto')



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.12,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.12,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)
# Fit the model

history = model.fit_generator(datagen.flow(X_train,y_train, batch_size=batch_size),

                              epochs = epochs, 

                              validation_data = (X_test,y_test), 

                              steps_per_epoch=X_train.shape[0] // batch_size,

                              callbacks=[lr_reduction, es], 

                              shuffle=True)
model.evaluate(X_train, y_train), model.evaluate(X_test, y_test)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
ypred = model.predict(X_test)

ypred = np.argmax(ypred, axis=1)

ytest = np.argmax(y_test, axis=1)



cf_matrix = confusion_matrix(ytest, ypred)



plt.figure(figsize=(20,8))

ax = sns.heatmap(cf_matrix, annot=True, fmt='g')

plt.show()



print("\n\n")

print(classification_report(ytest, ypred))
def create_sub():

    results = model.predict(testx)

    results = np.argmax(results,axis = 1)



    df_sub = pd.read_csv("../input/Kannada-MNIST/sample_submission.csv")

    df_sub['label'] = results

    df_sub.to_csv("submission.csv",index=False)
create_sub()