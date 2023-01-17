%%capture

import pandas as pd

import matplotlib.pyplot as plt

from sklearn import model_selection

from keras.utils import to_categorical

from keras import models, layers

from keras.preprocessing.image import ImageDataGenerator

import seaborn as sns

from sklearn.metrics import confusion_matrix
train=pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test=pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

labels=train['label']

train=train.drop(['label'],axis=1)
train=train.astype('float32')/255

test=test.astype('float32')/255

def reshapeAsImages(dataFramePixels,rows,columns):

    return dataFramePixels.values.reshape((len(dataFramePixels),rows,columns,1))

train_images=reshapeAsImages(train,28,28)

test_images=reshapeAsImages(test,28,28)
n=121

plt.imshow(train_images[n][:,:,0],cmap='gray_r')

plt.title("Train image labelled {}".format(labels[n]))

plt.axis('off')

plt.show()
train_x,valid_x,train_labels,valid_labels = model_selection.train_test_split(train_images, labels, test_size=0.01, random_state=20, shuffle=True)
print("train shape: {}, valid shape: {}, test shape{}".format(train_x.shape,valid_x.shape,test.shape))
train_y=to_categorical(train_labels)

valid_y=to_categorical(valid_labels)
tune_dropout=0.15

tune_convolutionFilters=64

tune_epochs=30

tune_batchsize=64



model = models.Sequential()



# Typical sequence

model.add(layers.Conv2D(filters=tune_convolutionFilters,

                        kernel_size=(3,3),

                        activation='relu',

                        input_shape=(28,28,1)))  # Now 'image' is 26x26

model.add(layers.MaxPooling2D((2,2)))            # Now 'image' is 13x13

model.add(layers.BatchNormalization())

model.add(layers.Dropout(tune_dropout)) # Arbitrarily set a portion of inputs to 0 to avoid overfitting



# Typical sequence

model.add(layers.Conv2D(filters=tune_convolutionFilters,

                        kernel_size=(3,3),

                        activation='relu')) # Now 'image' is 11x11

model.add(layers.MaxPooling2D((2,2)))       # Now 'image' is 5x5

model.add(layers.BatchNormalization())

model.add(layers.Dropout(tune_dropout)) # Arbitrarily set a portion of inputs to 0 to avoid overfitting



# Typical sequence

model.add(layers.Conv2D(filters=tune_convolutionFilters,

                        kernel_size=(3,3),

                        activation='relu')) # Now 'image' is 3x3

#model.add(layers.MaxPooling2D((2,2)))          

model.add(layers.BatchNormalization())

model.add(layers.Dropout(tune_dropout)) # Arbitrarily set a portion of inputs to 0 to avoid overfitting



# Last sequence

model.add(layers.Flatten()) # not an 'image' anymore but a linear sequence

model.add(layers.Dense(units=tune_convolutionFilters,activation='relu'))

model.add(layers.BatchNormalization())

model.add(layers.Dense(units=10,activation='softmax')) # We need as many dimensions as we have labels: 10



model.summary()
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
dataGenerator = ImageDataGenerator(

        featurewise_center=False,  

        samplewise_center=False,  

        featurewise_std_normalization=False,  

        samplewise_std_normalization=False, 

        rotation_range=15,      # symbols for numbers are oriented. Limited rotation allowable

        zoom_range = 0.1,       # Scaling for apparent size x0.9 to x1.1

        width_shift_range=3,  # about +/- 2 pixels (about 10% of width)

        height_shift_range=3, # about +/- 2 pixels (about 10% of height)

        horizontal_flip=False,  # symbols for numbers are oriented. No flip possible

        vertical_flip=False)



#dataGenerator.fit(train_x) # Not necessary since we do not use the featurewise_xxx

# For use without the dataGenerator

'''

model.fit(x=train_x,y=train_y,

          epochs=tune_epochs,

          batch_size=tune_batchsize,

          validation_data=(valid_x,valid_y))

'''



# For use with the dataGenerator

model.fit_generator(dataGenerator.flow(train_x, train_y, batch_size=tune_batchsize),

                    epochs=tune_epochs,

#                    batch_size=tune_batchsize,

                    steps_per_epoch=int(len(train_x)/tune_batchsize)+1,

                    validation_data=(valid_x, valid_y))

# Save model (architecture + weights) to JSON

model_json = model.to_json()

with open("model.json", "w") as json_file:

    json_file.write(model_json)

# serialize weights to HDF5

model.save_weights("model.h5")
pred_test_y = model.predict(test_images)

# The prediction is a vector of dimension n. We select the largest component (the best 'direction' we can imagine)

# Each input image will be associated to the 'closest' label (number) even if is not a number at all.

pred_test_labels = pred_test_y.argmax(axis=1)

submission=pd.DataFrame({'ImageId': list(range(1,len(pred_test_labels)+1)),

                         'Label': pred_test_labels})

submission.to_csv("submission.csv", columns=['ImageId','Label'], index=False, header=True)

submission.head()
conf_mat =  confusion_matrix(train_labels,model.predict(train_x).argmax(axis=1))

f,ax = plt.subplots(figsize=(7, 7))

sns.heatmap(conf_mat, cmap='Blues',annot=True, linewidths=.5, fmt= '.0f',ax=ax)
history=model.history.history

accuracy = history['acc']

val_accuracy = history['val_acc']

loss = history['loss']

val_loss = history['val_loss']

epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')

plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend()

plt.show()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')

plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

plt.legend()

plt.show()
# load json and create model

json_file = open('model.json', 'r')

loaded_model_json = json_file.read()

json_file.close()

model = models.model_from_json(loaded_model_json)

# load weights into new model

model.load_weights("model.h5")

 

# evaluate loaded model on test data

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])