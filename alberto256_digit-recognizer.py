from tensorflow.keras.models                     import Sequential
from tensorflow.keras.layers                     import Dense, Conv2D, MaxPooling2D, Dropout, Flatten 
from tensorflow.keras                            import utils
from tensorflow.keras.preprocessing              import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks                  import ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection                     import train_test_split
from google.colab                                import files
import numpy as np
import os
import matplotlib.pyplot as plt
file = files.upload()
p = os.path.abspath('kaggle.json')
print(p)
!pip install kaggle
!mkdir ~/.kaggle
p = os.path.abspath('.kaggle')
print(p)
!ls
!mv kaggle.json ~/.kaggle
!kaggle competitions download -c digit-recognizer
!ls
!unzip '/content/train.csv.zip'
!unzip '/content/test.csv.zip'
train_dataset = np.loadtxt('train.csv', skiprows=1, delimiter=',')
train_dataset[0:5]
x_train = train_dataset[:, 1:]
# Переформатируем данные в 2D, бэкенд TensorFlow
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
input_shape = (28, 28, 1) 
x_train /= 255.0
x_train[1].shape
y_train = train_dataset[:, 0]
y_train[:5]
y_train = utils.to_categorical(y_train)

random_seed = 10
X_train, X_val, Y_train, Y_val = train_test_split(x_train, y_train, test_size = 0.1, random_state=random_seed)
datagen = ImageDataGenerator(
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


datagen.fit(X_train)
i = 0
data = X_train[0]
data = np.expand_dims(data, axis=0)
for batch in datagen.flow(data, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(batch[0][:,:,0])
    i += 1
    if i % 6 == 0:
        break
plt.show()
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

print(model.summary())
сheckpoint = ModelCheckpoint('content/mnist-cnn.h5', 
                              monitor='val_acc', 
                              save_best_only=True,
                              verbose=1)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
batch_size = 256
history = model.fit(datagen.flow(X_train,Y_train, batch_size=batch_size), 
                    epochs=50,
                    validation_data=(X_val, Y_val),
                    steps_per_epoch=X_train.shape[0] // batch_size,
                    verbose=1,
                    callbacks=[сheckpoint, learning_rate_reduction])
plt.plot(history.history['accuracy'], 
         label='Share of correct answers on educational set')
plt.plot(history.history['val_accuracy'], 
         label='Share of correct answers on checking set')
plt.xlabel('Epochs')
plt.ylabel('share of correct answers')
plt.legend()
plt.show()
test_dataset = np.loadtxt('test.csv', skiprows=1, delimiter=",")
x_test = test_dataset.reshape(test_dataset.shape[0], 28, 28, 1)
x_test = x_test / 255.0
predictions = model.predict(x_test)
predictions[:5]
predictions = np.argmax(predictions, axis=1)
predictions[:5]
out = np.column_stack((range(1, predictions.shape[0]+1), predictions))
np.savetxt('submission.csv', out, header="ImageId,Label", 
            comments="", fmt="%d,%d")
!kaggle competitions submit -c digit-recognizer -m "Submition from Colab" -f submission.csv