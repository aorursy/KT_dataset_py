from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import os
import cv2
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
def pixalate_image(image, scale_percent = 40):
    width = int(128 * 40 / 100)
    height = int(128 * 40 / 100)
    dim = (width, height)

    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
    # scale back to original size
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)

    low_res_image = cv2.resize(small_image, dim, interpolation = cv2.INTER_AREA)

    return low_res_image
# x is noisy data and y is clean data
SIZE = 256
X_train=[]
Y_train=[]
path1 = '../input/chest-xray-pneumonia/chest_xray/train/NORMAL/'
files=os.listdir(path1)
for i in tqdm(files):
    img=cv2.imread(path1+'/'+i,0)   #Change 0 to 1 for color images
    img1=pixalate_image(img)
    img1=cv2.resize(img1,(256, 256))  
    img2=cv2.resize(img,(256, 256)) 
    X_train.append(img_to_array(img1))
    Y_train.append(img_to_array(img2))
    

X_test=[]
Y_test=[]
path1 = '../input/covidct/COVID-CT/CT_COVID/'
files=os.listdir(path1)
for i in tqdm(files):
    img=cv2.imread(path1+'/'+i,0)   #Change 0 to 1 for color images
    img1=pixalate_image(img)
    img1=cv2.resize(img1,(256, 256))  
    img2=cv2.resize(img,(256, 256)) 
    X_test.append(img_to_array(img1))
    Y_test.append(img_to_array(img2))
    
X_train = np.reshape(X_train, (len(X_train), SIZE, SIZE, 1))
X_train = X_train.astype('float32') / 255.

Y_train = np.reshape(Y_train, (len(Y_train), SIZE, SIZE, 1))
Y_train = Y_train.astype('float32') / 255.

X_test = np.reshape(X_test, (len(X_test), SIZE, SIZE, 1))
X_test = X_test.astype('float32') / 255.

Y_test = np.reshape(Y_test, (len(Y_test), SIZE, SIZE, 1))
Y_test = Y_test.astype('float32') / 255.

#Displaying images with noise
plt.figure(figsize=(10, 2))
for i in range(1,4):
    ax = plt.subplot(1, 4, i)
    plt.imshow(X_test[i].reshape(256, 256), cmap="gray")
plt.show()

#Displaying clean images
plt.figure(figsize=(10, 2))
for i in range(1,4):
    ax = plt.subplot(1, 4, i)
    plt.imshow(Y_test[i].reshape(256, 256), cmap="gray")
plt.show()

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(SIZE, SIZE, 1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
 

model.add(MaxPooling2D((2, 2), padding='same'))
     
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='relu', padding='same'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

model.summary()
history =model.fit(X_train, Y_train, epochs=10, batch_size=8, shuffle=True, verbose = 1,
          validation_split = 0.1)


print("Test_Accuracy: {:.2f}%".format(model.evaluate(np.array(X_test), np.array(Y_test))[1]*100))


model.save('Reso_autoencoder.model')

Reso_img = model.predict(X_test)


plt.figure(figsize=(40, 4))
for i in range(10):
    # display original
    ax = plt.subplot(3, 20, i + 1)
    plt.imshow(Y_test[i].reshape(SIZE,SIZE), cmap="gray")
    
    # display reconstructed (after noise removed) image
    ax = plt.subplot(3, 20, 40 +i+ 1)
    plt.imshow(Reso_img[i].reshape(SIZE,SIZE), cmap="gray")
plt.show()
#plotting training values

sns.set()
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)

#accuracy plot
plt.plot(epochs, accuracy, color='red', label='Training Accuracy')
plt.plot(epochs, val_accuracy, color='blue', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.figure()

#loss plot
plt.plot(epochs, loss, color='purple', label='Training Loss')
plt.plot(epochs, val_loss, color='green', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()