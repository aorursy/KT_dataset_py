from keras.layers import Conv2D, UpSampling2D, Dropout
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.io import imshow
from skimage.color import rgb2lab, lab2rgb, gray2rgb
from skimage.transform import resize
import skimage.io
from skimage.io import imsave
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
import os
import glob
import seaborn as sns
import pandas as pd
import cv2
#for train image pixel 
cy = []
cx = []
for i in (glob.glob("../input/flickr-image-dataset/flickr30k_images/flickr30k_images/*.jpg")):
    img = plt.imread(i)
    a = np.shape(img)
    c = np.reshape(img,(a[0]*a[1],a[2]))
    cy.append(np.shape(c)[0])
    cx.append(i)
columns = ['Images','pixels']
dt = np.array([cx,cy])
df = pd.DataFrame(dt.T, columns = columns)
df['pixels'] = df['pixels'].astype('int')
df = df.sort_values('pixels')
df.head()
# sns.set(style="darkgrid")
# mortality_age_plot = sns.barplot(x=df['Images'],
#                                  y=df['pixels'],
#                                  palette = 'muted',
#                                  order=df['Images'].tolist())

# plt.xticks(rotation=90)
# plt.show()
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))

for i in range(9):
    img = cv2.imread(df['Images'][i])
    ax[i // 3, i % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(cat_train_df['Images'][i])
  
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))

for i in range(9):
    img = cv2.imread(df['Images'][i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,100)
    ax[i // 3, i % 3].imshow(edges)
    print(df['Images'][i])
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
j = 0
for i in range((len(df['Images'])-1),(len(cat_train_df['Images'])-10),-1):
    img = cv2.imread(df['Images'][i])
    ax[j // 3, j % 3].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print(df['Images'][i])
    j += 1
new_style = {'grid': False}
plt.rc('axes', **new_style)
_, ax = plt.subplots(3, 3, sharex='col', sharey='row', figsize=(12, 12))
j = 0
for i in range((len(df['Images'])-1),(len(df['Images'])-10),-1):
    img = cv2.imread(df['Images'][i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,200,100)
    ax[j // 3, j % 3].imshow(edges)
    print(df['Images'][i])
    j += 1
def pixel_matrix(path):
    image = plt.imread(path)
    dims = np.shape(image)
    return np.reshape(image, (dims[0] * dims[1], dims[2]))# changing shape
def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

count = 0
for imagePath in df['Images']:
    # load the image, convert it to grayscale, and compute the
    # focus measure of the image using the Variance of Laplacian
    # method
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)

    # if the focus measure is less than the supplied threshold,
    # then the image should be considered "blurry"
    
    if fm < 110.0:
        count += 1
        
print("Total blur image is ",count)
path = '../input/flickr-image-dataset'
#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)
#Resize images, if needed
train = train_datagen.flow_from_directory(path, 
                                          target_size=(256, 256), 
                                          batch_size=340, 
                                          class_mode=None)
#iterating on each image and covert the RGB to Lab.
X =[]
Y =[]
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:,:,0]) 
        Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
      #so we divide the values by 128 to restrict values to between -1 and 1.
    except:
        print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)
#Encoder
model = Sequential()
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', strides=2, input_shape=(256, 256, 1)))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same', strides=2))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(512, (3,3), activation='relu', padding='same'))
model.add(Conv2D(256, (3,3), activation='relu', padding='same'))
#Decoder
#Decoder
#Note: For the last layer we use tanh instead of Relu. 
#This is because we are colorizing the image in this layer using 2 filters, A and B.
#A and B values range between -1 and 1 so tanh (or hyperbolic tangent) is used
#as it also has the range between -1 and 1. 
#Other functions go from 0 to 1.
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.compile(optimizer='adam', loss='mse' , metrics=['accuracy'])
model.summary()
history=model.fit(X,Y,validation_split=0.1, epochs=150, batch_size=16)
model.save('other_files/colorize_autoencoder.model')
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Model')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Model')
plt.legend()
plt.show()
tf.keras.models.load_model(
    'other_files/colorize_autoencoder.model',
    custom_objects=None,
    compile=True)
img1_color=[]
img1=img_to_array(load_img('../input/flickr-image-dataset/flickr30k_images/flickr30k_images/1000268201.jpg'))
img1 = resize(img1 ,(256,256))
img1_color.append(img1)
img1_color = np.array(img1_color, dtype=float)
img1_color = rgb2lab(1.0/255*img1_color)[:,:,:,0]
img1_color = img1_color.reshape(img1_color.shape+(1,))
output1 = model.predict(img1_color)
output1 = output1*128
result = np.zeros((256, 256, 3))
result[:,:,0] = img1_color[0][:,:,0]
result[:,:,1:] = output1[0]
imshow(lab2rgb(result))
imsave("result.png", lab2rgb(result))
vggmodel = VGG16()
newmodel = Sequential()
#num = 0
for i, layer in enumerate(vggmodel.layers):
    if i<19:          #Only up to 19th layer to include feature extraction only
        newmodel.add(layer)
newmodel.summary()
for layer in newmodel.layers:
    layer.trainable=False   #We don't want to train these layers again, so False. 
#Normalize images - divide by 255
train_datagen = ImageDataGenerator(rescale=1. / 255)
train = train_datagen.flow_from_directory(path, target_size=(224, 224), batch_size=32, class_mode=None)
#Convert from RGB to Lab
X =[]
Y =[]
for img in train[0]:
    try:
        lab = rgb2lab(img)
        X.append(lab[:,:,0]) 
        Y.append(lab[:,:,1:] / 128) #A and B values range from -127 to 128, 
      #so we divide the values by 128 to restrict values to between -1 and 1.
    except:
        print('error')
X = np.array(X)
Y = np.array(Y)
X = X.reshape(X.shape+(1,)) #dimensions to be the same for X and Y
print(X.shape)
print(Y.shape)
#now we have one channel of L in each layer but, VGG16 is expecting 3 dimension, 
#so we repeated the L channel two times to get 3 dimensions of the same L channel

vggfeatures = []
for i, sample in enumerate(X):
    sample = gray2rgb(sample)
    sample = sample.reshape((1,224,224,3))
    prediction = newmodel.predict(sample)
    prediction = prediction.reshape((7,7,512))
    vggfeatures.append(prediction)
vggfeatures = np.array(vggfeatures)
print(vggfeatures.shape)
#Decoder
model = Sequential()

model.add(Conv2D(256, (3,3), activation='relu', padding='same', input_shape=(7,7,512)))
model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.add(UpSampling2D((2, 2)))
model.summary()
model.compile(optimizer='Adam', loss='mse' , metrics=['accuracy'])
history=model.fit(vggfeatures, Y,validation_split=0.1 ,verbose=1, epochs=1000, batch_size=128)

model.save('colorize_autoencoder_VGG16.model')
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Test accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.title('Accuracy Model')
plt.legend()
plt.show()
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Test loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Loss Model')
plt.legend()
plt.show()
#Predicting using saved model.
model = tf.keras.models.load_model('../input/vgg16-colorize-autoencoder/colorize_autoencoder_VGG16_10000.model',
                                   custom_objects=None,
                                   compile=True)
# testpath = '/kaggle/input'
# files = os.listdir(testpath)
# for idx, file in enumerate(testpath):
test = img_to_array(load_img('../input/flickr-image-dataset/flickr30k_images/flickr30k_images/1000268201.jpg'))
test = resize(test, (224,224), anti_aliasing=True)
test*= 1.0/255
lab = rgb2lab(test)
l = lab[:,:,0]
L = gray2rgb(l)
L = L.reshape((1,224,224,3))
#print(L.shape)
vggpred = newmodel.predict(L)
ab = model.predict(vggpred)
#print(ab.shape)
ab = ab*128
cur = np.zeros((224, 224, 3))
cur[:,:,0] = l
cur[:,:,1:] = ab
imshow(lab2rgb(cur))
# imsave('images/colorization2/vgg_result/result'+str(idx)+".jpg", lab2rgb(cur))