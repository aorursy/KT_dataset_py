import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import cv2
from skimage import feature
from skimage import measure
os.listdir('/kaggle/input/thai-mnist-classification')
train_img_path = '/kaggle/input/thai-mnist-classification/train'
train_label_path = '/kaggle/input/thai-mnist-classification/mnist.train.map.csv'
pd.read_csv(train_label_path).set_index('id').to_dict()['category']

os.listdir(train_img_path)
pd.read_csv(train_label_path)
class getdata():
    def __init__(self,data_path,label_path=None):
        self.dataPath = data_path
        self.dataFile = os.listdir(self.dataPath)
        self.n_index = len(self.dataFile)
        if label_path is not None:
            self.labelPath = label_path
            self.label_dict = pd.read_csv(train_label_path).set_index('id').to_dict()['category']
            self.label = [self.label_dict[x] for x in self.dataFile]
        
        
    
    def get1img(self,img_index,mode='rgb',label = False):
        img = cv2.imread( os.path.join(self.dataPath,self.dataFile[img_index]) )
        if mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif mode == 'gray':
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if label:
            return img,self.label[img_index]
        return img
gdt = getdata(train_img_path,train_label_path)
plt.gray()
from skimage.morphology import convex_hull_image
from skimage.util import invert
temp_img = invert(gdt.get1img(1234,'gray'))
fig, [ax1,ax2] = plt.subplots(1, 2)
ax1.imshow(temp_img)
cvh =  convex_hull_image(temp_img)
ax2.imshow(cvh)
def convex_crop(img,pad=20):
    convex = convex_hull_image(img)
    r,c = np.where(convex)
    while (min(r)-pad < 0) or (max(r)+pad > img.shape[0]) or (min(c)-pad < 0) or (max(c)+pad > img.shape[1]):
        pad = pad - 1
    return img[min(r)-pad:max(r)+pad,min(c)-pad:max(c)+pad]
crop_img = convex_crop(temp_img,pad=10)
plt.imshow(crop_img)
def convex_resize(img):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = cv2.resize(img,(32,32))
    return img
def thes_resize(img,thes=40):
    img = invert(img)
    img = convex_crop(img,pad=20)
    img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 300):
        img = cv2.resize(img,(300,300))
        img = ((img > thes)*255).astype(np.uint8)
    if(min(img.shape) > 150):
        img = cv2.resize(img,(150,150))
        img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(80,80))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(50,50))
    img = ((img > thes)*255).astype(np.uint8)
    img = cv2.resize(img,(32,32))
    img = ((img > thes)*255).astype(np.uint8)
    return img
temp_img = gdt.get1img(128,'gray')
fig, [ax1,ax2] = plt.subplots(1, 2,figsize=(10,7))
ax1.imshow(convex_resize(temp_img))
ax1.set_title('Without thresholding')
ax2.imshow(thes_resize(temp_img))
ax2.set_title('Thresholding')
fig, ax = plt.subplots(5, 5, figsize=(15,15))
for i in range(5):
    for j in range(5):
        img_index = np.random.randint(0,gdt.n_index)
        ax[i][j].imshow(thes_resize(gdt.get1img(img_index,'gray')))
        ax[i][j].set_title('Class: '+str(gdt.label[img_index]))
        ax[i][j].set_axis_off()
from tqdm import tqdm
X = []
for i in tqdm(range(gdt.n_index)):
    X.append(thes_resize(gdt.get1img(i,'gray')))
X = np.array(X)
y =np.array(gdt.label)
X = X.reshape((-1,32,32,1))
X.shape,y.shape
import tensorflow as tf
y_cat = tf.keras.utils.to_categorical(y)
y_cat.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.25, random_state=1234)
X_train = X_train / 255.
X_test = X_test / 255.
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(6, (3,3), input_shape=(32, 32, 1), activation='relu'))
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Conv2D(16, (5,5), activation='relu')) 
model.add(tf.keras.layers.MaxPool2D()) 
model.add(tf.keras.layers.Flatten()) 
model.add(tf.keras.layers.Dropout(0.5)) #Add dropout for prevent overfit
model.add(tf.keras.layers.Dense(120, activation='relu'))
model.add(tf.keras.layers.Dense(84, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
model.summary()
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', 
                                            patience=5, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15,verbose=1)
history = model.fit(X_train, y_train, batch_size=64,validation_data=(X_test,y_test), 
                    epochs=100, callbacks=[learning_rate_reduction])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
