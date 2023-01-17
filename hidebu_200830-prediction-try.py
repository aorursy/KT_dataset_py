# initiating gpu using tensorflow.
import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#config.log_device_placement = True


#sess = tf.Session(config=config)
#set_session(sess)
#!pip install albumentations > /dev/null
#import albumentations
#importing libraries for the data processing and model.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import random
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from keras.models import load_model
%matplotlib inline
# defining the path and classes.
directory = '../input/state-farm-distracted-driver-detection/imgs/train'
test_directory = '../input/state-farm-distracted-driver-detection/imgs/test/'
random_test = '../input/driver/'
classes = ['c0','c1','c2','c3','c4','c5','c6','c7','c8','c9']
# defining a shape to be used for our models.
img_size1 = 64
img_size2 = 64
# Train class image for display.
for i in classes:
    path = os.path.join(directory,i)
    for img in os.listdir(path):
        print(img)
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap='gray')
        plt.show()
        break
    break
os.listdir(path)
# Test class image for display.
test_array = []
for img in os.listdir(test_directory):
    img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
    test_array = img_array
    plt.imshow(img_array, cmap='gray')
    plt.show()
    break
img
# checkking image size using shape.
print(img_array.shape)
# trying out the resize image functionality
new_img = cv2.resize(test_array,(img_size2,img_size1))
plt.imshow(new_img,cmap='gray')
plt.show()
#flipping_horizontal (8/30add)
hflip_img = cv2.flip(new_img, 1)
plt.imshow(hflip_img,cmap='gray')
plt.show()
#flipping_vertical (8/30add)
vflip_img = cv2.flip(new_img, 0)
plt.imshow(vflip_img,cmap='gray')
plt.show()
#flipping_horizontal and vertical (8/30add)
hvflip_img = cv2.flip(new_img, -1)
plt.imshow(hvflip_img,cmap='gray')
plt.show()
#blur (8/30add)
blur_img = cv2.blur(new_img, (5,5))

plt.imshow(blur_img,cmap='gray')
plt.show()
#Gaussian blur (8/30add)
gau_img = cv2.GaussianBlur(new_img, (5,5), 0)

plt.imshow(gau_img,cmap='gray')
plt.show()
#median blur (8/30add)
med_img = cv2.medianBlur(new_img, 5)

plt.imshow(med_img,cmap='gray')
plt.show()
#Binarization (8/30add)
ret, bin_img = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)

plt.imshow(bin_img,cmap='gray')
plt.show()
#Erosion (8/30add)
kernel = np.ones((10,10), np.uint8)
img_el = cv2.erode(new_img, kernel, iterations=1)

plt.imshow(img_el,cmap='gray')
plt.show()
#Dilation (8/30add)
kernel = np.ones((5,5), np.uint8)
img_dl = cv2.dilate(new_img, kernel, iterations=1)

plt.imshow(img_dl,cmap='gray')
plt.show()
#Opening (8/30add)
kernel = np.ones((5,5), np.uint8)
img_op = cv2.morphologyEx(new_img, cv2.MORPH_OPEN, kernel)

plt.imshow(img_op,cmap='gray')
plt.show()
#Closing (8/30add)
kernel = np.ones((5,5), np.uint8)
img_cl = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, kernel)

plt.imshow(img_cl,cmap='gray')
plt.show()
#GaussianNoise (8/30add)
def addGaussianNoise(new_img):
    row,col= new_img.shape
    mean = 0
    var = 0.1
    sigma = 100
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = new_img + gauss

    return noisy


gau_noi_img = addGaussianNoise(new_img)
plt.imshow(gau_noi_img,cmap='gray')
plt.show()
#Salt_Pepper_Noise (8/30add)
def add_Salt_Pepper_Noise(new_img, s_vs_p = 0.5, amount = 0.05):
    row,col = new_img.shape
    s_and_p = np.copy(new_img)
    # Salt mode
    num_salt = np.ceil(amount * new_img.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in new_img.shape]
    s_and_p[coords] = 1

    # Pepper mode
    num_pepper = np.ceil(amount* new_img.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in new_img.shape]
    s_and_p[coords] = 0
    return s_and_p

sap_noi_img = add_Salt_Pepper_Noise(new_img, s_vs_p = 0.5, amount = 0.05)
plt.imshow(sap_noi_img,cmap='gray')
plt.show()
#NegaPosiDiverse (8/30add)
ngp_img = cv2.bitwise_not(new_img)

plt.imshow(ngp_img,cmap='gray')
plt.show()
#Canny (8/30add)
canny_img = cv2.Canny(new_img, 200, 200)

plt.imshow(canny_img,cmap='gray')
plt.show()
#Rotation
height,width = new_img.shape[:2]
center = (int(width/2), int(height/2)) # 中心点
angle = 45 # 左回転
M = cv2.getRotationMatrix2D(center, angle, 1)
rotated_img = cv2.warpAffine(new_img, M, (width, height))
plt.imshow(rotated_img,cmap='gray')
plt.show()
#Shifted
moving_x = -10
moving_y = -10
M = np.float32([[1, 0, moving_x], [0, 1, moving_y]])
shifted_img = cv2.warpAffine(new_img, M, (width, height))
plt.imshow(shifted_img,cmap='gray')
plt.show()
os.listdir(test_directory)
img
test_directory
try_img_size1 = 128
try_img_size2 = 128
img = 'img_58997.jpg'

test_array = []
img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
test_array = img_array
plt.imshow(img_array, cmap='gray')
plt.show()
# trying out the resize image functionality
new_img = cv2.resize(test_array,(try_img_size2,try_img_size1))
plt.imshow(new_img,cmap='gray')
plt.show()
#Canny (8/30add)
canny_img = cv2.Canny(new_img, 150, 150)

plt.imshow(canny_img,cmap='gray')
plt.show()
#NegaPosiDiverse (8/30add)
ngp_img = cv2.bitwise_not(new_img)

plt.imshow(ngp_img,cmap='gray')
plt.show()
#Binarization (8/30add)
ret, bin_img = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)

plt.imshow(bin_img,cmap='gray')
plt.show()
#NegaPosi -> GaussianNoise (8/30add)
gau_noi_img = addGaussianNoise(ngp_img)
plt.imshow(gau_noi_img,cmap='gray')
plt.show()
#NegaPosi -> Salt and Pepper (8/30add)
sap_noi_img = add_Salt_Pepper_Noise(ngp_img, s_vs_p = 0.5, amount = 0.15)
plt.imshow(sap_noi_img,cmap='gray')
plt.show()
#Binarization (8/30add)
ret, bin_img = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)

sap_noi_img = add_Salt_Pepper_Noise(bin_img, s_vs_p = 0.5, amount = 0.10)
plt.imshow(sap_noi_img,cmap='gray')
plt.show()

# creating a training dataset.
training_data = []
i = 0
def create_training_data():
    for category in classes:
        path = os.path.join(directory,category)
        class_num = classes.index(category)
        
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_img = cv2.resize(img_array,(img_size2,img_size1))
            
            #Additional preprocessing(8/30)
            #Binarization (8/30add)
            ret, bin_img = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)           
            training_data.append([
                bin_img,class_num])
            #Binarization　-> Salt and Pepper
            sap_img = add_Salt_Pepper_Noise(bin_img, s_vs_p = 0.5, amount = 0.01)
            training_data.append([
                sap_img,class_num])
            #Binarization　-> Salt and Pepper
            sap_img = add_Salt_Pepper_Noise(bin_img, s_vs_p = 0.5, amount = 0.03)
            training_data.append([
                sap_img,class_num])
            #Binarization　-> Salt and Pepper
            sap_img = add_Salt_Pepper_Noise(bin_img, s_vs_p = 0.5, amount = 0.05)
            training_data.append([
                sap_img,class_num])
            #Rotation
            height,width = new_img.shape[:2]
            center = (int(width/2), int(height/2)) # 中心点
            angle = 20 # 左回転
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_img = cv2.warpAffine(bin_img, M, (width, height))
            training_data.append([
                rotated_img,class_num])
            #Shifted
            moving_x = -10
            moving_y = -10
            M = np.float32([[1, 0, moving_x], [0, 1, moving_y]])
            shifted_img = cv2.warpAffine(bin_img, M, (width, height))
            training_data.append([
                shifted_img,class_num])
for category in classes:
    path = os.path.join(directory,category)
    class_num = classes.index(category)
        
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(img_size2,img_size1))
        training_data.append([
            new_img,class_num])
        
        print('path     :', path)
        print('img      :', img)
        print('img_array:', img_array)
        print('img_array_shape:', img_array.shape)
        print('new_img  :', new_img)
        print('new_img_shape:', new_img.shape)
        print('class_num:', class_num)
        
        break
training_data
training_data = []
i = 0
# Creating a test dataset.
testing_data = []
i = 0
def create_testing_data():        
    for img in os.listdir(test_directory):
        img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
        new_img = cv2.resize(img_array,(img_size2,img_size1))
        #Binarization (8/30add)
        ret, bin_img = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)          
        testing_data.append([img,
            bin_img])
for img in os.listdir(test_directory):
    img_array = cv2.imread(os.path.join(test_directory,img),cv2.IMREAD_GRAYSCALE)
    new_img = cv2.resize(img_array,(img_size2,img_size1))
    testing_data.append([img,
                         new_img])
    
    print('test_directory     :', test_directory)
    print('img      :', img)
    print('img_array:', img_array)
    print('img_array_shape:', img_array.shape)
    print('new_img  :', new_img)
    print('new_img_shape:', new_img.shape)
        
    break    
testing_data
testing_data = []
i = 0
import time
start = time.time()
create_training_data()
print('Elapsed_time: ', time.time()-start, '[sec]')
start = time.time()
create_testing_data()
print('Elapsed_time: ', time.time()-start, '[sec]')
print('training_data.size:', len(training_data))
print('testing_data.size :', len(testing_data))
random.shuffle(training_data)
x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)
print('features: ', x[0])
print('label   : ', y[0])
x[0].shape
len(x)
X = np.array(x).reshape(-1,img_size2,img_size1,1)
X.shape,X[0].shape
np.array(x).shape
X[0]
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=50)
Y_train = np_utils.to_categorical(y_train,num_classes=10)
Y_test = np_utils.to_categorical(y_test,num_classes=10)
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(img_size1,img_size2,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.3))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same'))
model.add(BatchNormalization(axis = 3))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units = 512,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(units = 128,activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(10,activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
callbacks = [EarlyStopping(monitor='val_acc',patience=5)]
batch_size = 50
n_epochs = 20
results = model.fit(x_train,Y_train,batch_size=batch_size,epochs=n_epochs,verbose=1,validation_data=(x_test,Y_test),callbacks=callbacks)
# Plot training & validation accuracy values
plt.plot(results.history['accuracy'])
plt.plot(results.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(results.history['loss'])
plt.plot(results.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
preds = model.predict(np.array(testing_data[0][1]).reshape(-1,img_size2,img_size1,1))
model.save_weights('./driverdistraction_lr_weights.h5', overwrite=True)
model.save('./driverdistraction_lr_weights.h5')
loaded_model = load_model('./driverdistraction_lr_weights.h5')
test_data = np.array(testing_data[1001][1]).reshape(-1,img_size2,img_size1,1)
test_data.shape
preds = model.predict(test_data)
#preds= np.argmax(preds)
preds
preds= np.argmax(preds)
preds
classes = {0: "safe driving",
1: "texting - right",
2: "talking on the phone - right",
3: "texting - left",
4: "talking on the phone - left",
5: "operating the radio",
6: "drinking",
7: "reaching behind",
8: "hair and makeup",
9: "talking to passenger",
}


for key,value in classes.items():
    if preds==key:
        predicted = value

predicted     
print(predicted)
new_img = cv2.resize(testing_data[1000][1],(img_size2,img_size1))
plt.imshow(new_img,cmap='gray')
plt.show()
testing_data
x_test=[]
y_test=[]

for test_id, feature in testing_data:
    x_test.append(feature)
    y_test.append(test_id)
print('features: ', x_test[0])
print('test_id : ', y_test[0])
X_test = np.array(x_test).reshape(-1,img_size2,img_size1,1)
X_test.shape,X_test[0].shape
np.array(x).shape
X_test[0]
preds = model.predict(X_test)
preds
preds.shape
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])
    result1.loc[:, 'img'] = pd.Series(test_id, index=result1.index)
    result1 = result1.sort_values(['img'])

    result1.to_csv(index=False)
    return result1
info = '200824'
submission = create_submission(preds, y_test, info)
submission.head()
now = datetime.datetime.now()

if not os.path.isdir('subm'):
    os.mkdir('subm')
suffix = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
sub_file = os.path.join('subm', 'submission_' + suffix + '.csv')
submission.to_csv('submission.csv', index=False)

