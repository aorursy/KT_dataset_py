import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from tqdm import tqdm

from sklearn.model_selection import train_test_split

%matplotlib inline
#With this backend, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, 
#directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.


train_df = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test_df = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')
#read CSV file
IMG_SIZE = 224
#set image size
def load_img(path):
    image = cv2.imread(path)
    return image

"""
Argument: path to image
Output: Image (cv2)
"""
fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)#select 9 random images
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()): #array.ravel() ;return flattened array
    img = load_img(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axis.imshow(img)
def remove_unwanted_space(image, threshold=7):# remove black area
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray_image > threshold
    return image[np.ix_(mask.any(1), mask.any(0))]
def preprocess_img(path):
    image = load_img(path)
    image = remove_unwanted_space(image, 5)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image,4, cv2.GaussianBlur(image, (0,0), 30), -4, 128)# doan nay kho, no lam tien xu li cho anh sang hon
    #https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    #https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html 
    return image
fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()):
    img = preprocess_img(image)
    axis.imshow(img)
def circle_crop(img):
    circle_img = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8) # create matrix of 0
    cv2.circle(circle_img, ((int)(IMG_SIZE/2),(int)(IMG_SIZE/2)), int(IMG_SIZE/2), 1, thickness=-1)
    #image = cv2.circle(image, center_coordinates, radius, color, thickness) 
    img = cv2.bitwise_and(img, img, mask=circle_img)
    return img
fig, axes = plt.subplots(3,3,figsize=(10,10))
selection = np.random.choice(train_df.index, size=9, replace=False)
images = '../input/aptos2019-blindness-detection/train_images/'+train_df.loc[selection]['id_code']+'.png'
for image, axis in zip(images, axes.ravel()):
    img = circle_crop(preprocess_img(image))
    axis.imshow(img)
N = train_df.shape[0]
train = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(train_df['id_code'])):
    train[i,:,:,:] = circle_crop(preprocess_img('../input/aptos2019-blindness-detection/train_images/'+image_id+'.png'))
N = test_df.shape[0]
test = np.empty((N, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
for i, image_id in enumerate(tqdm(test_df['id_code'])):
    test[i,:,:,:] = circle_crop(preprocess_img('../input/aptos2019-blindness-detection/test_images/'+image_id+'.png'))
X_train, X_val, y_train, y_val = train_test_split(train, train_df['diagnosis'], test_size=0.15, random_state=42)# function to automatically split train/test
BATCH_SIZE = 32 #batch zsize
train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        zoom_range=[0.9, 1.0],
        fill_mode='constant',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        rotation_range=120
    )#data augumentation
val_data_generator = tf.keras.preprocessing.image.ImageDataGenerator()
train_gen = train_data_generator.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_gen = val_data_generator.flow(X_val, y_val, batch_size=BATCH_SIZE)
resnet = tf.keras.applications.ResNet50(include_top=False, weights='../input/weight/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(IMG_SIZE, IMG_SIZE, 3))
#include top: include the 3 fully-connected layers at the top of the network.
model = tf.keras.Sequential([
    resnet,
    tf.keras.layers.GlobalAveragePooling2D(),
    #pooling layer (được dùng giữa các convolutional layer,)
    tf.keras.layers.BatchNormalization(),
    #Batch Normalize đã normalize dữ liệu về zero mean, và chuẩn hoá variance về dạng unit. Bằng cách này đã giải quyết được rất tốt các vấn đề kể trên. Những ưu điểm của batch norm như sau:
    #Cho phép learning rate lớn [vấn đề 1]
    #Giảm sự phụ thuộc vào việc initialize [vấn đề 2]
    tf.keras.layers.Dropout(0.5),
    #skip connection 
    #Lý giải cho việc này, bạn có thể đọc bài báo http://papers.nips.cc/paper/4878-understanding-dropout.pdf.
    tf.keras.layers.Dense(256, activation='relu'),
    #Just your regular densely-connected NN layer.
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='relu')
    #Just your regular densely-connected NN layer.
])
checkpoint = tf.keras.callbacks.ModelCheckpoint('model_weights.hdf5', monitor='val_loss', verbose=1, 
                             save_best_only=True, mode='min')
#load check point
#verbose=1 animate progress bar
optimizer = tf.keras.optimizers.Adam(lr=0.00005)
# just a regular optimizer
model.compile(optimizer=optimizer, loss='mse')
# MSE : Mean square error ( a regression loss function)
#https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0?gi=8cbfdfe4e200
#Once the model is created, you can config the model with losses and metrics with model.compile(), train the model with model.fit(), or use the model to do prediction with model.predict().
steps_per_epoch = int(np.ceil(X_train.shape[0]/BATCH_SIZE))
val_steps_per_epoch = int(np.ceil(X_val.shape[0]/BATCH_SIZE))
history = model.fit(train_gen, validation_data=val_gen, steps_per_epoch=steps_per_epoch, 
                              validation_steps=val_steps_per_epoch, callbacks=[checkpoint], epochs=25)
model.load_weights('model_weights.hdf5')
prediction = model.predict(test)
for i, pred in enumerate(prediction):
    if pred < 0.5:
        prediction[i] = 0
    elif pred < 1.5:
        prediction[i] = 1
    elif pred < 2.5:
        prediction[i] = 2
    elif pred < 3.5:
        prediction[i] = 3
    else:
        prediction[i] = 4
prediction = np.squeeze(prediction.astype(np.int8))
#https://www.tutorialspoint.com/numpy/numpy_squeeze.htm
# removes one-dimensional entry from the shape of the given array
sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = prediction
sample.to_csv("submission.csv", index=False)