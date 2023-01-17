import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

data = np.load('/kaggle/input/cuboulder-image-labelling/train_and_test.npz')
X_train, y_train, X_test = data['X_train'], data['y_train'], data['X_test']
def show(point, class_name=None, ax=plt):
    ax.imshow(point)
    if ax == plt:
        ax.xticks([])
        ax.yticks([])
    else:
        ax.set_title('class: %s' % class_name)
        ax.set_xticks([])
        ax.set_yticks([])

def show_tile(points, labels, w, h, start=0):
    fig, axs = plt.subplots(w, h, figsize=(20,20))
    for i, point in enumerate([(x,y) for x in range(0, w) for y in range(0,h)]):
        show(points[i + start], labels[i + start], axs[point])

show_tile(X_train, y_train, 10, 6)
# detecting circles with opencv 
# modified from https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
import cv2 


def blur(img):
    return cv2.blur(img, (3,3))


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def circles(img, draw_on_image=False):
    gray_blurred = blur(gray(img))
    
    detected_circles = None
    
    min_thresh = 20
    while min_thresh > 1 and detected_circles is None:
        min_thresh -= 1
        detected_circles = cv2.HoughCircles(gray_blurred, 
                                            method=cv2.HOUGH_GRADIENT, 
                                            dp=1, 
                                            minDist=20, 
                                            param1=20,
                                            param2=min_thresh,
                                            minRadius=7,
                                            maxRadius=30) 
    # Convert the circle parameters a, b and r to integers. 
    detected_circles = np.uint16(np.around(detected_circles)) 
    if draw_on_image:
        out = img.copy()
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 

            # Draw the circumference of the circle. 
            cv2.circle(out, (a, b), r, (0, 255, 0), 2) 

            # Draw a small circle (of radius 1) to show the center. 
            cv2.circle(out, (a, b), 1, (0, 0, 255), 3) 
        return out
    else:
        return detected_circles[0][0]

    
# https://stackoverflow.com/a/47629363/2821370
def background_subtract_circle(img, circle):
    x,y,r = circle
    mask = np.zeros_like(img[:,:,:])
    cv2.circle(mask, (x,y), r, (255,255,255), -1, 8, 0)
    out = img&mask
    return out

def crop_image(img):
    return background_subtract_circle(img, circles(img))

point = X_train[700]

#plt.imshow(circles(point, True), cmap='gray', vmin=0, vmax=255)
#plt.imshow(edges(point), cmap='gray', vmin=0, vmax=255)

X_train_cropped = [crop_image(img) for img in X_train]
X_test_cropped = [crop_image(img) for img in X_test]

plt.imshow(X_train_cropped[700], cmap='gray', vmin=0, vmax=255);
# https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
# https://keras.io/getting-started/functional-api-guide/
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Model

input_img = Input(shape=(32, 32, 3))

output_1 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(input_img)
output_1_pool = MaxPooling2D((2,2))(output_1)

output_2 = Conv2D(32, kernel_size=(3,3), activation='relu', padding='same')(output_1_pool)
output_pool = MaxPooling2D((3,3))(output_2)

img_output = Flatten()(output_pool)

digits_1 = Dense(64, activation='relu')(img_output)
digits_2 = Dropout(0.1)(digits_1)
digits_3 = Dense(64, activation='relu')(digits_2)

digits_output = Dense(43, activation='softmax')(digits_3)

model = Model(inputs=input_img, outputs=digits_output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()
# https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator


xtrain, xtest, ytrain, ytest = train_test_split(X_train_cropped, y_train, test_size=0.2)
#ytrain = to_categorical(ytrain)
#ytest = to_categorical(ytest)

xtrain = np.array(xtrain)
xtest = np.array(xtest)
ytrain = np.array(ytrain)
ytest = np.array(ytest)

xval = np.array(X_test_cropped)
xval = xval / 255

xtrain = xtrain/255
xtest = xtest/255

#xtrain = xtrain.reshape(xtrain.shape[0], 32, 32, 1)
#xtest = xtest.reshape(xtest.shape[0], 32, 32, 1)
#xval = xtest.reshape(xtest.shape[0], 32, 32, 1)
# model.fit_generator(
#     train_datagen.flow(xtrain, ytrain, batch_size=32), 
#     #steps_per_epoch=len(xtrain) / 32,
#     validation_data=test_datagen.flow(xtest, ytest, batch_size=32),
#     epochs=10)

model.fit(xtrain, ytrain,
          epochs=10,
          batch_size=16,
          validation_data=(xtest, ytest))
# model.save_weights('model.h5')
import pandas as pd

predictions = model.predict_generator(test_datagen.flow(xval))

df = pd.DataFrame(predictions.round(3))
df.columns = ['oh_0','oh_1','oh_2','oh_3','oh_4','oh_5','oh_6','oh_7','oh_8','oh_9','oh_10','oh_11','oh_12','oh_13','oh_14','oh_15','oh_16','oh_17','oh_18','oh_19','oh_20','oh_21','oh_22','oh_23','oh_24','oh_25','oh_26','oh_27','oh_28','oh_29','oh_30','oh_31','oh_32','oh_33','oh_34','oh_35','oh_36','oh_37','oh_38','oh_39','oh_40','oh_41','oh_42']
df.insert(0, 'id', df.index+1)
df.to_csv('predictions.csv', sep=',', index=False)
len(predictions)