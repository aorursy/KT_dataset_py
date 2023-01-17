import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import urllib.request
import matplotlib.patches as patches

import keras
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Convolution2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

# Download Opencv face data xml
!wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

train_data = pd.read_csv('../input/training.csv')
train_data.isnull().any().value_counts()
# THIS IS VERY IMPORTANT 
# THIS IS VERY IMPORTANT 
# THIS IS VERY IMPORTANT 
train_data.fillna(method = 'ffill',inplace = True)
# THIS IS VERY IMPORTANT 
# THIS IS VERY IMPORTANT 
# THIS IS VERY IMPORTANT 
x_train = train_data['Image'].apply(lambda x: pd.Series(list(map(int, x.split(' ')))))
y_train = train_data.drop('Image', axis=1)
x_train = x_train.values.reshape(-1,96,96).astype(float)
y_train = y_train.values

def print_image_with_kf(img_array, key_points):
    # merge the key features on to image

    for i in range(0, len(key_points), 2):
        x, y = key_points[i], key_points[i+1]
        # Set black color at the place where there is a key point
        img_array[int(y)][int(x)] = 0
        
    plt.imshow(img_array, cmap='gray')
    plt.show()
        
print_image_with_kf(x_train[96], y_train[96])
salen_url = "https://media.glamour.com/photos/5ba8ea573f965a344b7bcc18/master/w_644,c_limit/selena.jpg"

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

salena = url_to_image(salen_url)
# Not using lena reference image, instead using salena :D
# Lets get the bouding box for salena image
def get_bb(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    return face_cascade.detectMultiScale(grayscale_image, 1.25, 6)

boundin_box = get_bb(salena)
# get the positions, width and height of the face detected by the opencv
x, y, w, h = list(boundin_box[0])
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(cv2.cvtColor(salena, cv2.COLOR_BGR2RGB))

# Create a Rectangle patch
# (x, y), w, h
# x, y, w, h = list(boundin_box[0])
rect = patches.Rectangle((x, y), w, h,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

plt.show()
# So now we are able to draw bouding box on top of the face
model = Sequential([Flatten(input_shape=(96,96)),
                         Dense(128, activation="relu"),
                         Dropout(0.1),
                         Dense(64, activation="relu"),
                         Dense(30)
                         ])

model.compile(optimizer='adam', 
              loss='mse',
              metrics=['mae','accuracy'])

model.fit(x_train, y_train, epochs=270, batch_size=128,validation_split = 0.2)
test_data = pd.read_csv('../input/test.csv')
test_data.fillna(method = 'ffill',inplace = True)

x_test = test_data['Image'].apply(lambda x: pd.Series(list(map(int, x.split(' ')))))
x_test = x_test.values.reshape(-1,96,96)
features = model.predict(x_test, batch_size=1)
print_image_with_kf(x_test[200], features[200])
import math

# Convert feature maps to key mapping
feature_to_index_map = {k:i for k, i in zip(train_data.keys(), range(len(train_data.keys())))}

# Actually left is right and right is left in our perspective
# i guess its the value wrt the person, and image is mirror?

# so all left feature values are for right

# Left outer,  left inner
right_eyebrow = [("left_eyebrow_inner_end_x", "left_eyebrow_inner_end_y"), ("left_eyebrow_outer_end_x", "left_eyebrow_outer_end_y")]

# right inner, right outer
left_eyebrow = [("right_eyebrow_outer_end_x", "right_eyebrow_outer_end_y"), ("right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y")]

mouth_horizontal = [("mouth_right_corner_x", "mouth_right_corner_y") , ("mouth_left_corner_x", "mouth_left_corner_y")]


def get_features(features, keys, flat=False):
    d = []
    for k in keys:
        x = features[feature_to_index_map[k[0]]]
        y = features[feature_to_index_map[k[1]]]        
        if flat:
            d.append(x)
            d.append(y)
        else:
            d.append((x, y))
    return d

def euclidean_distance(plot1, plot2):
    return math.sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

def resize(image, window_height = 500):
    aspect_ratio = float(image.shape[1])/float(image.shape[0])
    window_width = window_height/aspect_ratio
    image = cv2.resize(image, (int(window_height),int(window_width)))
    return image
# features[200][0]
import copy

trail_img = x_test[208]
trail_f = features[208]

# print_image_with_kf(copy.deepcopy(trail_img), [80.365685, 28.525555, 54.415924, 29.468622])
print_image_with_kf(copy.deepcopy(trail_img), [54.415924, 29.468622, 80.365685, 28.525555])


rigth_ear_url = "https://i.imgur.com/XbBtt6F.png"
left_ear_url = "https://i.imgur.com/YKauHHj.png"
mouth_url = "https://i.imgur.com/NgWyFsx.png"

rigth_ear_snap = url_to_image(rigth_ear_url)
rigth_ear_gray_snap = cv2.cvtColor(rigth_ear_snap, cv2.COLOR_BGR2GRAY)

left_ear_snap = url_to_image(left_ear_url)
left_ear_gray_snap = cv2.cvtColor(left_ear_snap, cv2.COLOR_BGR2GRAY)

mouth_snap = url_to_image(mouth_url)
mouth_gray_snap = cv2.cvtColor(mouth_snap, cv2.COLOR_BGR2GRAY)

right_eyebrow_features = get_features(trail_f, right_eyebrow)
left_eyebrow_features = get_features(trail_f, left_eyebrow)
mouth_features = get_features(trail_f, mouth_horizontal)

rigth_ear_gray_snap_resized = resize(rigth_ear_gray_snap, euclidean_distance(*right_eyebrow_features))
left_ear_gray_snap_resized = resize(left_ear_gray_snap, euclidean_distance(*left_eyebrow_features))
mouth_gray_snap_resized = resize(mouth_gray_snap, euclidean_distance(*mouth_features))

l_img = copy.deepcopy(trail_img)

def merge_images(l_img, s_img, x_offset, y_offset):
#     l_img[y_offset : y_offset + s_img.shape[0], x_offset : x_offset + s_img.shape[1]] = s_img
    shape = s_img.shape
    s_img = list(s_img)
    for y in range(shape[0]):
        for x in range(shape[1]):
            l_img[max([y + y_offset - shape[0], 0])][x + x_offset] = s_img[y][x]
    return l_img

l_img = merge_images(l_img, rigth_ear_gray_snap_resized, int(right_eyebrow_features[0][0]), int(right_eyebrow_features[0][1]))
l_img = merge_images(l_img, left_ear_gray_snap_resized, int(left_eyebrow_features[0][0]), int(left_eyebrow_features[0][1]))
# l_img = merge_images(l_img, mouth_gray_snap_resized, int(mouth_features[0][0]), int(mouth_features[0][1]))

plt.imshow(l_img, cmap='gray')
plt.show()
salena_gray = url_to_image(salen_url)
boundin_box = get_bb(salena_gray)
salena_gray = cv2.cvtColor(salena_gray, cv2.COLOR_BGR2GRAY)

salena_bb_x, salena_bb_y, salena_bb_w, salena_bb_h = list(boundin_box[0])

salena_gray = merge_images(salena_gray, rigth_ear_gray_snap_resized, int(right_eyebrow_features[0][0]) + salena_bb_x, int(right_eyebrow_features[0][1]) + salena_bb_y)
salena_gray = merge_images(salena_gray, left_ear_gray_snap_resized, int(left_eyebrow_features[0][0]) + + salena_bb_x, int(left_eyebrow_features[0][1]) + + salena_bb_y)

plt.imshow(salena_gray, cmap='gray')
plt.show()