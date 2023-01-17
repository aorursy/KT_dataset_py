# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
# load the dataset
db_face_images = np.load('../input/face-images-with-marked-landmark-points/face_images.npz')['face_images']
print(db_face_images.shape)
df_facial_keypoints = pd.read_csv('../input/face-images-with-marked-landmark-points/facial_keypoints.csv')
pd.set_option('display.max_columns', None)
#visualising the dataframe
df_facial_keypoints.head()
#Checking for the number of NAN values row wise
nan_value = df_facial_keypoints.isnull().sum(axis = 1)
print (nan_value)
#Getting the indices with 15 keypoints of all non null rows
indices = np.nonzero(nan_value == 0)[0] 
print(indices)
fig_row = 6
fig_col = 5

num_plot = fig_row * fig_col
random_indices_vector = np.random.choice(db_face_images.shape[2], num_plot, replace = False)
random_indices_mat = random_indices_vector.reshape(fig_row, fig_col)
plt.close('all')

fig, ax = plt.subplots(nrows = fig_row, ncols =fig_col, figsize = (14, 18))

for i in range(fig_row):
    for j in range(fig_col):
        curr_indice = random_indices_mat[i][j]
        curr_img = db_face_images[:,:,curr_indice]
        
        x_feature_cord = np.array(df_facial_keypoints.iloc[curr_indice, 0::2].tolist())
        y_feature_cord = np.array(df_facial_keypoints.iloc[curr_indice, 1::2].tolist())
        
        ax[i][j].imshow(curr_img, cmap = 'gray')
        ax[i][j].scatter(x_feature_cord,y_feature_cord,c='b',s=15)
        ax[i][j].set_axis_off()
        ax[i][j].set_title('image_index = %d' %(curr_indice),fontsize=10)
        
#Getting the modifed image database with 15 keypoints 
db_face_images = db_face_images[:,:,indices]
db_face_images.shape
#Reseting the index of keypoits as per indices
df_facial_keypoints = df_facial_keypoints.iloc[indices,:].reset_index(drop=True)
df_facial_keypoints.shape
df_facial_keypoints.head()
#Converting both dataset and dataframe into array for further modification

db_face_images = np.moveaxis(db_face_images, -1, 0)
db_face_images.shape

# Images are gray scale
db_images = np.asarray(db_face_images).reshape(db_face_images.shape[0],96,96,1)
print(db_images.shape)

df_keypoints = np.array(df_facial_keypoints)
print(df_keypoints.shape)

#Let's create a function to plot the image
def plot_sample(image, keypoint, axis, title):
    image = image.reshape(96,96)
    axis.imshow(image, cmap='gray')
    axis.scatter(keypoint[0::2], keypoint[1::2], marker='*', s=20)
    plt.title(title)
#Let's create a new label for images and keypoints
db_images_modify = db_images
df_keypoints_modify = df_keypoints
fig, axis = plt.subplots()
plot_sample(db_images[50], df_keypoints[50], axis, "Sample Image & Keypoints")
#Various Image Agumentation choices
sample = 50
horizontal_flip = True
rotation_augmentation = True
brightness_augmentation = True
shift_augmentation = True
random_noise_augmentation = True
#Function for flipping of images horizontally
def flip(images, keypoints):
    flipped_keypoints = []
    flipped_images = np.flip(images, axis=2)   # Flip column-wise (axis=2)
    for idx, sample_keypoints in enumerate(keypoints):
        flipped_keypoints.append([96.-coor if idx%2==0 else coor for idx,coor in enumerate(sample_keypoints)])    # Subtract only X co-ordinates of keypoints from 96 for horizontal flipping
    return flipped_images, flipped_keypoints

if horizontal_flip:
    db_images_flipped, df_keypoints_flipped = flip(db_images, df_keypoints)
    print("Shape of flipped_images:",np.shape(db_images_flipped))
    print("Shape of flipped_keypoints:",np.shape(df_keypoints_flipped))
    
    #Adding flipped images and keypoints to my modified dataset and dataframe
    db_images_modify = np.concatenate((db_images_modify, db_images_flipped))
    df_keypoints_modify = np.concatenate((df_keypoints_modify, df_keypoints_flipped))
    fig, axis = plt.subplots()
    plot_sample(db_images_flipped[sample], df_keypoints_flipped[sample], axis, "Horizontally Flipped")
    
    
print("Shape of images database after shifting:",db_images_modify.shape)
print("Shape of keypoints dataframe after shifting:",df_keypoints_modify.shape)
    
pixel_shift = [12]    # shift amount in pixels (includes shift from all 4 corners)

#Function fot translation
def shift(images, keypoints):
    shifted_images = []
    shifted_keypoints = []
    for shift in pixel_shift:    # Augmenting over several pixel shift values
        for (shift_x,shift_y) in [(-shift,-shift),(-shift,shift),(shift,-shift),(shift,shift)]:
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            for image, keypoint in zip(images, keypoints):
                shifted_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                shifted_keypoint = np.array([(point+shift_x) if idx%2==0 else (point+shift_y) for idx, point in enumerate(keypoint)])
                if np.all(0.0<shifted_keypoint) and np.all(shifted_keypoint<96.0):
                    shifted_images.append(shifted_image.reshape(96,96,1))
                    shifted_keypoints.append(shifted_keypoint)
    shifted_keypoints = np.clip(shifted_keypoints,0.0,96.0)
    return shifted_images, shifted_keypoints

if shift_augmentation:
    db_images_shifted, df_keypoints_shifted = shift(db_images, df_keypoints)
    print(f"Shape of shifted_images:",np.shape(db_images_shifted))
    print(f"Shape of shifted_keypoints:",np.shape(df_keypoints_shifted))
    
    db_images_modify = np.concatenate((db_images_modify, db_images_shifted))
    df_keypoints_modify = np.concatenate((df_keypoints_modify, df_keypoints_shifted))
    fig, axis = plt.subplots()
    plot_sample(db_images_shifted[sample], df_keypoints_shifted[sample], axis, "Shift Augmentation")
    
print("Shape of images database after shifting:",np.shape(db_images_modify))
print("Shape of keypoints dataframe after shifting:",np.shape(df_keypoints_modify))
from math import sin, cos, pi


rotation_angles = [12]    # Rotation angle in degrees (includes both clockwise & anti-clockwise rotations)

#Function for Rotation of the Images
def rotate(images, keypoints):
    rotated_images = []
    rotated_keypoints = []
    print("Augmenting for angles (in degrees): ")
    
    for angle in rotation_angles:    # Rotation augmentation for a list of angle values
        for angle in [angle,-angle]:
            print(f'{angle}', end='  ')
            M = cv2.getRotationMatrix2D((48,48), angle, 1.0)
            angle_rad = -angle*pi/180.     # Obtain angle in radians from angle in degrees (notice negative sign for change in clockwise vs anti-clockwise directions from conventional rotation to cv2's image rotation)
            
            # For train_images
            for image in images:
                rotated_image = cv2.warpAffine(image, M, (96,96), flags=cv2.INTER_CUBIC)
                rotated_images.append(rotated_image)
            
            # For train_keypoints
            for keypoint in keypoints:
                rotated_keypoint = keypoint - 48.    # Subtract the middle value of the image dimension
                for idx in range(0,len(rotated_keypoint),2):
                    # https://in.mathworks.com/matlabcentral/answers/93554-how-can-i-rotate-a-set-of-points-in-a-plane-by-a-certain-angle-about-an-arbitrary-point
                    rotated_keypoint[idx] = rotated_keypoint[idx]*cos(angle_rad)-rotated_keypoint[idx+1]*sin(angle_rad)
                    rotated_keypoint[idx+1] = rotated_keypoint[idx]*sin(angle_rad)+rotated_keypoint[idx+1]*cos(angle_rad)
                rotated_keypoint += 48.   # Add the earlier subtracted value
                rotated_keypoints.append(rotated_keypoint)
            
    return np.reshape(rotated_images,(-1,96,96,1)), rotated_keypoints

#For more details on the transformation of the images below is the link.
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

if rotation_augmentation:
    db_images_rotated, df_keypoints_rotated = rotate(db_images, df_keypoints)
    print("\nShape of rotated_images:",np.shape(db_images_rotated))
    print("Shape of rotated_keypoints:\n",np.shape(df_keypoints_rotated))
    
    #Concatenating the train images with rotated image & train keypoints with rotated train points
    db_images_modify = np.concatenate((db_images_modify, db_images_rotated))
    df_keypoints_modify = np.concatenate((df_keypoints_modify, df_keypoints_rotated))
    fig, axis = plt.subplots()
    plot_sample(db_images_rotated[sample], df_keypoints_rotated[sample], axis, "Rotation Augmentation")
    
print("Shape of images database after shifting:",np.shape(db_images_modify))
print("Shape of keypoints dataframe after shifting:",np.shape(df_keypoints_modify))
#Writing a function to add noise
def add_noise(images):
    noisy_images = []
    for image in images:
        noisy_image = cv2.add(image, 0.009*np.random.randn(96,96,1))    # Adding random normal noise to the input image & clip the resulting noisy image between [-1,1]
        noisy_images.append(noisy_image.reshape(96,96,1))
    return noisy_images

if random_noise_augmentation:
    db_images_noisy = add_noise(db_images)
    print("Shape of noisy_train_images:",np.shape(db_images_noisy))
    
    db_images_modify = np.concatenate((db_images_modify, db_images_noisy))
    df_keypoints_modify = np.concatenate((df_keypoints_modify, df_keypoints))
    fig, axis = plt.subplots()
    plot_sample(db_images_noisy[sample], df_keypoints[sample], axis, "Random Noise Augmentation")
    
print("Shape of images database after shifting:",np.shape(db_images_modify))
print("Shape of keypoints dataframe after shifting:",np.shape(df_keypoints_modify))
print("Shape of final train_images: {}".format(np.shape(db_images_modify)))
print("Shape of final train_keypoints: {}".format(np.shape(df_keypoints_modify)))

if horizontal_flip:
    print("Horizontal Flip Augmentation: ")
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(db_images_flipped[i], df_keypoints_flipped[i], axis, "")
    plt.show()

if shift_augmentation:
    print("Shift Augmentation: ")
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(db_images_shifted[i], df_keypoints_shifted[i], axis, "")
    plt.show()
    
if rotation_augmentation:
    print("Rotation Augmentation: ")
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(db_images_rotated[i], df_keypoints_rotated[i], axis, "")
    plt.show()
    
if random_noise_augmentation:
    print("Random Noise Augmentation: ")
    fig = plt.figure(figsize=(20,8))
    for i in range(10):
        axis = fig.add_subplot(2, 5, i+1, xticks=[], yticks=[])
        plot_sample(db_images_noisy[i], df_keypoints[i], axis, "")
    plt.show()
X = np.array(db_images_modify)
print(X.shape)
y = np.array(df_keypoints_modify)
print(y.shape)
#checking the type of both X and y dataset
type(X), type(y)
from sklearn.model_selection import train_test_split

X_train, X_test,y_train, y_test = train_test_split(X, y, test_size=0.20)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# convert from int to float
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

#Normalize the input image
X_train = X_train / 255
X_test = X_test / 255

from keras.models import Sequential
from keras.layers.advanced_activations import ReLU
from keras.layers import Dense, Conv2D, Flatten, AvgPool2D, BatchNormalization, Dropout, Activation, MaxPooling2D
from keras.models import Model

model = Sequential()

model.add(Conv2D(32, (3, 3),use_bias=False, input_shape = (96, 96, 1)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64,(3,3), use_bias=False))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
    
model.add(Conv2D(128,(3,3), use_bias=False))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3), use_bias=False))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dropout(0.1))  
model.add(Dense(30))

model.summary()
from keras import optimizers

opt = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss="mean_squared_error",optimizer= opt, metrics = ['accuracy'])
model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 128)
import os
model.save('model_facial_landmark.hdf5', overwrite = True)
#Evaluating and predicting the test dataset
predicted = model.evaluate(x = X_test, y = y_test)

print("Loss = " + str(predicted[0]))
print("Test Accuracy = " + str(predicted[1]))

y_test_pred = model.predict(X_test)
#Showing test images
from keras.preprocessing.image import img_to_array, array_to_img
plt.imshow(array_to_img(X_test[0]))
y_test_pred[0][0]
fig = plt.figure(figsize=(20,18))
for i in range(20):
    axis = fig.add_subplot(4, 5, i+1, xticks=[], yticks=[])
    plot_sample(X_test[i], y_test_pred[i], axis, "")
plt.show()
