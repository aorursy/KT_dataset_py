import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from skimage.io import imshow

# Regresseros
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Input data files consisting of the images
img_ds = np.load("../input/olivetti_faces.npy")
img_ds.shape
# Sample images of a subject
img_cnt = 10
plt.figure(figsize=(15,15))
for i in range(img_cnt):
    plt.subplot(1,10,i+1)
    x=img_ds[i+30] # 3rd subject
    imshow(x)
plt.show()
img_ds_flt = img_ds.reshape(img_ds.shape[0],img_ds.shape[1]*img_ds.shape[2])
print(img_ds_flt.shape)
targets = np.load("../input/olivetti_faces_target.npy")
print(targets)
# We can see below first 10 index belongs to one image and so on
training_img_data = img_ds_flt[targets<30] # First 30 types of images out of 40 ie 30 * 10 =300
test_img_data = img_ds_flt[targets>=30] # Test on rest 10 independent people from number 30th to 39th  10 * 10 = 100
# Test on a subset of people
#     Generate 10 random integers between 0 and 100
# // is unconditionally "flooring division",
n_faces = test_img_data.shape[0]//10
n_faces
face_ids = np.random.randint(0 , 100, size =n_faces) # To select some random images from 100 images
face_ids
test_img_data.shape
# We will select the random 10 images from test data
test_img_data = test_img_data[face_ids, :] 
test_img_data
#Total pixels in any image
n_pixels = img_ds_flt.shape[1]
n_pixels
#Select upper half of the faces as predictors
X_train = training_img_data[:, :(n_pixels + 1) // 2]
X_train
#Lower half of the faces will be target(s)                 
y_train = training_img_data[:, n_pixels // 2:]
y_train
# Similarly for test data. Upper and lower half
X_test = test_img_data[:, :(n_pixels + 1) // 2]
y_test = test_img_data[:, n_pixels // 2:]
# Prepare a dictionary of estimators after instantiating each one of them
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32,random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "multi_gbm" : MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
}
# Create an empty dictionary to collect prediction values
y_test_predict = dict()
# Fit each model by turn and make predictions
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)
#y_test_predict['RandomForestRegressor'].shape
#Just check    
y_test_predict['Extra trees'].shape
## Processing output -> Each face should have this dimension
image_shape = (64, 64)
plt.figure(figsize=(15,15))
j = 0
for i in range(n_faces):
    actual_face = test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))# Horizental stack upper actual half and lower predict half
    j = j+1# Image index
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()
plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()
plt.figure(figsize=(15,15))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face
    imshow(y)
  
plt.show()
plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()
plt.figure(figsize=(10,10))
j = 0
for i in range(n_faces):
    actual_face =    test_img_data[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['multi_gbm'][i]))
    j = j+1
    plt.subplot(5,5,j)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(5,5,j)
    y = actual_face.reshape(image_shape)
    imshow(y)
  
plt.show()