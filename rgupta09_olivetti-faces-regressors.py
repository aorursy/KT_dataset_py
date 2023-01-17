# Call the required libraries
# For data manipulation
import numpy as np

# For plotting faces
import matplotlib.pyplot as plt   
from skimage.io import imshow

# For the dataset
from sklearn.datasets import fetch_olivetti_faces

# For the required Regressors
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
# Load the faces datasets
data = np.load("../input/olivetti_faces.npy")
data.shape
# Extract data component for target
targets = np.load("../input/olivetti_faces_target.npy")
targets.size
# See an image
firstImage = data[0]
imshow(firstImage)
# Flatten each image
data = data.reshape(data.shape[0], data.shape[1] * data.shape[2])     
# Flattened 64 X 64 array # 64 X 64 = 4096
data.shape                               
# Patition datasets into two (fancy indexing)
targets < 30                # Output is true/false
train = data[targets < 30]  # First 30 types of images out of 40 ie 30 * 10 =300
test = data[targets >= 30]  # Test on rest independent people  10 * 10 = 100
# Test on a subset of people
#     Generate 10 random integers between 0 and 100
n_faces = test.shape[0]//10             # // is unconditionally "flooring division",
n_faces
face_ids = np.random.randint(0 , 100, size =n_faces)
face_ids
# There are now n_faces random-faces from within 1 to 100
test = test[face_ids, :]   
test
# Total pixels in any image
n_pixels = data.shape[1]
n_pixels
# Select upper half of the faces as predictors
X_train = train[:, :(n_pixels + 1) // 2]    # // is unconditionally "flooring division",
X_train
# Lower half of the faces will be target(s)                 
y_train = train[:, n_pixels // 2:]
y_train
# Similarly for test data. Upper and lower half
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
# Fit multi-output estimators

# Prepare a dictionary of estimators after instantiating each one of them
ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10,
                                       max_features=32,     # Out of 20000
                                       random_state=0),
    "K-nn": KNeighborsRegressor(),                          # Accept default parameters
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
    "RandomForest": RandomForestRegressor(),
}

# Create an empty dictionary to collect prediction values
y_test_predict = dict()
# Fit each model by turn and make predictions
#     Iterate over dict items
for name, estimator in ESTIMATORS.items():     
    estimator.fit(X_train, y_train)                    
    y_test_predict[name] = estimator.predict(X_test)
# Verify    
y_test_predict['Ridge'].shape       
y_test_predict['RandomForest']. shape
## Processing output
# Each face should have this dimension
image_shape = (64, 64)
# Plotting
## For 'Ridge' regression

plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Ridge'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
    

## For 'Extra trees' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Extra trees'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
## For 'Linear regression' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(n_faces):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
## For '"K-nn' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
## For '"Random Forest' regression
plt.figure(figsize=( 2 * n_faces * 2, 5))
j = 0
for i in range(5):
    actual_face =    test[i].reshape(image_shape)
    completed_face = np.hstack((X_test[i], y_test_predict['RandomForest'][i]))
    j = j+1
    plt.subplot(4,5,j)
    y = actual_face.reshape(image_shape)
    x = completed_face.reshape(image_shape)
    imshow(x)
    j = j+1
    plt.subplot(4,5,j)
    x = completed_face.reshape(image_shape)
    imshow(y)
  
plt.show()
