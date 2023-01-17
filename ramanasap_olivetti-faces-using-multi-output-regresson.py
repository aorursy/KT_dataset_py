# For Data manipulation

import numpy as np



#For Plotting

import matplotlib.pyplot as plt   

from skimage.io import imshow



#Regressors

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import RidgeCV

from sklearn.ensemble import RandomForestRegressor
images = np.load("../input/olivetti_faces.npy")

images.shape
targets = np.load("../input/olivetti_faces_target.npy")

targets.shape   
firstImage = images[0]

imshow(firstImage) 
# Flatten individual image

data = images.reshape(images.shape[0], images.shape[1] * images.shape[2])



# Flattened 64 X 64 array

data.shape

'''There are ten different images of each of 40 distinct subjects.

That is each subject has 10 images but taken differently

For some subjects, the images were taken at different times, varying the lighting, facial expressions

(open / closed eyes,smiling / not smiling) and facial details (glasses / no glasses).

All the images were taken against a dark homogeneous background

with the subjects in an upright, frontal position (with tolerance  for some side     movement) '''



def printSubject(subject):

    plt.figure(figsize=(20,20))

    for i in range (10):

        plt.subplot(1, 10, i+1)

        x = images[i+subject]

        imshow(x)

    plt.show()



#plot first and last subject

subject1 = 0

subject40 = 390



printSubject(subject1)

printSubject(subject40)
# Patition datasets into two (fancy indexing)

targets < 30 

# First 30 types of images out of 40 ie 30 * 10 =300

train = data[targets < 30]

train.shape
# Test on rest independent people  10 * 10 = 100

test = data[targets >= 30]

test.shape
# Test on a subset of people

#     Generate 10 random integers between 0 and 100

# // is unconditionally "flooring division",

n_faces = test.shape[0]//12

n_faces
face_ids = np.random.randint(0 , 100, size =n_faces)

face_ids
test = test[face_ids, :]   

test.shape
#Total pixels in any image

n_pixels = data.shape[1]

n_pixels
#Select upper half of the faces as predictors

X_train = train[:, :(n_pixels + 1) // 2]

X_train
#Lower half of the faces will be target(s)                 

y_train = train[:, n_pixels // 2:]

y_train
# Similarly for test data. Upper and lower half

X_test = test[:, :(n_pixels + 1) // 2]

y_test = test[:, n_pixels // 2:]



# Fit multi-output estimators

ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "RandomForestRegressor": RandomForestRegressor()    

}



# Create an empty dictionary to collect prediction values

y_test_predict = dict()



# Fit each model by turn and make predictions

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)

    y_test_predict[name] = estimator.predict(X_test)

y_test_predict['RandomForestRegressor'].shape
image_shape = (64, 64)
# For 'Ridge' regression

plt.figure(figsize=(  n_faces * 1, n_faces))

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
# For 'Extra trees' regression

plt.figure(figsize=(  n_faces * 1, n_faces))

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
# For 'Linear regression' regression

plt.figure(figsize=(  n_faces * 1, n_faces))

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
plt.figure(figsize=(  n_faces * 1, n_faces))

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
# For 'RandomForestRegressor' regression

plt.figure(figsize=(  n_faces * 1, n_faces))

j = 0

for i in range(5):

    actual_face =    test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['RandomForestRegressor'][i]))

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