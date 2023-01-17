import numpy as np

import matplotlib.pyplot as plt   

from skimage.io import imshow

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
# Patition datasets into two (fancy indexing)

targets < 30 

train = data[targets < 30]

train.shape
test = data[targets >= 30]

test.shape
# Generate 8 random integers between 0 and 100

n_faces = test.shape[0]//12             # // is unconditionally "flooring division"

face_ids = np.random.randint(0 , 100, size = n_faces)

# Random 'n_faces' faces from within 1 to 100

sub_test = test[face_ids, :]

face_ids



# Total pixels in any image

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
ESTIMATORS = {

    "Extra trees": ExtraTreesRegressor(n_estimators=10,

                                       max_features=32,

                                       random_state=0),

    "K-nn": KNeighborsRegressor(),

    "Linear regression": LinearRegression(),

    "Ridge": RidgeCV(),

    "RandomForestRegressor": RandomForestRegressor()    

}
# Create an empty dictionary 

y_test_predict = dict()



# Fit each model by turn and make predictions

for name, estimator in ESTIMATORS.items():     

    estimator.fit(X_train, y_train)

    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces

image_shape = (64, 64)



n_cols = 1 + len(ESTIMATORS)

plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))

plt.suptitle("Face completion with multi-output estimators", size=16)



for i in range(n_faces):

    true_face = np.hstack((X_test[i], y_test[i]))



    if i:

        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)

    else:

        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,

                          title="true faces")



    sub.axis("off")

    sub.imshow(true_face.reshape(image_shape),

               cmap=plt.cm.gray,

               interpolation="nearest")



    for j, est in enumerate(sorted(ESTIMATORS)):

        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))



        if i:

            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)



        else:

            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,

                              title=est)



        sub.axis("off")

        sub.imshow(completed_face.reshape(image_shape),

                   cmap=plt.cm.gray,

                   interpolation="nearest")



plt.show()
image_shape = (64, 64)

# For 'Extra trees' regression

plt.figure(figsize=(15,15))

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


# For 'Ridge' regression

plt.figure(figsize=(15,15))

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
plt.figure(figsize=(15,15))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['Linear regression'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face

    imshow(y)

  

plt.show()
plt.figure(figsize=(10,10))

j = 0

for i in range(n_faces):

    actual_face =    sub_test[i].reshape(image_shape)

    completed_face = np.hstack((X_test[i], y_test_predict['K-nn'][i]))

    j = j+1

    plt.subplot(4,4,j)

    x = completed_face.reshape(image_shape)

    imshow(x)

    j = j+1

    plt.subplot(4,4,j)

    y = actual_face.reshape(image_shape)

    imshow(y)

  

plt.show()
# For 'RandomForestRegressor' regression

plt.figure(figsize=(10,10))

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