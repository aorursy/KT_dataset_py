import numpy as np

import pandas as pd

import copy

import datetime



%matplotlib notebook

import matplotlib.pyplot as plt
## Load dataset



IMG_SHAPE = (96, 96, 3)

original_data = pd.read_csv(r"../input/updated_train.csv")
print("Dimensions of the dataset: ", original_data.shape)

print(original_data.loc[1,:])

column_names = list(original_data.columns)
# Check what's the percentage of missing values

original_data.isnull().sum()/original_data.shape[0] *100
# Some of the images have missing values. Here we drop those having more missing values than `threshold`

threshold = 30

data = original_data.dropna(thresh = threshold)



# Number of features that needs to be estimated (x and y)

features_to_estimate = data.shape[1]-1

print(features_to_estimate)
from sklearn.model_selection import train_test_split



Y = data.iloc[:, :-1]

X_raw = data.iloc[:,-1]

dim_output = len(Y.columns.values)

original_batch_size = len(X_raw)

print(X_raw.shape, Y.shape)
def convert_X(Xdf):

    """

    Function to convert each "Image" field in the csv file into a (96, 96, 1) image. 

    """

    batch_size = len(Xdf)

    X = np.zeros((batch_size, 96, 96, 3))



    for row in range(batch_size):

        for idx in range(3):

            X[row, :,:, idx] = np.array([float(i) for i in Xdf.iloc[row].split(" ")]).reshape(1,96,96)/255

    return X





X = convert_X(X_raw)

print(X.shape)
Y = np.array(Y)



plt.figure()

for i in range(25):

    idx = np.random.randint(0, high = X.shape[0])

    plt.subplot(5,5,1+i)

    plt.imshow(X[idx, :,:,0],cmap = "gray")

    plt.plot(Y[idx,::2],Y[idx,1::2], "*r")
X0 = copy.deepcopy(X)

Y0 = np.array(Y)



# Flip up/down

Y1 = copy.deepcopy(Y0)

Y1[:,1::2] = IMG_SHAPE[1] - Y0[:,1::2]

Xaugmented = np.append(X0, X0[:,::-1,:,:], axis = 0)

Yaugmented = np.append(Y0, Y1, axis = 0)



# Flip left/right

Y1 = copy.deepcopy(Y0)

Y1[:,::2] = IMG_SHAPE[0] - Y0[:,::2]

Xaugmented = np.append(Xaugmented, X0[:,:,::-1,:], axis = 0)

Yaugmented = np.append(Yaugmented, Y1, axis = 0)



# Flip all

Y1 = copy.deepcopy(Y0)

Y1[:,::2] = IMG_SHAPE[0] - Y0[:,::2]

Y1[:,1::2] = IMG_SHAPE[0] - Y0[:,1::2]

Xaugmented = np.append(Xaugmented, X0[:,::-1,::-1,:], axis = 0)

Yaugmented = np.append(Yaugmented, Y1, axis = 0)
plt.figure()

for i in range(25):

    idx = np.random.randint(0, high = Xaugmented.shape[0])

    plt.subplot(5,5,1+i)

    plt.imshow(Xaugmented[idx, :,:,0],cmap = "gray")

    plt.plot(Yaugmented[idx,::2],Yaugmented[idx,1::2], "*r")
# Split train, validation and test set



XXtrain, Xtest, YYtrain, Ytest = train_test_split(X, Y, test_size = 0.05)

Xtrain, Xval, Ytrain, Yval = train_test_split(XXtrain, YYtrain, test_size = 0.2)



print("Train set: ", Xtrain.shape, Ytrain.shape)

print("Validation set: ", Xval.shape, Yval.shape)

print("Test set: ", Xtest.shape, Ytest.shape)
import tensorflow as tf

keras = tf.keras

# Specify include_top=False not to include the last layer

base_model = keras.applications.MobileNetV2(input_shape = IMG_SHAPE,

                                           include_top = False,

                                           weights = "imagenet")

lam = 0.7

fine_tune_at = 100

for layer in base_model.layers[fine_tune_at:]:

    layer.trainable =  True

    layer.kernel_regularizer=tf.keras.regularizers.l2(lam)
base_model.summary()
global_average_pooling_layer = tf.keras.layers.GlobalAveragePooling2D()

intermediate_layer = tf.keras.layers.Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lam))

dense_layer = tf.keras.layers.Dense(features_to_estimate, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(lam))
model = tf.keras.Sequential([base_model,

                             global_average_pooling_layer,

                             intermediate_layer,

                             dense_layer])

learning_rate = 0.0002

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

              loss="mse",

              metrics= [tf.keras.metrics.RootMeanSquaredError()])

model.summary()
len(model.trainable_variables)
history = model.fit(x=np.array(Xtrain),

    y=np.array(Ytrain),    

    epochs=2000,

    batch_size=128,

    verbose=1,

    initial_epoch=0,

    validation_data=(np.array(Xval), np.array(Yval)),

    shuffle=True)
loss = history.history["loss"]

loss_val = history.history["val_loss"]

plt.figure()

plt.semilogy(loss, label = "Train")

plt.semilogy(loss_val, label = "Validation")

plt.xlabel("Epoch")

plt.ylabel("Loss (MSE)")

# plt.plot(rmse, label = "Train")

# plt.plot(rmse_val, label = "Validation")

plt.legend()





rmse = history.history["root_mean_squared_error"]

rmse_val = history.history["val_root_mean_squared_error"]

plt.figure()

plt.semilogy(rmse, label = "Train")

plt.semilogy(rmse_val, label = "Validation")

plt.xlabel("Epoch")

plt.ylabel("RMSE")

plt.hlines(2.6, 0, 2000, label = "Human baseline", zorder = 3)

plt.legend()

plt.show()
plt.figure()

XXX = np.array(Xtrain)

YYY = np.array(Ytrain)

prediction = model.predict(XXX)

for i in range(25):

    idx =np.random.randint(low = 0, high = Xtrain.shape[0]) 

    plt.subplot(5,5,i+1)

    plt.imshow(XXX[idx,:,:,0], cmap = "gray")

    plt.plot(YYY[idx,::2], YYY[idx, 1::2], "*r")

    plt.plot(prediction[idx,::2], prediction[idx,1::2], "ob")
plt.figure()

XXX = np.array(Xval)

YYY = np.array(Yval)

prediction = model.predict(XXX)

for i in range(25):

    idx =np.random.randint(low = 0, high = XXX.shape[0]) 

    plt.subplot(5,5,i+1)

    plt.imshow(XXX[idx,:,:,0], cmap = "gray")

    plt.plot(YYY[idx,::2], YYY[idx, 1::2], "*r")

    plt.plot(prediction[idx,::2], prediction[idx,1::2], "ob")
# saving the model

model.save("mymodel_5.h5")
test_performance = model.evaluate(x = Xtest,

                                  y = Ytest,

                                  verbose=0)

print("Performance on the test set: ", test_performance[1])

print("Human baseline: ", 2.6)
lut = pd.read_csv("IdLookupTable.csv")

test_database = pd.read_csv("test.csv")
Xsubmission = convert_X(test_database.iloc[:,-1])
Ysubmission = model.predict(Xsubmission)
print(Ysubmission)

print(Ysubmission.shape)
idx = 1

plt.figure()

plt.imshow(Xsubmission[idx,:,:,0], cmap = "gray")

plt.plot(Ysubmission[idx,::2], Ysubmission[idx, 1::2], "*r")
results = []

for i in range(len(lut)):

    image_idx = lut.ImageId[i]

    idx_feature_to_retrieve = column_names.index(lut.FeatureName[i])

    results.append(Ysubmission[image_idx-1, idx_feature_to_retrieve])

lut.Location = results
lut
lut.to_csv("my_submission.csv", index=False, columns=("RowId","Location"))