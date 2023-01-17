import numpy as np 
import pandas as pd 
import keras
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.models import Sequential
import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/sign-language-mnist/sign_mnist_train.csv")
df_test = pd.read_csv("../input/sign-language-mnist/sign_mnist_test.csv")
print("Training data sample view:")
print(df_train.head())

print("Test data sample view:")
print(df_test.head())
print("Describing Training data:")
print(df_train.describe())

print("Describing Test data:")
print(df_test.describe())
print("Training data info:")
df_train.info()

print("Test data info:")
df_test.info()
'''Converting train/test data-frame to numpy array'''

train_x = df_train[df_train.columns[1::]].to_numpy()           
train_y = df_train[df_train.columns[0]].to_numpy()             #training label set [0 - 25]

test_x = df_test[df_test.columns[1::]].to_numpy() 
test_y = df_test[df_test.columns[0]].to_numpy()                #test label set [0 - 25]

print("SUMMARY OF DATA:")

print("train_x shape: " + str(train_x.shape))
print("train_y shape: " + str(train_y.shape))
print("test_x shape: " + str(test_x.shape))
print("test_y shape: " + str(test_y.shape))
'''Normalizing the train/tets data'''

train_x = train_x/255
test_x = test_x/255
index = 121
plt.imshow(train_x[index].reshape(28, 28))
plt.show()
print(train_y[index])
'''Making sequential deep learning model using keras
   input layer: shape -- (784, number of examples)
   layer 1: shape -- (128, number of examples) with "relu" activation
   layer 2: shape -- (64, number of examples) with "relu" activation
   layer 3: shape -- (64, number of examples) with "relu" activation
   layer 4: shape -- (32, number of examples) with "relu" activation
   layer 5: shape -- (26, number of examples) with "softmax" activation. This layer is the ouput layer'''

model = Sequential()

model.add(Dense(input_shape = (784, ), units = 128, activation = "relu"))
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = 32, activation = "relu"))
model.add(Dense(units = 26, activation = "softmax"))

'''using "Adam" optimizer'''

opt = Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999)

'''Compiling the model using the "sparse_categorical_crossentropy" loss function'''

model.compile(optimizer = opt, loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
'''Fitting the model using training set'''

sign_language_model = model.fit(train_x, train_y, epochs = 50, validation_split = 0.1)
plt.figure(figsize=(15, 5))

plt.subplot(1,2,1)

plt.plot(sign_language_model.history["accuracy"], label = "training set")
plt.plot(sign_language_model.history["val_accuracy"], label = "validation set")
plt.title("accuracy versus epochs curve")
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(loc='best')

plt.subplot(1,2,2)

plt.plot(sign_language_model.history["loss"], label = "training set")
plt.plot(sign_language_model.history["val_loss"], label = "validation set")
plt.title("loss versus epochs curve")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc='best')

plt.show()
model_acc = (model.evaluate(test_x, test_y))[1]*100

print("Test set accuracy is: " + str(model_acc) + " %")