import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split



from tensorflow.keras.models import Sequential, Model, load_model

from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

from tensorflow.keras import optimizers

from keras.utils.np_utils import to_categorical



import os
train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
print(train.shape)

train.head()
# The 'label' column is separated from the rest in the training set



label = train['label'].to_numpy()

train['label'].value_counts().plot(kind='bar')

train = train.drop(columns=['label']).to_numpy()

test = test.to_numpy()



# The reshaping is important as need to put it in form of an image of dimension 28x28.

# The extra 1 denotes it is an gray scale image. For an RGB image it would be 3

train=np.reshape(train,(42000,28,28,1))

test=np.reshape(test,(28000,28,28,1))
#Visualizing few random digits from the training set



fig=plt.figure(figsize=(10, 4))

columns = 6

rows = 1

for i in range(columns*rows):

    fig.add_subplot(rows, columns, i+1)

    plt.axis('off')

    img1 = train[i+1000]

    plt.imshow(np.squeeze(img1,axis=2), cmap="gray")
#Here I have used the second method, the first method is equally applicable:



meu = train.mean()

sig = train.std()

train = (train - meu)/sig



meu = test.mean()

sig = test.std()

test = (test - meu)/sig
print("Labels before one-hot encoding:")

print(label[5:8])

print("Labels after one-hot encoding:")

print(to_categorical(label)[5:8])



label = to_categorical(label)
# Spilitting the training set to get validation results



X_train, X_val, Y_train, Y_val = train_test_split(train, label, test_size=0.1, random_state=0)
model = Sequential()



model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(28,28,1)))

model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=2))

model.add(Flatten())

model.add(Dense(512,activation="relu"))

model.add(Dense(10,activation="softmax"))

model.compile(loss="categorical_crossentropy",optimizer="adam", metrics=["accuracy"])



model.summary()
model.fit(X_train,Y_train, epochs=10, validation_data=(X_val,Y_val))
result = model.evaluate(X_val, Y_val, batch_size = 32)

print("Loss on validation: %f \nAccuracy on validation: %f " %(result[0] ,result[1]*100))
ylabel = model.predict_classes(X_val)

ytest = [np.argmax(y, axis=None, out=None) for y in Y_val]

conf_matrix = confusion_matrix(ytest, ylabel)

sns.heatmap(conf_matrix, cmap="YlGnBu",annot=True, fmt='g')

plt.xlabel('predicted value')

plt.ylabel('actual value')
#Displaying the picture along with the predictions



fig=plt.figure(figsize=(15, 6))

columns = 6

rows = 3

for i in range(columns*rows):

    fig.add_subplot(rows, columns, i+1)

    plt.axis('off')

    img1 = X_val[i]

    value ='%1.0f'%(ylabel[i])

    plt.text(5, 25,value,color='black',fontsize=12,bbox=dict(facecolor='yellow'))

    plt.imshow(np.squeeze(img1,axis=2), cmap="gray")
# Let me print both the prediction and the actual values on 

# the images which will help in better visualisation



y_mis = [i for i in range(len(ytest)) if ytest[i]!=ylabel[i]]

print(y_mis) #indices of the mis-classifications



fig=plt.figure(figsize=(15, 6))

columns = 6

rows = 3

x=0

for i in range(columns*rows):

    fig.add_subplot(rows, columns, i+1)

    plt.axis('off')

    img1 = X_val[y_mis[x]]

    predicted_value ='%1.0f'%(ylabel[y_mis[x]])

    plt.text(5, 25,predicted_value,color='black',fontsize=12,bbox=dict(facecolor='yellow'))

    actual_value ='%1.0f'%(ytest[y_mis[x]])

    plt.text(20, 25,actual_value,color='white',fontsize=12,bbox=dict(facecolor='green'))

    plt.imshow(np.squeeze(img1,axis=2), cmap="gray")

    x=x+1
img=X_val[8]

fig=plt.figure(figsize=(2,2))

plt.imshow(np.squeeze(img,axis=2), cmap="gray")

plt.show()



img = np.expand_dims(img, axis=0)

conv1_output = Model(inputs=model.input, outputs=model.get_layer('conv2d').output)

conv2_output = Model(inputs=model.input, outputs=model.get_layer('conv2d_1').output)

conv1_features = conv1_output.predict(img)

conv2_features = conv2_output.predict(img)



import matplotlib.image as mpimg



fig=plt.figure(figsize=(14,7))

columns = 8

rows = 4



print("Output for each filter of first Convolution Layer")

for i in range(columns*rows):

    #img = mpimg.imread()

    fig.add_subplot(rows, columns, i+1)

    plt.axis('off')

    plt.title('filter'+str(i))

    plt.imshow(conv1_features[0, :, :, i], cmap='gray')

plt.show()



fig=plt.figure(figsize=(14,7))

print("Output for each filter of second Convolution Layer")

for i in range(columns*rows):

    #img = mpimg.imread()

    fig.add_subplot(rows, columns, i+1)

    plt.axis('off')

    plt.title('filter'+str(i))

    plt.imshow(conv2_features[0, :, :, i], cmap='gray')

plt.show()
answers = model.predict_classes(test, verbose=0)

submissions=pd.DataFrame({"ImageId": list(range(1,len(answers)+1)),"Label": answers})

submissions.to_csv("submission.csv", index=False, header=True)