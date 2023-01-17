import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.models import Sequential

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D,BatchNormalization
train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.shape, test.shape
train.head()
test.head()
Y_train=train['label']



# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1)



Y_train.value_counts()
g = sns.countplot(Y_train)

plt.title('The distribution of the digits in the dataset', weight='bold', fontsize='18')
# Check the data

X_train.isnull().any().describe()
# Check the data

test.isnull().any().describe()
# Normalize the data

X_train = X_train / 255

test = test / 255
# Reshape image in 3 dimensions (height = 28px, width = 28px , channel = 1)

X_train = X_train.values.reshape(-1,28,28,1)

test = test.values.reshape(-1,28,28,1)
X_train[0].shape
plt.figure(figsize=(15,8))

for i in range(50):

    plt.subplot(5,10,i+1)

    plt.imshow(X_train[i].reshape((28,28)),cmap='binary')

    plt.axis("off")

plt.show()
print("The shape of the labels before One Hot Encoding",Y_train.shape)

Y_train = to_categorical(Y_train, num_classes = 10)

print("The shape of the labels after One Hot Encoding",Y_train.shape)
Y_train[0]
# Split the train and the validation set for the fitting

random_seed = 2

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.3, random_state=random_seed)
import matplotlib.pyplot as plt

# Some examples

g = plt.imshow(X_train[0][:,:,0])
datagen = ImageDataGenerator(zoom_range = 0.1, width_shift_range = 0.1, height_shift_range = 0.1, rotation_range = 10) 
model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 64, kernel_size = (5, 5), activation = 'relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides = (2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(512, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(1024, activation = 'relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer='adam',metrics=['accuracy'],loss='categorical_crossentropy')
reduction_lr = ReduceLROnPlateau(monitor='val_accuracy',patience=2, verbose=1, factor=0.2, min_lr=0.00001)
hist = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=32),epochs=20,validation_data = (X_val,Y_val),callbacks=[reduction_lr])
loss = pd.DataFrame(model.history.history)

loss[['loss', 'val_loss']].plot()

loss[['accuracy', 'val_accuracy']].plot()
final_loss, final_acc = model.evaluate(X_val, Y_val, verbose=0)

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(final_loss, final_acc))
y_pred = model.predict(X_val, batch_size = 64)



y_pred = np.argmax(y_pred,axis = 1)

y_pred = pd.Series(y_pred,name="Label")

y_pred
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(10,10))

ax1 = plt.subplot2grid((1,2),(0,0))

train_loss = hist.history['loss']

test_loss = hist.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title(' Loss vs. Epoch',weight='bold', fontsize=18)



ax1 = plt.subplot2grid((1,2),(0,1))

train_loss = hist.history['loss']

test_loss = hist.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title(' Accuracy vs. Epoch',weight='bold', fontsize=18)
Y_val.shape, y_pred.shape
Y_val = np.argmax(Y_val,axis = 1)

Y_val = pd.Series(Y_val,name="Label")
from sklearn.metrics import confusion_matrix

cmatrix = confusion_matrix(Y_val, y_pred)



plt.figure(figsize=(15,8))

plt.title('Confusion matrix of the test/predicted digits ', weight='bold', fontsize=18)

sns.heatmap(cmatrix,annot=True,cmap="Reds",fmt="d",cbar=False)
# #We use np.argmax with y_test and predicted values: transform them from 10D vector to 1D

# # class_y = np.argmax(Y_val,axis = 1) 

# # class_num=np.argmax(y_pred, axis=1)

# #Detect the errors

# errors = (y_pred - Y_val != 0)

# #Localize the error images

# predicted_er = y_pred[errors]

# y_test_er = Y_val[errors]

# x_test_er = X_val[errors]

#Plot the misclassified numbers

# plt.figure(figsize=(15,9))



# for i in range(30):

#     plt.subplot(5,6,i+1)

#     plt.imshow(x_test_er[i].reshape((-1,28,28,1)),cmap='binary')

#     plt.title( np.argmax(predicted_er[i]), size=13, weight='bold', color='red')

#     plt.axis("off")





# plt.show()



# test = test.values.reshape(-1, 28, 28, 1) / 255

y_pred1 = model.predict(test, batch_size = 64)



y_pred1 = np.argmax(y_pred1,axis = 1)

y_pred1 = pd.Series(y_pred1,name="Label")

y_pred1
y_pred1
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),y_pred1],axis = 1)

submission.to_csv("submission.csv",index=False)