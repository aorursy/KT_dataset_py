import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns





import os

print(os.listdir("../input"))



import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv("../input/train.csv")
train.info()
train.shape
train.Id.describe()
train.head(10)
train.tail(10)
y_train = train["Id"]

X_train = train.drop(labels = ["Id"], axis = 1)

y_train.head()
train.isnull().sum().sum()
from keras.preprocessing import image

from keras.applications.imagenet_utils import preprocess_input



def prepareImages(train, shape, path):

    

    x_train = np.zeros((shape, 100, 100, 3))

    count = 0

    

    for fig in train['Image']:

        

       

        img = image.load_img("../input/"+path+"/"+fig, target_size=(100, 100, 3))

        x = image.img_to_array(img)

        x = preprocess_input(x)



        x_train[count] = x

        if (count%500 == 0):

            print("Processing image: ", count+1, ", ", fig)

        count += 1

    

    return x_train
x_train = prepareImages(train, train.shape[0], "train")
x_train = x_train / 255.0

print("x_train shape: ",x_train.shape)
plt.imshow(x_train[0][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[0,0]))

plt.axis("off")

plt.show()
plt.imshow(x_train[25360][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[25360,0]))

plt.axis("off")

plt.show()
plt.imshow(x_train[55][:,:,0], cmap="gray")

plt.title(plt.title(train.iloc[55,0]))

plt.axis("off")

plt.show()
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_train[0:10]  
y_train.shape
from keras.utils.np_utils import to_categorical

y_train = to_categorical(y_train, num_classes = 5005)
print(y_train.shape)

y_train 
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential # to create a cnn model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop,Adam

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau



model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu', input_shape = (100,100,3)))

model.add(Conv2D(filters = 16, kernel_size = (5,5), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), padding = 'Same', activation = 'relu'))

model.add(MaxPool2D(pool_size = (2,2), strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(256, activation = 'relu'))

model.add(BatchNormalization())

model.add(Dense(y_train.shape[1], activation = "softmax"))
model.summary()
optimizer = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics=["accuracy"])
epochs = 100

batch_size = 1000
history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, callbacks=[learning_rate_reduction])
plt.plot(history.history['loss'], color='r', label="Train Loss")

plt.title("Train Loss")

plt.xlabel("Number of Epochs")

plt.ylabel("Loss")

plt.legend()

plt.show()
plt.plot(history.history['acc'], color='g', label="Train Accuracy")

plt.title("Train Accuracy")

plt.xlabel("Number of Epochs")

plt.ylabel("Accuracy")

plt.legend()

plt.show()
print('Train accuracy of the model: ',history.history['acc'][-1])
print('Train loss of the model: ',history.history['loss'][-1])
test = os.listdir("../input/test/")

print(len(test))
col = ['Image']

test_data = pd.DataFrame(test, columns=col)

test_data['Id'] = ''
x_test = prepareImages(test_data, test_data.shape[0], "test")

x_test /= 255
predictions = model.predict(np.array(x_test), verbose=1)
for i, pred in enumerate(predictions):

    test_data.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))
test_data.head(10)

test_data.to_csv('submission_3.csv', index=False)