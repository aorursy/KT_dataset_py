import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input, Activation, BatchNormalization, LeakyReLU

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from tensorflow.keras.utils import plot_model



from sklearn.metrics import accuracy_score, classification_report
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
training_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_train.csv')

testing_data = pd.read_csv('/kaggle/input/fashionmnist/fashion-mnist_test.csv')
training_data.head()
targets = { 0: 'T-shirt/top',

            1: 'Trouser',

            2: 'Pullover',

            3: 'Dress',

            4: 'Coat',

            5: 'Sandal',

            6: 'Shirt',

            7: 'Sneaker',

            8: 'Bag',

            9: 'Ankle boot'}
print(f"No.of null values in testing_data = {training_data.isnull().values.sum()}")

print(f"No.of null values in testing_data = {testing_data.isnull().values.sum()}")
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (16,6))



sns.countplot(x = 'label', data = training_data, ax = ax[0])

sns.countplot(x = 'label', data = testing_data, ax = ax[1])

X = training_data.drop('label', axis = 1)

y = training_data.loc[:, 'label']



X_test = testing_data.drop('label', axis = 1)

y_test = testing_data.loc[:, 'label']
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size = 0.8, random_state = 50)
fig, ax = plt.subplots(nrows = 1, ncols = 2, sharey = True, figsize = (16,6))



sns.countplot(y_train, ax = ax[0])

sns.countplot(y_val, ax = ax[1])

def standardize(df):

    df = df / 255.0

    return np.array(df)





X_train = standardize(X_train)

X_test = standardize(X_test)

X_val = standardize(X_val)
X_train[0]
X_train = X_train.reshape(-1,28,28,1)

X_test = X_test.reshape(-1,28,28,1)

X_val = X_val.reshape(-1,28,28,1)
print(f"Train_set - {X_train.shape}\nTest_set - {X_test.shape}\nVal_set - {X_val.shape}")
i = 0



plt.imshow(X_train[i].reshape(28,28))

plt.title(targets[y_train[i]])
datagen = ImageDataGenerator(horizontal_flip = True)

datagen.fit(X_train)
def functional_api_model_architecture(inputs):

    

    x = inputs

    x = Conv2D(filters = 256, kernel_size = (3,3),input_shape = (28,28,1))(x)

    x = LeakyReLU()(x)

    x = Conv2D(filters = 256, kernel_size = (3,3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization(axis = -1)(x)

    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Dropout(rate = 0.3)(x)



    x = Conv2D(filters = 256,kernel_size = (3,3))(x)

    x = LeakyReLU()(x)

    x = Conv2D(filters = 256,kernel_size = (3,3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization(axis = -1)(x)

    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Dropout(rate = 0.3)(x)

    

    x = Conv2D(filters = 256,kernel_size = (3,3))(x)

    x = LeakyReLU()(x)

    x = BatchNormalization(axis = -1)(x)

    x = MaxPooling2D(pool_size = (2,2))(x)

    x = Dropout(rate = 0.3)(x)



    x = Flatten()(x)

    x = Dense(units = 64)(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(rate = 0.5)(x)



    x = Dense(units = 128)(x)

    x = Activation('relu')(x)

    x = BatchNormalization()(x)

    x = Dropout(rate = 0.5)(x)



    x = Dense(units = 10)(x)

    x = Activation('softmax', name = 'output_cat')(x)



    return x
def functional_api_model_init():

    

    inputs = Input(shape = (28,28,1))

    model = Model(inputs = inputs, outputs = functional_api_model_architecture(inputs))

    model.summary()

    plot_model(model, "my_model.png", show_shapes=True)

    print("Your Model was successfully created")

    

    return model
model = functional_api_model_init()
from IPython.display import Image

Image("/kaggle/working/my_model.png")
adam = Adam(learning_rate = 0.001)

loss = { 'output_cat' : 'sparse_categorical_crossentropy'}
model.compile(optimizer = adam, loss = loss, metrics = ['accuracy'])
earlystopping = EarlyStopping(patience = 7, monitor='val_loss')

reduce_LR_plateau = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 4)
epochs = 50

batch_size = 256
hist = model.fit(datagen.flow(X_train, y_train), epochs = epochs, validation_data = (X_val, y_val),

          batch_size = batch_size, callbacks = [earlystopping, reduce_LR_plateau], verbose = 1)
fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (10,18))



ax[0].plot(hist.history['val_accuracy'], c = 'b', label = 'Validation_accuracy')

ax[0].plot(hist.history['accuracy'], c = 'r', label = 'Training accuracy')

ax[0].set_title('Validation Accuracy vs Training Accuracy')



ax[1].plot(hist.history['val_loss'], c = 'b', label = 'Validation Loss')

ax[1].plot(hist.history['loss'], c = 'r', label = 'Training Loss')

ax[1].set_title('Validation Loss vs Training Loss')



plt.legend()
predictions = model.predict(X_test)

predictions = np.argmax(predictions, axis = 1)
print(f"Accuracy Score : {accuracy_score(y_test, predictions)}")

print(f"\n\n\n{classification_report(y_test, predictions)}")
false_predictions = [i for i in range(y_test.shape[0]) if predictions[i] != y_test[i]]
rows = 5 

cols = 5

index = 0



fig, ax = plt.subplots(nrows = rows, ncols = cols, figsize = (15,20))



for i in range(5):

    for j in range(5):

        false_index = false_predictions[index]

        ax[i][j].imshow(X_test[false_index, :,:,-1])

        ax[i][j].set_title(f"P : {targets[predictions[false_index]]}\nA : {targets[y_test[false_index]]}")

        index += 1

        ax[i][j].axis("off")

        



plt.show()

plt.tight_layout()
import cv2 

import requests

from PIL import Image

from io import BytesIO
def preprocess_image(url):

    

    response = requests.get(url)

    img = Image.open(BytesIO(response.content))

    

    fig, ax = plt.subplots(nrows = 1, ncols = 3,figsize = (15,20))

    ax[0].imshow(img)

    ax[0].set_title("Image")

    

    

    ## Grayscale and Normalization

    img = np.array(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   ## From 3 channel RGB to 1 channel Grayscale

    print(f"Shape of image array after converting it to Grayscale : {img.shape}")

    img = img / 255.0

    ax[1].imshow(img)

    ax[1].set_title("Grayscale Image")

    

    ## Resizing

    img = cv2.resize(img, (28,28))

    print(f"Shape of image after Grayscale and Resizing : {img.shape}")

    ax[2].imshow(img)

    ax[2].set_title("Grayscale & Resized")   

    

    plt.tight_layout()

    

    ## Making it model ready

    img = np.expand_dims(img, axis = [0,3])

    print(f"At last ready to be predicted by model, shape - {img.shape}")

    

    return img





def predict_image(url):

    

    img = preprocess_image(url)

    predicted_label = model.predict(img)

    predicted_label = np.argmax(predicted_label, axis = 1)[0]

    return targets[predicted_label]

predict_image("https://images.all-free-download.com/images/graphicthumb/blank_black_tshirt_stock_photo_168263.jpg")
predict_image("https://dievca.files.wordpress.com/2014/06/victoria-secret-goddess-maxi-dress-white-black-background.jpg")