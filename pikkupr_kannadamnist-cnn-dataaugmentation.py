# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt # to plot charts



from sklearn.metrics import accuracy_score, classification_report

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# from tensorflow import set_random_seed



seed = 108

np.random.RandomState(seed)

# set_random_seed(seed)
# Kers modules

from keras.optimizers import SGD

from keras.models import Sequential

from keras.utils import to_categorical

from keras.preprocessing.image import ImageDataGenerator

from keras.layers.normalization import BatchNormalization

from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPool2D

from keras.callbacks import EarlyStopping, History, LearningRateScheduler
# path where the dataset is kept

data_dir = "/kaggle/input/Kannada-MNIST/"
# read the data from the csv

train_df = pd.read_csv(data_dir+"train.csv")

train_df.head(2)
# check the number of rows and columns

train_df.shape
# extract the labels from the dataframe

y = train_df.values[:, 0]



# convert the y to categorical using one-hot encoding

y = to_categorical(y)

print("Shape of y: ", y.shape)

print("Sample of y: ", y[0])
# extract the pixel values from the dataframe

X = train_df.values[:, 1:]/255.0 # all the columns but 1st



# reshape each row into 28x28 size

X = X.reshape(-1, 28, 28, 1) # -1 tells the system to automatically figure out the size of the first dimention



print("Shape of X: ", X.shape)

print("Sample of X: ")

plt.imshow(X[0].reshape(28, 28))
# split the data into train(85%) and test (15%)

validation_split = .15



# stratify makes sure that data of all the classes - Tshirt, Trouser, etc. are split equally between train and test

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, stratify=y, random_state=seed)
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
class CNN_Modeling():

    

    def __init__(self, model_confs):

        self.epochs = 30

        self.batch_size = 80

        self.annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose=0)

        

        self.models = []

        self.history = []

        self.model_names = []

        self.model_confs = model_confs

    

    def build_models(self, feature_maps, kernel_size, dense_size, drop_rate):

        model = Sequential()



        # add the convolution and max pool layers with provided kernel_size

        for i, fm in enumerate(feature_maps):

            # add conv layer

            model.add(Conv2D(fm, kernel_size=kernel_size, padding='same', activation='relu', input_shape=(28, 28, 1)))



            # add MaxPool

            if i == len(feature_maps)-1:

                model.add(MaxPool2D())

            else:

                model.add(MaxPool2D(padding='same'))



            # add the Dropout

            model.add(Dropout(drop_rate))



        # convert the output from the convolution layers into a linear array

        model.add(Flatten())



        # add a dense layer with size - dense_size

        for dns in dense_size:

            if dns>0:

                model.add(Dense(dns, activation='relu'))

                model.add(Dropout(drop_rate))



        # add the final softmax layer with size equal to the number of categories

        model.add(Dense(10, activation='softmax'))



        # compile the model

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



        # return the model

        return model

    

    

    def get_model_name(self, conf):

        # add the convolution layers name

        name = "-".join(

            ["{}C{}P1".format(ft, conf['kernel_size']) for ft in conf['feature_maps']])



        # add the dense layer size

        name = name + "-" + "-".join(map(str, conf['dense_size']))



        # add the drop out

        name = name + '-D%d'%round(conf["drop_rate"]*100)



        return name    

    

    

    def train_models(self, _x_train, _y_train, _x_val, _y_val):

        # to store the models and their history

        self.models = [None]*len(self.model_confs)

        self.history = [None]*len(self.model_confs)

        self.model_names = [None]*len(self.model_confs)

        

        for i, model_conf in enumerate(self.model_confs):

            

            # create the model

            self.models[i] = self.build_models(**model_conf)

            

            # get and store the model name

            self.model_names[i] = self.get_model_name(model_conf)



            # fir the model

            self.history[i] = self.models[i].fit(_x_train,_y_train, batch_size=self.batch_size, epochs=self.epochs, 

                          validation_data = (_x_val,_y_val), callbacks=[self.annealer], verbose=0)



            print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

                self.model_names[i], self.epochs, max(self.history[i].history['accuracy']), 

                max(self.history[i].history['val_accuracy']))

            )

            

            

    def plot_accuracy_chart(self, accuracy='val_accuracy'):

        # set the image size

        plt.figure(figsize=(15,5))

        

        # plot the accuracy lines

        for i in range(len(self.models)):

            sns.lineplot(

                x=range(self.epochs),

                y=self.history[i].history[accuracy], 

                label=self.model_names[i]

            )

            

            

    def predict(self, model_name, _x_predict):

        # get the model

        given_model = self.models[self.model_names.index(model_name)]

        

        return given_model.predict_classes(_x_predict)

    

    

    def get_model(self, model_name):

        # find and return the model

        return self.models[self.model_names.index(model_name)]
# configuration of each of the model

model_confs = [{"feature_maps": [32, 64, 128], "kernel_size": 5, "dense_size": [256], "drop_rate": 0.3}]



cnn_model_before_data_aug = CNN_Modeling(model_confs)

cnn_model_before_data_aug.train_models(X_train, y_train, X_val, y_val)

cnn_model_before_data_aug.plot_accuracy_chart()
# get the predicted values of the validation dataset

val_pred_labels = cnn_model_before_data_aug.predict("32C5P1-64C5P1-128C5P1-256-D30", X_val)



# get the labels for the y_val

y_val_labels = np.argmax(y_val, axis=1)



# misclassifications

misclassses = np.where(np.argmax(y_val, axis=1) != val_pred_labels)[0]



print("Number of misclassifications: ", len(misclassses))
plt.figure(figsize=(10,5))

for i in range(18):

    plt.subplot(3,6,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(X_val[misclassses[i]].reshape(28, 28), cmap=plt.cm.binary)

    plt.xlabel("{} --> {}".format(y_val_labels[misclassses[i]], val_pred_labels[misclassses[i]]))
# set the configuration of the ImageGenerator

datagen = ImageDataGenerator(

            rotation_range=10,  

            zoom_range = 0.1,  

            width_shift_range=0.1, 

            height_shift_range=0.1

)



# get the best model

model_name = "32C5P1-64C5P1-128C5P1-256-D30"

bes_model_so_far = cnn_model_before_data_aug.get_model(model_name)



# train the best model

epochs = 50

history = bes_model_so_far.fit_generator(datagen.flow(X_train, y_train, batch_size=64), 

                epochs=epochs, steps_per_epoch=X_train.shape[0]//64,validation_data=(X_val,y_val), 

                callbacks=[cnn_model_before_data_aug.annealer], verbose=0)



print("CNN {0}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

    model_name,epochs,max(history.history['accuracy']),max(history.history['val_accuracy'])))
plt.figure(figsize=(15,5))

sns.lineplot(

    x=range(epochs),

    y=history.history["val_accuracy"], 

    label=model_name

)
predict_df = pd.read_csv(data_dir+"test.csv")

predict_df.head(2)
# get the labels of the data

img_ids = predict_df.values[:, 0]



# extract the pixel values from the dataframe

X_predict = predict_df.values[:, 1:]/255.0 # all the columns but 1st



# reshape each row into 28x28 size

X_predict = X_predict.reshape(-1, 28, 28, 1) # -1 tells the system to automatically figure out the size of the first dimention



print("Shape of X: ", X_predict.shape)

print("Sample of X: ")

plt.imshow(X_predict[0].reshape(28, 28))
predicted_labels = bes_model_so_far.predict_classes(X_predict)
final_prediction = pd.DataFrame()

final_prediction['id'] = img_ids

final_prediction['label'] = predicted_labels

final_prediction.to_csv("submission.csv", index=False)