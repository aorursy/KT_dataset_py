# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import pickle



from keras.utils.np_utils import to_categorical

import matplotlib.pyplot as plt

from keras.layers import *

from keras.activations import *

from keras.models import *

from keras.optimizers import *

from keras.initializers import *

from keras.callbacks import *

from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



        

# Any results you write to the current directory are saved as output.
# data class

class TRAFFIC:

    def __init__(self, dataset="1"):

        

        #load data

        with open('../input/traffic-signs-preprocessed/data'+dataset+'.pickle', 'rb') as file:

            self.data = pickle.load(file, encoding='latin1')

         

        # transpose for images 32 x 32 x 3

        self.x_train = self.data['x_train'].transpose(0, 2, 3, 1)

        self.x_validation = self.data['x_validation'].transpose(0, 2, 3, 1)

        self.x_test = self.data['x_test'].transpose(0, 2, 3, 1)

            

        # shapes of pictures

        self.height = self.x_train.shape[1]

        self.width = self.x_train.shape[2]

        self.depth = self.x_train.shape[3]

        

        #sizes

        self.train_size = self.x_train.shape[0]

        self.validation_size = self.x_validation.shape[0]

        self.test_size = self.x_test.shape[0]

        

        self.labels = pd.read_csv('../input/traffic-signs-preprocessed/label_names.csv', delimiter=',')

        self.num_classes = self.labels.shape[0]  

        

        # One hot encoding

        self.y_train = to_categorical(self.data['y_train'], num_classes=self.num_classes)

        self.y_validation = to_categorical(self.data['y_validation'], num_classes=self.num_classes)

        self.y_test = to_categorical(self.data['y_test'], num_classes=self.num_classes)

        

    def info(self):

        print("x_train: ", self.x_train.shape)

        print("x_validation: ", self.x_validation.shape)

        print("x_test: ", self.x_test.shape)

        print("y_train: ", self.y_train.shape)

        print("y_validation: ", self.y_validation.shape)

        print("y_test: ", self.y_test.shape)

        
# Define the CNN

def create_model(optimizer, lr, width, height, depth, num_classes):

    

    input_img = Input(shape=(width, height, depth))



    x = Conv2D(filters=32, kernel_size=3, strides=(1, 1), padding="same")(input_img)

    x = BatchNormalization()(x)

    x = Activation("relu")(x) 

    x = MaxPool2D()(x)

    

    x = Conv2D(filters=64, kernel_size=9, strides=(1, 1), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x) 

    x = MaxPool2D()(x)

    

    x = Conv2D(filters=128, kernel_size=18, strides=(1, 1), padding="same")(x)

    x = BatchNormalization()(x)

    x = Activation("relu")(x) 

    x = MaxPool2D()(x)

    

    x = Flatten()(x)

    x = Dense(128)(x)

    x = Activation("relu")(x)

    x = Dense(64)(x)

    x = Activation("relu")(x)

    x = Dense(num_classes)(x)

    output_pred = Activation("softmax")(x)

   

    optimizer = optimizer(lr=lr)

    model = Model(inputs=input_img, outputs=output_pred)

    model.compile(

        loss="categorical_crossentropy", 

        optimizer=optimizer, 

        metrics=["accuracy"])

    model.summary()

    

    return model
# load data

data = TRAFFIC(dataset="1")



#print info

data.info()



#show example images



rows = 5

cols = 5



fig, axs = plt.subplots(rows,cols, figsize = (25,25))



for i in range(rows):

    for j in range(cols):

        ClassId = np.argmax(data.y_train[rows*i+j])

        label = data.labels[data.labels["ClassId"] == ClassId]["SignName"].to_string()        

        axs[i,j].imshow(data.x_train[rows*i+j])

        axs[i,j].set_title(label)

    

fig.show()
# hyperparameter

lr = 1e-3

optimizer = Adam

batch_size = 8

epochs = 15

do_grid_search = False
# create model

model = create_model(optimizer, lr, data.width, data.height, data.depth, data.num_classes)
# define learning rate sheduler

def schedule(epoch):

    lr = 0.001/(epoch+1)



    return lr



lrs = LearningRateScheduler(

    schedule=schedule,

    verbose=1)
# For grid search

if do_grid_search == True:

    model = KerasClassifier(

        build_fn=create_model,

        epochs=epochs,

        batch_size=batch_size,

        verbose=1,

        width = data.width,

        height = data.height,

        depth = data.depth,

        num_classes = data.num_classes,

        lr = 0.001)



    #candidates

    optimizer_candidates = [Adam, RMSprop]

    batch_size_candidates = [8, 16, 32, 64]



    param_grid = {

        "optimizer": optimizer_candidates,

        "batch_size": batch_size_candidates}



    grid = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        n_jobs=1,

        verbose=1,

        cv=3)



    # fit

    grid_result = grid.fit(data.x_train, data.y_train)



    # Summary

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    means = grid_result.cv_results_["mean_test_score"]

    stds = grid_result.cv_results_["std_test_score"]

    params = grid_result.cv_results_["params"]



    for mean, stdev, param in zip(means, stds, params):

        print("%f (%f) with: %r" % (mean, stdev, param))
# training

history = model.fit(

    x=data.x_train, 

    y=data.y_train, 

    verbose=1, 

    #batch_size=batch_size, 

    epochs=epochs, 

    validation_data=(data.x_validation, data.y_validation),

    callbacks=[lrs])
# summarize history for accuracy

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
# score on test data

score = model.evaluate(data.x_test, data.y_test, batch_size=batch_size)

print("Test performance: ", score)
# store wrong predictions



pred = model.predict(data.x_test)



# init

wrong = np.array([])



for k in range(pred.shape[0]):

    ClassId_pred = np.argmax(pred[k])

    ClassId_true= np.argmax(data.y_test[k])

    if ClassId_pred != ClassId_true: 

        wrong = np.append(wrong, k)

        

print("Number of wrong predictions: ", wrong.size)

print("Percentage if wrong predictions: {0:.3f}".format((wrong.size/pred.shape[0])*100), "%")
# show example images

rows = 5

cols = 5



fig, axs = plt.subplots(rows,cols, figsize = (25,25))



for i in range(rows):

    for j in range(cols):

            ClassId_pred = np.argmax(pred[int(wrong[rows*i+j])])

            label_pred = data.labels[data.labels["ClassId"] == ClassId_pred]["SignName"].to_string()

            ClassId_true= np.argmax(data.y_test[int(wrong[rows*i+j])])

            label_true = data.labels[data.labels["ClassId"] == ClassId_true]["SignName"].to_string()

            axs[i,j].imshow(data.x_test[int(wrong[rows*i+j])])

            axs[i,j].set_title("pred: "+label_pred+"\n true: "+label_true)

    

fig.show()