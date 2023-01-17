# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns 

import matplotlib.pyplot as plt

from keras.utils.np_utils import to_categorical

from IPython.display import SVG

from keras.utils.vis_utils import model_to_dot

from keras.utils import plot_model

%matplotlib inline 

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import warnings

warnings.filterwarnings(action="ignore")



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashion-mnist_test.csv")
train.shape, test.shape
# ASSIGN X_train AND y_train

X_train = train.drop('label',axis = 1)

y_train = train.label
y_train.value_counts()
plt.rcParams['figure.figsize'] = 10, 8



sns.countplot(y_train);
# Checking for null values in train and test

X_train.isnull().any().sum(), test.isnull().any().sum()
labels = {0 : "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",

          5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle Boot"}



sample_images = []

sample_labels = []



for k in labels.keys():

    samples = train[train["label"] == k].head(4)

    for j, s in enumerate(samples.values):

        img = np.array(samples.iloc[j, 1:]).reshape(28,28)

        sample_images.append(img)

        sample_labels.append(samples.iloc[j, 0])
f, ax = plt.subplots(5,8, figsize=(16,10))



for i, img in enumerate(sample_images):

    ax[i//8, i%8].imshow(img, cmap='Greens')

    ax[i//8, i%8].axis('off')

    ax[i//8, i%8].set_title(labels[sample_labels[i]])

plt.show()   
sample_images = []

sample_labels = []



for k in labels.keys():

    samples = test[test["label"] == k].head(4)

    for j, s in enumerate(samples.values):

        img = np.array(samples.iloc[j, 1:]).reshape(28,28)

        sample_images.append(img)

        sample_labels.append(samples.iloc[j, 0])
f, ax = plt.subplots(5,8, figsize=(16,10))



for i, img in enumerate(sample_images):

    ax[i//8, i%8].imshow(img, cmap='Blues')

    ax[i//8, i%8].axis('off')

    ax[i//8, i%8].set_title(labels[sample_labels[i]])

plt.show()   
# data preprocessing

def data_preprocessing(raw):

    out_y = to_categorical(raw.label, num_classes = 10)

    num_images = raw.shape[0]

    x_as_array = raw.values[:,1:]

    x_shaped_array = x_as_array.reshape(num_images, 28, 28, 1)

    out_x = x_shaped_array / 255

    return out_x, out_y
X, y = data_preprocessing(train)

X_test, y_test = data_preprocessing(test)
from sklearn.model_selection import train_test_split



X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state= 5)
# Example

g = plt.imshow(X_train[0][:,:,0])
print("Fashion MNIST train -  rows:",X_train.shape[0]," columns:", X_train.shape[1:4])

print("Fashion MNIST valid -  rows:",X_val.shape[0]," columns:", X_val.shape[1:4])

print("Fashion MNIST test -  rows:",X_test.shape[0]," columns:", X_test.shape[1:4])
sns.countplot(np.argmax(y_train,axis=1));
from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
model = Sequential()



model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu',

                 input_shape = (28, 28, 1)))

model.add(BatchNormalization())

model.add(Conv2D(filters = 16, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation='relu'))

model.add(BatchNormalization())

model.add(MaxPool2D(strides=(2,2)))

model.add(Dropout(0.25))



model.add(Flatten())

model.add(Dense(512, activation='relu'))

model.add(Dropout(0.25))

model.add(Dense(1024, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
model.summary()
plot_model(model, to_file='model.png')

SVG(model_to_dot(model).create(prog='dot', format='svg'))
from keras.optimizers import Adamax

optimizer = Adamax(lr=0.001)

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
train_model = model.fit(X_train, y_train,

                  batch_size=86,

                  epochs=50,

                  verbose=1,

                  validation_data=(X_val, y_val),

                  callbacks=[learning_rate_reduction])
def create_trace(x,y,ylabel,color):

        trace = go.Scatter(

            x = x,y = y,

            name=ylabel,

            marker=dict(color=color),

            mode = "markers+lines",

            text=x

        )

        return trace

    

def plot_accuracy_and_loss(train_model):

    hist = train_model.history

    acc = hist['acc']

    val_acc = hist['val_acc']

    loss = hist['loss']

    val_loss = hist['val_loss']

    epochs = list(range(1,len(acc)+1))

    

    trace_ta = create_trace(epochs,acc,"Training accuracy", "Green")

    trace_va = create_trace(epochs,val_acc,"Validation accuracy", "Red")

    trace_tl = create_trace(epochs,loss,"Training loss", "Blue")

    trace_vl = create_trace(epochs,val_loss,"Validation loss", "Magenta")

   

    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=('Training and validation accuracy',

                                                             'Training and validation loss'))

    fig.append_trace(trace_ta,1,1)

    fig.append_trace(trace_va,1,1)

    fig.append_trace(trace_tl,1,2)

    fig.append_trace(trace_vl,1,2)

    fig['layout']['xaxis'].update(title = 'Epoch')

    fig['layout']['xaxis2'].update(title = 'Epoch')

    fig['layout']['yaxis'].update(title = 'Accuracy', range=[0,1])

    fig['layout']['yaxis2'].update(title = 'Loss', range=[0,1])



    

    iplot(fig, filename='accuracy-loss')



plot_accuracy_and_loss(train_model);
score = model.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
from sklearn.metrics import confusion_matrix



# Predict the validation dataset

y_pred = model.predict(X_test)

# Convert predictions to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix



plt.rcParams['figure.figsize'] = 14, 12

sns.heatmap(confusion_mtx, annot=True, fmt='d');