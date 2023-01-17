import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import keras

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import *

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import to_categorical

from sklearn.model_selection import GridSearchCV

import pprint

pp = pprint.PrettyPrinter(indent = 4)
import os

print(os.listdir("../input"))
train = "../input/fashion-mnist_train.csv"

test  = "../input/fashion-mnist_test.csv"
def read_dataset(data_file):

    df = pd.read_csv(data_file)

    label_column = 'label'

    y = df[label_column].values

    X = df.drop(label_column, axis=1).values

    return (X, y)
(x_train, y_train) = read_dataset(train)

(x_test, y_test) = read_dataset(test)
print("There are {} fashion images with {} pixels in x_train dataset. \n".format(x_train.shape[0],x_train.shape[1]))



print("There are {} fashion Labels in y_train dataset. \n".format(y_train.shape[0]))



print("There are {} fashion images with {} pixels in x_train dataset. \n".format(x_test.shape[0],x_test.shape[1]))



print("There are {} fashion Labels in y_test dataset. \n".format(y_test.shape[0]))
f, ax = plt.subplots(1,5)

f.set_size_inches(80, 40)

for i in range(5,10):

    ax[i-5].imshow(x_train[i].reshape(28, 28))

plt.show()
def draw_articles(articles, labels):

    fig, axs = plt.subplots(1, len(articles), figsize=(30,30))

    for i in range(len(articles)):

        axs[i].set_title(labels[i])

        axs[i].imshow(articles[i].reshape((28,28)), cmap=plt.cm.binary)

    plt.show()
def ImageDisplay(list_data, label, one_hot=False):

    fig = pyplot.figure()

    axis = fig.add_subplot(1,1,1)

    list_data=np.reshape(list_data, (28,28))

    plot_img = axis.imshow(list_data, cmap=mpl.cm.Greys)

    plot_img.set_interpolation('none')

    if one_hot :

        ShowLabelName (label)

    else:

        print ("Label : "+str(CLASSES[str(label)]))
label_map = {0: 'T-Shirt/Top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}

examples = []

labels = []



for i in label_map:

    k = np.where(y_train==i)[0][0]

    examples.append(x_train[k])

    labels.append(label_map[i])

draw_articles(examples, labels)
x_train = x_train.astype('float32') / 255

y_train = to_categorical(y_train)



x_test = x_test.astype('float32') / 255

y_test = to_categorical(y_test)
def build_model(optimizer, learning_rate, activation, dropout_rate,

                initializer,num_unit):

    keras.backend.clear_session()

    model = Sequential()

    model.add(Dense(num_unit, kernel_initializer=initializer,

                    activation=activation, input_shape=(784,)))

    model.add(Dropout(dropout_rate))

    model.add(Dense(num_unit, kernel_initializer=initializer,

                    activation=activation))

    model.add(Dropout(dropout_rate)) 

    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy',

                  optimizer=optimizer(lr=learning_rate),

                  metrics=['accuracy'])

    return model
# [:1] is for testing



batch_size = [20, 50, 100][:1]



epochs = [1, 20, 50][:1]



initializer = ['lecun_uniform', 'normal', 'he_normal', 'he_uniform'][:1]



learning_rate = [0.1, 0.001, 0.02][:1]



dropout_rate = [0.3, 0.2, 0.8][:1]



num_unit = [10, 5][:1]



activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear'][:1]



optimizer = [SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam][:1]
# Creat the wrapper and pass params to GridSearchCV 

# parameters is a dict with all values



parameters = dict(batch_size = batch_size,

                  epochs = epochs,

                  dropout_rate = dropout_rate,

                  num_unit = num_unit,

                  initializer = initializer,

                  learning_rate = learning_rate,

                  activation = activation,

                  optimizer = optimizer)



model = KerasClassifier(build_fn=build_model, verbose=0)



models = GridSearchCV(estimator = model, param_grid=parameters, n_jobs=1)
best_model = models.fit(x_train, y_train)

print('Best model :')

pp.pprint(best_model.best_params_)