# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import print_function



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# fonction pour reshape les features d'un example et afficher l'image correspondante

def showDigits(features, title = None):

    f_shape = features.shape

    if len(f_shape) == 0:

        nb_features = 0

    elif len(f_shape) == 1:

        nb_features = 1

        img = features.reshape((28, 28))

        plt.imshow(img)

        if not title is None:

            plt.title(title)

        return None

    elif len(f_shape) > 25:

        nb_features = 25

    else:

        nb_features = f_shape[0]

    

    nb_plots = nb_features

    nb_plots_j = int(np.round(np.sqrt(nb_plots) + 0.5))

    nb_plots_i = nb_plots_j 

    

    fig, axs = plt.subplots(nb_plots_i ,nb_plots_j, figsize=(15, 15), facecolor='w', edgecolor='k')

    axs = axs.ravel()

    

    for p in range(nb_plots):

        img = features[p, :].reshape((28, 28))

        axs[p].imshow(img)

        if not title is None:

            axs[p].set_title(title[p])

        

    return None
train_data = pd.read_csv("../input/train.csv")

train_data.head()
cols = train_data.columns

m = 5000

X_train = np.array(train_data[cols[1:]])[:m, :] # features

y_train = np.array(train_data.label)[:m] # labels

m, n = X_train.shape

print("number of training examples: m = ", m)

print("number of features: n = ", n)
idx = np.random.randint(0, m, 16)

showDigits(X_train[idx, :])
from keras.models import Sequential

from keras.layers import Dense

# fix random seed for reproducibility

np.random.seed(7)
model = Sequential() # mode sequential => couches les unes a la suite des autres

model.add(Dense(50, input_dim=n, activation='sigmoid')) # hidden layer en entree: toutes les features (n = 784) et possede en 50 sorties (pour 50 basic units) 

model.add(Dense(10, activation='sigmoid'))
# Compile model

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model

model.fit(X_train, y_train, epochs=10, batch_size=10)
test_data = pd.read_csv("../input/test.csv")

test_data.head()

# print(test_data.shape)
lut = np.array(range(10))

X_test = np.array(test_data)

predictions = model.predict(X_test)
print(predictions[0])



print(predictions[0] == np.max(predictions[0]) )



lut = np.array(range(10))



print(lut[predictions[0] == np.max(predictions[0])])

idx = np.random.randint(0, m, 16)

title = [str("Prediction: %d" % lut[predictions[i] == np.max(predictions[i])]) for i in idx]

showDigits(X_test[idx, :], title=title)
# hidden_layer = model.layers[0]

# hidden_layer.get_input_at(0)