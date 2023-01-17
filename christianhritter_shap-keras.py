# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from sklearn.model_selection import train_test_split

# Any results you write to the current directory are saved as output.

import shap



from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from keras.optimizers import RMSprop

from keras.callbacks import ReduceLROnPlateau

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding



import matplotlib.pyplot as plt
df = pd.read_csv("/kaggle/input/digit-recognizer-dataset/train.csv")

df.shape
df.head()
df.values.shape
X = df.drop('label',axis=1).values

y = df['label'].values
X = X.reshape(-1,28,28,1)

X.shape
y = to_categorical(y, num_classes = 10)

y.shape
x_train, x_val, y_train, y_val = train_test_split(X,y,random_state=42)
plt.imshow(x_train[0][:,:,0])
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out



model = Sequential()



model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Dropout(0.25))





model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 

                 activation ='relu'))

model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Dropout(0.25))





model.add(Flatten())

model.add(Dense(256, activation = "relu"))

model.add(Dropout(0.5))

model.add(Dense(10, activation = "softmax"))



# Define the optimizer

optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)



# Compile the model

model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])



model.summary()
# Set a learning rate annealer

# reduces lr by factor if no improvement after 3 epochs

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 

                                            patience=3, 

                                            verbose=1, 

                                            factor=0.5, 

                                            min_lr=0.00001)
x_train.shape,y_train.shape,x_val.shape,y_val.shape




epochs = 10 # Turn epochs to 30 to get 0.9967 accuracy

batch_size = 86



history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, 

          validation_data = (x_val, y_val), verbose = 2)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1,figsize=(16,8))

ax[0].plot(history.history['loss'], color='b', label="Training loss",marker='o')

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0],marker='o')

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy",marker='o')

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy",marker='o')

legend = ax[1].legend(loc='best', shadow=True)
y_pred_val = model.predict(x_val)

y_pred_val.shape
# select a set of background examples to take an expectation over

background = x_train[np.random.choice(x_train.shape[0], 100, replace=False)]
# explain predictions of the model on four images

e = shap.GradientExplainer(model, background)
for i in range(1,5):

    plt.figure()

    plt.title(y_pred_val[i])

    plt.imshow(x_val[i][:,:,0])



shap_values = e.shap_values(x_val[1:5])
# plot the feature attributions

shap.image_plot(shap_values, -x_val[1:5])