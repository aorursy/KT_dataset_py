# Importing Modules

import numpy as np

import pandas as pd



# Keras modules

from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

from keras.utils import to_categorical



# Train test split

from sklearn.model_selection import train_test_split



# Display and plotting

from IPython.display import Image

import plotly_express as px

from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

init_notebook_mode(connected=True)



import os

print(os.listdir("../input"))
Image(filename = "../input/american_sign_language.PNG")
# Loading train and test sets

train = pd.read_csv("../input/sign_mnist_train.csv")

test = pd.read_csv("../input/sign_mnist_test.csv")
print("Train:")

print(train.head())

print("\nTest:")

print(test.head())
# Looking at training data info

print(train.info())
# Checking distribution of data

label_dist = pd.DataFrame(train['label'].value_counts()).reset_index()

label_dist.columns = ['Label','Count']

px.bar(label_dist,x = "Label", color = "Label", y = "Count")
# Defining X and Ys

X_train = train.iloc[:,1:].copy()

Y_train = train.iloc[:,0].copy()



X_test = test.iloc[:,1:].copy()

Y_test = test.iloc[:,0].copy()
# Splitting training model into train and validation sets for deep learning model

X_train, X_val, Y_train, Y_val = train_test_split(X_train,Y_train, test_size = 0.3)
# Rescaling data to fall between 0 and 1

X_train = X_train/255

X_val = X_val/255

X_test = X_test/255
# Converting to Numpy array and Reshaping X_train and test data

X_train = np.array(X_train).reshape(X_train.shape[0],28,28,1)

X_val = np.array(X_val).reshape(X_val.shape[0],28,28,1)

X_test = np.array(X_test).reshape(X_test.shape[0],28,28,1)
# Categorizing Ys

Y_train = to_categorical(Y_train)

Y_val = to_categorical(Y_val)

Y_test = to_categorical(Y_test)
# Building Keras model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())

model.add(Dense(units = 25, activation = 'softmax'))
# Compiling model

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])
history = model.fit(X_train,Y_train, validation_data = (X_val, Y_val),epochs = 10, batch_size = 64)
data = pd.DataFrame(history.history).reset_index()

data.columns = ['Epoch', "Validation_Loss","Validation_Accuracy","Loss","Accuracy"]

trace1 = go.Scatter(

    x = (data['Epoch'] + 1).values,

    y = data['Loss'].values,

    name = "Loss",

    mode = "lines+markers"

)

trace2 = go.Scatter(

    x = (data['Epoch'] + 1).values,

    y = data['Validation_Loss'].values,

    name = "Validation_Loss",

    mode = "lines+markers"

)

trace3 = go.Scatter(

    x = (data['Epoch'] + 1).values,

    y = data['Validation_Accuracy'].values,

    name = "Validation_Accuracy",

    mode = "lines+markers"

)

trace4 = go.Scatter(

    x = (data['Epoch'] + 1).values,

    y = data['Accuracy'].values,

    name = "Accuracy",

    mode = "lines+markers"

)

fig1 =[trace1,trace2]

fig2 = [trace3,trace4]

iplot(fig1)

iplot(fig2)
# Final Model

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu', input_shape = (28,28,1)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model.add(Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Conv2D(filters = 128, kernel_size = (3,3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = 2))

model.add(Flatten())

model.add(Dense(units = 25, activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['accuracy'])
model.fit(X_train,Y_train, validation_data = (X_val, Y_val),epochs = 4, batch_size = 64)
# Model evaluation

model.evaluate(X_test,Y_test)