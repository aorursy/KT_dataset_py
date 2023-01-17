import numpy as np

import pandas as pd

from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.callbacks import ModelCheckpoint

from subprocess import check_output

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import Binarizer, StandardScaler

import matplotlib.pyplot as plt



print(check_output(["ls", "../input"]).decode("utf8"))
np.random.seed(80)

dataset = pd.read_csv("../input/diabetes.csv")

print('dataset head....')

dataset.head(4)

# Dataset information

dataset.describe()
# Get matrix of features and labels

features = list(dataset.columns.values)

features.remove('Outcome')

x_features = (dataset[features]).values

y_labels = dataset['Outcome'].values
# Preprocessing features 

scaler = StandardScaler()

scaler.fit(x_features)

scaled_x_features = scaler.transform(x_features)
x_train, x_test, y_train, y_test = train_test_split(scaled_x_features, y_labels, test_size=0.20, random_state=0)

print(x_train.shape)

print(x_test.shape)
# Define the model

model = Sequential()

# Input layer

model.add(Dense(64, input_dim=8, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.4))

# Hidden layer 1

model.add(Dense(32, kernel_initializer='uniform', activation='relu'))

model.add(Dropout(0.5))

# Output layer

model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

model.summary()
# Compile the model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Train the model

fit_machine = model.fit(x_train, 

                      y_train,

                      epochs=40,

                      batch_size=80,

                      validation_split=0.10,

                      verbose=1)
# Model Accuracy

plt.plot(fit_machine.history['acc'])

plt.plot(fit_machine.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Model Accuracy

plt.plot(fit_machine.history['loss'])

plt.plot(fit_machine.history['val_loss'])

plt.title('Model Accuracy')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'])

plt.show()
# Evaluate the model

eval_params = model.evaluate(x_test, y_test, verbose=0)

print(eval_params)

print('accuracy: ', eval_params[1]*100)