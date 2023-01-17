# Installing required libraries

import pandas as pd

import tensorflow as tf

from sklearn.datasets import load_breast_cancer

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
# Loading the dataset and storing it into variable data

data=load_breast_cancer()
# Type of data is bunch meaning the actual data is inside a particular key inside this bunch.

print('Type of data ' + str(type(data)))

print('Keys inside the bunch are '+ str(data.keys()))
# Checking the shape of actual data

print('No of rows: '+str(data.data.shape[0]))

print('No of columns: '+str(data.data.shape[1]))

print('Type of data: ' + str(type(data.data)))
# Checking the target variable and target names

print('Target variables: ' + str(data.target_names))

print('Feature Names: ' + str(data.feature_names))
# Splitting the data into training and testing set and standardizing the data.

X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33) 

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
# Building the model

model = tf.keras.models.Sequential()

# Using sigmoid activation to get the output of either 0 or 1

# Units means neurons. Lets test with 4 neurons. Input shape is number of features.

# Using single layer.

model.add(tf.keras.layers.Dense(units=1, input_shape=(30,), activation='sigmoid'))

# compiling the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

r = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)
print('Train Score: ' + str(model.evaluate(X_train, y_train)))

print('Test Score: ' + str(model.evaluate(X_test, y_test)))

loss_accuracy_df = pd.DataFrame(model.history.history)

loss_accuracy_df.plot()