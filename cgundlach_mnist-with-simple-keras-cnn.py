from keras.models import Sequential, Model
from keras.layers import ZeroPadding2D, Conv2D
from keras.layers import MaxPooling2D, BatchNormalization, Input
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import Adam
import numpy as np
import pandas as pd
# Set the path of the data
PATH = "../input"
# Import dataset
train = pd.read_csv(PATH + "/train.csv")
test = pd.read_csv(PATH + "/test.csv")
# We will resize the vectors into 28x28x1 arrays
# example = X_train.values[1]
# np.resize(example, (28,28, 1)).shape
X_train = []
y_train = []

for index, row in train.iterrows():
    label = row["label"]
    row = row.drop(["label"], axis=0)
    image = np.resize(row.values, (28, 28, 1))
    entry = { 'label': label, 'image': image}
#     data.append(entry)
    X_train.append(image)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)
# train_processed = pd.DataFrame(data=data); train_processed.head()
# data = []
X_test = []

for index, row in test.iterrows():
    image = np.resize(row.values, (28, 28, 1))
    entry = {'image': image}
    X_test.append(image)
#     data.append(entry)
    
X_test = np.array(X_test)
# test_processed = pd.DataFrame(data=data); test_processed.head()
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(n_values=10)
y_train_encoded = enc.fit_transform(y_train.reshape(-1, 1))
inputs = Input(shape=(28,28,1,))

x = ZeroPadding2D((1,1))(inputs)
x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Conv2D(32, kernel_size=(3,3), activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = MaxPooling2D((2,2), strides=(2,2))(x)
x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
x = ZeroPadding2D((1,1))(x)
x = Conv2D(64, kernel_size=(3,3), activation='relu')(x)
# Flatten and dense layers
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.5)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(.5)(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])
train_pct = .75
split_pnt = int(train_pct * len(X_train))
X_val, y_val = X_train[split_pnt:], y_train_encoded[split_pnt:]
X_train, y_train = X_train[:split_pnt], y_train_encoded[:split_pnt]
print(len(X_train), len(X_val))
model.fit(X_train, y_train, batch_size=64, validation_data=(X_val, y_val), epochs=3)

model.fit(X_train, y_train, batch_size=64, validation_data=(X_val, y_val), epochs=8)
model.save("models/mnist.h5")
test.head()
preds = model.predict(X_test)
data = []
for i in range(0, len(preds)):
    entry = { 'ImageId': i+1, 'Label': np.argmax(preds[i]) }
    data.append(entry)
submit = pd.DataFrame(data=data)
submit.head()
submit.to_csv('mnist_submit.csv', index=False)
