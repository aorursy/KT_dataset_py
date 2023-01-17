# Imports
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from keras import utils
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
# Get Data

# Labeld for training
train_df = pd.read_csv("../input/train.csv")

# Unlabeld for prediction!! 
test_df = pd.read_csv("../input/test.csv")
# split training data to obtain training and test set 
y = train_df["label"].copy()
X = train_df.drop("label", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# One hot encoding
y_train_enc = utils.to_categorical(y_train, num_classes=10)
y_test_enc = utils.to_categorical(y_test, num_classes=10)
# Normalization
X_train = X_train/255
X_test = X_test/255
# Set up Neural Network
model = Sequential()
model.add(Dense(units=784,  input_dim=784))
model.add(Dense(units=10, activation='softmax'))

# store weights to reload later
model.save_weights('model.h5')
# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit training data
history = model.fit(X_train, y_train_enc, validation_data=(X_test, y_test_enc), epochs=5, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test_enc, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
history_df = pd.DataFrame.from_dict(history.history)
plt.plot(history_df["acc"],'b', label="accuracy")
plt.plot(history_df["loss"],'r', label="loss")
plt.title("Test: Accuracy vs. Loss")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.legend()
plt.plot(history_df["val_acc"],'b', label="accuracy")
plt.plot(history_df["val_loss"],'r', label="loss")
plt.title("Validation: Accuracy vs. Loss")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.legend()
# One hot encoding
y_enc = utils.to_categorical(y, num_classes=10)

# Normalize
X = X/255

# reset model with initial weights
model.load_weights('model.h5')
# Compile
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit training data
history = model.fit(X, y_enc, epochs=5, batch_size=200, verbose=2)

# Final evaluation of the model
scores = model.evaluate(X, y_enc, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# Normaoization
test_df = test_df/255

# Predict on Trainingsdataset
y_pred = model.predict(test_df, batch_size = 5, verbose = 1, steps = None)
# select the index with the maximum probability
results = np.argmax(y_pred,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
