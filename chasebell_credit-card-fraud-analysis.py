import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras import models
from keras import backend as K
from keras import regularizers

# Clearing Keras backend
K.clear_session()

# setting matplotlib style to ggplot
plt.style.use('ggplot')
%matplotlib inline
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def build_model():
    """Function that builds a densely connect neural network"""
    model = models.Sequential()
    model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.002), activation='relu', input_shape=(train_x.shape[1], )))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(32, kernel_regularizer=regularizers.l2(0.002), activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.002), activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model
df = pd.read_csv('../input/creditcardfraud/creditcard.csv')
fraud = df.loc[df['Class'] == 1]

# this returns a subset of all fraudulent cases, a total of 492 rows. Let's select 5000 samples and use the class_weight parameter
clean = df.loc[df['Class'] == 0].iloc[:5000]

# Concatenating the fraud and non-fraud datasets
data = pd.concat([clean, fraud])

# Randomly shuffling the dataset and reseting the index
data = data.sample(frac=1).reset_index(drop=True)
data.head()
# Splitting the dataset into training and testing

index = int(data.shape[0] * 0.25)

# Dropping the 'Time' column and splitting the labels into seperate dataframes
test_x = data[data.columns[1:-2]].iloc[:index]
test_y = data[data.columns[-1]].iloc[:index]

train_x = data[data.columns[1:-2]].iloc[index:]
train_y = data[data.columns[-1]].iloc[index:]

mean = train_x.mean()
std = train_x.std()
train_x = (train_x - mean)/std

# Plotting the distribution after standardization
train_x.plot.hist(bins = 50, color='b', legend=None)
plt.title("Training Data Standardized Distribution")
plt.show()
test_x -= mean
test_x /= std

# Plotting the distribution after standardization
test_x.plot.hist(bins = 50, color='b', legend=None)
plt.title("Test Data Standardized Distribution")
plt.show()
# Clearing memory from last evaluation to avoid overfitting on test data
K.clear_session()

num_epochs = 40

# Model instantiation
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(train_x.shape[1], )))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

class_weights = {0: 1.,
                1: 15.}

history = model.fit(train_x, train_y, epochs = num_epochs, class_weight = class_weights, batch_size = 64, validation_split=0.15)

# Variables for plotting performance
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, num_epochs + 1)

plt.plot(epochs, acc, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc, 'bo', label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.show()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'bo', label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.show()
K.clear_session()
model.fit(train_x, train_y, epochs = 80, class_weight = class_weights, batch_size = 64)
results = model.evaluate(test_x, test_y)
print(f"--------------SIMPLE MODEL--------------\nLOSS: {results[0]:.2f}\nACCURACY: {results[1]*100:.2f}%")
# Clearing memory from last evaluation to avoid overfitting on test data
K.clear_session()

k = 4
num_val_samples = len(train_x) // k
num_epochs = 50
all_val_acc_history = []
acc_history = []

for i in range(k):
    print(f"Processing fold #: {i}")
    print(f"RANGE: [{i*num_val_samples}, {(i+1)*num_val_samples}]")
    val_data = train_x[i*num_val_samples : (i+1)*num_val_samples]
    val_targets = train_y[i*num_val_samples : (i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_x[:i*num_val_samples], train_x[(i+1)*num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_y[:i*num_val_samples], train_y[(i+1)*num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=32, verbose=0, validation_data=(val_data, val_targets))
    all_val_acc_history.append(history.history['val_acc'])
    acc_history.append(history.history['acc'])

avg_val_acc = [np.mean([x[i] for x in all_val_acc_history]) for i in range(num_epochs)]
avg_acc = [np.mean([x[i] for x in acc_history]) for i in range(num_epochs)]

smooth_val_acc = smooth_curve(avg_val_acc)
smooth_acc = smooth_curve(avg_acc)
plt.plot(range(1, len(smooth_val_acc) +1), smooth_val_acc, label='Validation Accuracy')   
plt.plot(range(1, len(smooth_acc) +1), smooth_acc, label='Training Accuracy')  
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()
K.clear_session()

model = build_model()
history = model.fit(train_x, train_y, epochs=num_epochs, batch_size=64)
results = model.evaluate(test_x, test_y)
print(f"--------------K-FOLD MODEL--------------\nLOSS: {results[0]:.2f}\nACCURACY: {results[1]*100:.2f}%")
