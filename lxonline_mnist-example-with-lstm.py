%matplotlib inline
from keras.datasets import mnist # For Loading the MNIST data
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 9]
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential # To build a sequential model
from keras.layers import Dense, LSTM, CuDNNLSTM, Dropout, BatchNormalization # For the layers in the model
from keras.callbacks import EarlyStopping, TensorBoard # For our early stopping and tensorboard callbacks
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import time
# Get MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
classes = np.unique(y_train)
classes_str = [str(x) for x in classes]
nb_classes = len(classes)

# Normalise our data
x_train = x_train / 255
x_test = x_test / 255

# Sample the training data
#x_train, x_del, y_train, y_del = train_test_split(x_train, y_train, test_size=0.95, random_state=42)
# Apply corrections (to test data only)
corrections = {}

for (index, new_value) in corrections.items():
    y_test[index] = new_value
# Function to display a single digit
def plot_single(label, pixels):
    plt.figure(figsize=(5,3))
    plt.title('Label is {label}'.format(label=label))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
def display_single(x, y, index):
    plot_single(y[index], x[index])
    
display_single(x_train, y_train, 7)
x_train.shape[1:]
es_callback = EarlyStopping(patience = 2)

input_shape = x_train.shape[1:]

model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(input_shape), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.1))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(lr=0.001, amsgrad=False),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=128, epochs=50, callbacks=[es_callback])
print(f"Validation Accuracy: {history.history['val_acc'][-1]}")

# Validation Accuracy: 0.9835 (256 in first LSTM layer, 32 in Dense layer)
# Validation Accuracy: 0.9865 (256 in first LSTM layer, 64 in Dense layer)
# Validation Accuracy: 0.985 (256 in first LSTM layer, 256 in 2nd LSTM layer, 64 in Dense layer)
# Validation Accuracy: 0.9869 (256 in first LSTM layer, Dropout of 2nd LSTM set to 0.1, 64 in Dense layer, Patience = 2)
# Validation Accuracy: 0.9858 (as above but with BatchNorm after each layer)
predictions = model.predict_classes(x_test)

incorrect_indexes = np.where(predictions != y_test)[0]
print(f"There are {len(incorrect_indexes)} incorrect guesses out of {len(y_test)} images")
# Function to display a single digit (with guess)
def plot_single(label, predicted, pixels, index):
    plt.title(f"Index: {index}\nPredicted value: {predicted}\nActual value: {label}")
    plt.imshow(pixels, cmap='gray')
    plt.axis('off')
    
def display_single_predicted(x, y, predictions, index):
    plot_single(y[index], predictions[index], x[index], index)
    
grid_size = 4
incorrect_sample = np.random.choice(incorrect_indexes, grid_size**2, replace=False)
for i in range(1, grid_size**2 + 1):
    plt.subplot(grid_size, grid_size, i)
    display_single_predicted(x_test, y_test, predictions, incorrect_sample[i-1])
    
plt.tight_layout()
plt.show()
# Compute confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot normalized confusion matrix
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cm[cm < 0.001] = np.nan
np.set_printoptions(precision=2)
plt.figure(figsize=(10,10))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.RdYlGn)
#plt.clim(0.0001, 1);
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(nb_classes)
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

print(classification_report(y_test, predictions, target_names=classes_str))
