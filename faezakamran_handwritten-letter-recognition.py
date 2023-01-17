import numpy as np 

import pandas as pd 

import warnings



warnings.filterwarnings('ignore', category=FutureWarning)



data1 = pd.read_csv("../input/letters.csv")

files1 = data1['file']

letters1 = data1['letter']

backgrounds1 = data1['background']

#data1.head()



data2 = pd.read_csv("../input/letters2.csv")

files2 = data2['file']

letters2 = data2['letter']

backgrounds2 = data2['background']

#data2.head()



data3 = pd.read_csv("../input/letters3.csv")

files3 = data3['file']

letters3 = data3['letter']

backgrounds3 = data3['background']

data3.head()

# Any results you write to the current directory are saved as output.
import h5py



# Read the h5 file

f = h5py.File('../input/LetterColorImages_123.h5', 'r')

# List all groups

keys = list(f.keys())

keys 
# Create tensors and targets

backgrounds = np.array(f[keys[0]])

tensors = np.array(f[keys[1]])

targets = np.array(f[keys[2]])

print ('Tensor shape:', tensors.shape)

print ('Target shape', targets.shape)

print ('Background shape:', backgrounds.shape)
# Concatenate series

letters = pd.concat((letters1, letters2), axis=0, ignore_index=True)

letters = pd.concat((letters, letters3), axis=0, ignore_index=True)

len(letters)
# Normalize the tensors

tensors = tensors.astype('float32')/255
import matplotlib.pylab as plt

from matplotlib import cm

%matplotlib inline



# Read and display a tensor using Matplotlib

print('Label: ', letters[10])

plt.figure(figsize=(3,3))

plt.imshow(tensors[10]);
# Print the target unique values

print(set(targets))
from keras.utils import to_categorical



# One-hot encoding the targets, started from the zero label

cat_targets = to_categorical(np.array(targets-1), 33)

cat_targets.shape
from sklearn.model_selection import train_test_split



# Split the data

x_train, x_test, y_train, y_test = train_test_split(tensors, cat_targets, 

                                                    test_size = 0.2, 

                                                    random_state = 1)

n = int(len(x_test)/2)

x_valid, y_valid = x_test[:n], y_test[:n]

x_test, y_test = x_test[n:], y_test[n:]



# Print the shape

x_train.shape, y_train.shape, x_valid.shape, y_valid.shape, x_test.shape, y_test.shape
from keras.preprocessing import image as keras_image

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from keras.metrics import top_k_categorical_accuracy, categorical_accuracy

from keras.models import Sequential, load_model

from keras.layers import Dense, LSTM, GlobalAveragePooling1D, GlobalAveragePooling2D

from keras.layers.advanced_activations import PReLU, LeakyReLU

from keras.layers import Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D



def top_3_categorical_accuracy(y_true, y_pred):

    return top_k_categorical_accuracy(y_true, y_pred, k=3)



model = Sequential()



# Define a model architecture    

model.add(Conv2D(32, (5, 5), padding='same', input_shape=x_train.shape[1:]))

model.add(LeakyReLU(alpha=0.02))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(Conv2D(196, (5, 5)))

model.add(LeakyReLU(alpha=0.02))



model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.2))



model.add(GlobalMaxPooling2D())



model.add(Dense(1024))

model.add(LeakyReLU(alpha=0.02))

model.add(Dropout(0.5)) 



model.add(Dense(33))

model.add(Activation('softmax'))



# Compile the model

model.compile(loss='categorical_crossentropy', optimizer='adam', 

              metrics=[categorical_accuracy, top_3_categorical_accuracy])

# Create callbacks

checkpointer = ModelCheckpoint(filepath='weights.best.model.hdf5', 

                               verbose=2, save_best_only=True)

lr_reduction = ReduceLROnPlateau(monitor='val_loss', 

                                 patience=5, verbose=2, factor=0.75)

# Train the model

history = model.fit(x_train, y_train, 

                    epochs=50, batch_size=512, verbose=2,

                    validation_data=(x_valid, y_valid),

                    callbacks=[checkpointer, lr_reduction])
# Plot the Neural network fitting history

def history_plot(fit_history, n):

    plt.figure(figsize=(18, 12))

    

    plt.subplot(211)

    plt.plot(fit_history.history['loss'][n:], color='slategray', label = 'train')

    plt.plot(fit_history.history['val_loss'][n:], color='#4876ff', label = 'valid')

    plt.xlabel("Epochs")

    plt.ylabel("Loss")

    plt.legend()

    plt.title('Loss Function');  

    

    plt.subplot(212)

    plt.plot(fit_history.history['categorical_accuracy'][n:], color='slategray', label = 'train')

    plt.plot(fit_history.history['val_categorical_accuracy'][n:], color='#4876ff', label = 'valid')

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")    

    plt.legend()

    plt.title('Accuracy');



# Plot the training history

history_plot(history, 0)
# Load the model with the best validation accuracy

model.load_weights('weights.best.model.hdf5')

# Calculate classification accuracy on the testing set

score = model.evaluate(x_test, y_test)

score
# Create a list of symbols

symbols = ['а','б','в','г','д','е','ё','ж','з','и','й',

           'к','л','м','н','о','п','р','с','т','у','ф',

           'х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']
# Model predictions for the testing dataset

y_test_predict = model.predict_classes(x_test)
# Display true labels and predictions

fig = plt.figure(figsize=(14, 14))

for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):

    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])

    ax.imshow(np.squeeze(x_test[idx]))

    pred_idx = y_test_predict[idx]

    true_idx = np.argmax(y_test[idx])

    ax.set_title("{} ({})".format(symbols[pred_idx], symbols[true_idx]),

                 color=("#4876ff" if pred_idx == true_idx else "darkred"))