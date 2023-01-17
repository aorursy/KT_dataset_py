import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from keras.models import Sequential, load_model

from keras.layers import Conv2D, Dense, Flatten, MaxPool2D, Dropout

from keras.optimizers import Adam

from keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint

from keras.wrappers.scikit_learn import KerasClassifier          # Import sklearn wrapper from keras

from sklearn.model_selection import RandomizedSearchCV           # Hyperparameter tuning
# Load Data

train = pd.read_csv('../input/digit-recognizer/train.csv')

print(train.head())



test = pd.read_csv('../input/digit-recognizer/test.csv')

print(test.head())
# Slice the train data into X_train & y_train

X_train = train.drop(labels = ['label'], axis = 1)

y_train = train.label



# Clear memory space by deleting train data

del train
# Check for missing values

X_train.isnull().any().describe()
test.isnull().any().describe()
# Check data distribution across labels

y_train.value_counts()

sns.countplot(y_train)
# Perform grayscale normalization

X_train = X_train / 255.0

test = test / 255.0
# Reshape image in 3 dimentions (28, 28, 1)

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)

print(X_train.shape, test.shape)
# Encoding Labels

y_train = to_categorical(y_train, num_classes = 10)
# Split train and validation set for fitting

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,

                                                  test_size = 0.1,

                                                  random_state = 10,

                                                  stratify = y_train)
# Visualize the few images for better sense

fig, ax = plt.subplots(nrows = 1, ncols = 3)

for i in range(3):

    ax[i].imshow(X_train[i][:, :, 0], cmap = 'gray')

    ax[i].axis('off')

plt.show()
# Keras callback variable

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose = 0)



# Build CNN

nets = 3

model = [0] * nets



for i in range(nets):

    model[i] = Sequential()

    model[i].add(Conv2D(filters = 32, kernel_size = 5, padding = 'same', 

                        activation = 'relu',

                        input_shape = (28, 28, 1)))

    model[i].add(MaxPool2D())

    if i > 0:

        model[i].add(Conv2D(filters = 48, kernel_size = 5, padding = 'same', 

                            activation = 'relu'))

        model[i].add(MaxPool2D())

    if i > 1:

        model[i].add(Conv2D(filters = 64, kernel_size = 5, padding = 'same', 

                            activation = 'relu'))

        model[i].add(MaxPool2D(padding = 'same'))

    model[i].add(Flatten())

    model[i].add(Dense(256, activation = 'relu'))

    model[i].add(Dense(10, activation = 'softmax'))

    model[i].compile(optimizer = 'adam', loss = 'categorical_crossentropy', 

                     metrics = ['accuracy'])



# Train network

history = [0] * nets

names = ['CNN-1', 'CNN-2', 'CNN-3']

epochs = 20



for i in range(nets):

    history[i] = model[i].fit(X_train, y_train, batch_size = 100, 

                              epochs = epochs,

                              validation_data = (X_val, y_val), 

                              callbacks = [annealer], verbose = 0)

    print('{0}: Epochs = {1: d}, Train accuracy = {2: .5f}, Validation accuracy = {3: .5f}'.format(names[i],

           epochs, max(history[i].history['accuracy']), 

           max(history[i].history['val_accuracy'])))

# Plot accuracy

for i in range(nets):

    plt.plot(history[i].history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.legend(names, loc = 'upper left')

plt.show()
# Build CNN

nets = 7

model = [0] * nets



for i in range(nets):

    model[i] = Sequential()

    model[i].add(Conv2D(i*8+8, kernel_size = 5, 

                        padding = 'same', 

                        activation = 'relu',

                        input_shape = (28, 28, 1)))

    model[i].add(MaxPool2D())

    model[i].add(Conv2D(i*16+16, kernel_size = 5, 

                        padding = 'same', 

                        activation = 'relu'))

    model[i].add(MaxPool2D())

    model[i].add(Flatten())

    model[i].add(Dense(256, activation = 'relu'))

    model[i].add(Dense(10, activation = 'softmax'))

    model[i].compile(optimizer = 'adam', 

                     loss = 'categorical_crossentropy',

                     metrics = ['accuracy'])



# Train network

history = [0] * nets



names = []

for i in range(nets):

    names.append(i*8+8)



for i in range(nets):

    history[i] = model[i].fit(X_train, y_train, 

                              batch_size = 100, 

                              epochs = epochs,

                              validation_data = (X_val, y_val), 

                              callbacks = [annealer], verbose = 0)

    print('CNN {0: d} maps: Epochs = {1: d}, Train accuracy = {2: .5f}, Validation accuracy = {3: .5f}'.format(names[i],

                epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))
# Plot accuracy

for i in range(nets):

    plt.plot(history[i].history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.legend(names, loc = 'upper left')

plt.show()
# Build CNN

nets = 8

model = [0] * nets



for i in range(nets):

    model[i] = Sequential()

    model[i].add(Conv2D(48, kernel_size = 5,

                        padding = 'same',

                        activation = 'relu',

                        input_shape = (28, 28, 1)))

    model[i].add(MaxPool2D())

    model[i].add(Conv2D(96, kernel_size = 5,

                        activation = 'relu'))

    model[i].add(MaxPool2D())

    model[i].add(Flatten())

    if i>0:

        model[i].add(Dense(2**(i+4), activation = 'relu'))

    model[i].add(Dense(10, activation = 'softmax'))

    model[i].compile(optimizer = 'adam',

                     loss = 'categorical_crossentropy',

                     metrics = ['accuracy'])



# Train network

history = [0] * nets



names = [0]

for i in range(nets - 1):

    names.append(2**(i+5))



for i in range(nets):

    history[i] = model[i].fit(X_train, y_train,

                              batch_size = 100,

                              epochs = epochs,

                              validation_data = (X_val, y_val),

                              callbacks = [annealer], verbose = 0)

    print('CNN {0: d}N: Epochs {1: d}, Training accuracy {2: .5f}, Validation accuracy {3: .5f}'.format(names[i],

              epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))
# Plot accuracy

for i in range(nets):

    plt.plot(history[i].history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.legend(names, loc = 'upper left')

plt.show()
# Build CNN

nets = 8

model = [0] * nets



names = []

for i, n in enumerate(range(8)):

    names.append(f'{n*10}%')



for i in range(nets):

    model[i] = Sequential()

    model[i].add(Conv2D(48, kernel_size = 5,

                        padding = 'same',

                        activation = 'relu',

                        input_shape = (28, 28, 1)))

    model[i].add(MaxPool2D())

    model[i].add(Dropout(i*0.1))

    model[i].add(Conv2D(96, kernel_size = 5,

                        activation = 'relu'))

    model[i].add(MaxPool2D())

    model[i].add(Dropout(i*0.1))

    model[i].add(Flatten())

    model[i].add(Dense(256, activation = 'relu'))

    model[i].add(Dropout(i*0.1))

    model[i].add(Dense(10, activation = 'sigmoid'))

    model[i].compile(optimizer = 'adam', 

                     loss = 'categorical_crossentropy',

                     metrics = ['accuracy'])



# Train network

history = [0] * nets



for i in range(nets):

    history[i] = model[i].fit(X_train, y_train,

                              batch_size = 100,

                              epochs = epochs,

                              validation_data = (X_val, y_val),

                              callbacks = [annealer],

                              verbose = 0)

    print('CNN Dropouts = {0}: Epochs = {1: d}, Training accuracy = {2: .5f}, Validation accuracy = {3: .5f}'.format(names[i],

              epochs, max(history[i].history['accuracy']), max(history[i].history['val_accuracy'])))
# Plot accuracy

for i in range(nets):

    plt.plot(history[i].history['val_accuracy'])

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.title('Model Accuracy')

plt.legend(names, loc = 'upper left')

plt.show()
# Function that create our CNN model

def create_model(optimizer = 'adam', activation = 'relu'):

    model = Sequential()

    model.add(Conv2D(48, kernel_size = 5,

                     padding = 'same',

                     activation = activation,

                     input_shape = (28, 28, 1)))

    model.add(MaxPool2D())

    model.add(Dropout(0.4))

    model.add(Conv2D(96, kernel_size = 5,

                     activation = activation))

    model.add(MaxPool2D())

    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(256, activation = activation))

    model.add(Dropout(0.4))

    model.add(Dense(10, activation = 'sigmoid'))

    model.compile(optimizer = optimizer, 

                  loss = 'categorical_crossentropy',

                  metrics = ['accuracy'])

    return model



# Create a model as a sklearn estimator

model = KerasClassifier(build_fn = create_model)



# Define a series of parameters

params = dict(optimizer = ['sgd', 'adam'], activation = ['relu', 'tanh'],

              batch_size = [50, 100, 150, 200], epochs = [10, 20, 30, 50])



# Create a random search cv object and fit it to data

random_search = RandomizedSearchCV(model, param_distributions = params, cv = 3,

                                   n_iter = 10)

random_search_results = random_search.fit(X_train, y_train, 

                                          validation_data = (X_val, y_val))
# Print results

print('Best: {0: 5f} using {1}'.format(random_search_results.best_score_,

                                       random_search_results.best_params_))
# Callbacks

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5, 

                               verbose = 0,

                               restore_best_weights = True)

annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x, verbose = 0)

model_checkpoint = ModelCheckpoint('Digit_Recognizer.hdf5', monitor='val_accuracy', 

                                   verbose=1, save_best_only=True, 

                                   mode='max')



# Build CNN

cnn = create_model(optimizer = 'adam', activation = 'relu')



# Train Network

results = cnn.fit(X_train, y_train, validation_data = (X_val, y_val),

                    batch_size = 50, epochs = 30,

                    callbacks = [early_stopping, model_checkpoint, annealer])
# Accuracy Curve

plt.plot(results.history['accuracy'], 'o-', label = 'Accuracy')

plt.plot(results.history['val_accuracy'], 'o-', label = 'Val Accuracy')

plt.title('Accuracy')

plt.legend(loc = 'best')
# Accuracy Curve

plt.plot(results.history['loss'], 'o-', label = 'Loss')

plt.plot(results.history['val_loss'], 'o-', label = 'Val Loss')

plt.title('Accuracy')

plt.legend(loc = 'best')
# Load best model

model = load_model('Digit_Recognizer.hdf5')

model.evaluate(X_val, y_val)



# Making predictions

pred = model.predict(test)

pred = np.argmax(pred, axis=1)



# Making Submission file

pred = pd.Series(pred,name="Label")

result = pd.concat([pd.Series(range(1,28001),name = "ImageId"),pred],axis = 1)



result.to_csv("./digit_recognizer.csv",index=False)