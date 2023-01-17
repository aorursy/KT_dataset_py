import warnings

import pandas as p

import numpy as np

import keras as k

import matplotlib.pyplot as plot

import seaborn as sns

import pydot



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split



%matplotlib inline
original_train_data = p.read_csv('train.csv')

original_test_data = p.read_csv('test.csv')
def shape_for_display(ds):

    _working_ds = ds

    

    if 'label' in ds.columns:

        _working_ds = ds.drop(['label'], axis=1)        

        

    return np.array(_working_ds).reshape(-1, 28, 28)



def shape_for_cnn(ds):

    _working_ds = ds

    

    if 'label' in ds.columns:

        _working_ds = ds.drop(['label'], axis=1)

        

    _working_ds = MinMaxScaler().fit_transform(_working_ds)

        

    return np.array(_working_ds).reshape(-1, 28, 28, 1)



def extract_label(ds):

    retval = None

    

    if 'label' in ds.columns:

        retval = k.utils.to_categorical(ds['label'])

        

    return retval

small_fig_size = (5, 5)



def show_some_numbers(train_ds, test_ds):

    plot.figure(figsize=small_fig_size)

    plot.subplot(2, 2, 1)

    plot.imshow(train_ds[4056], cmap=plot.cm.binary)

    plot.subplot(2, 2, 2)

    plot.imshow(test_ds[4056], cmap=plot.cm.binary)

    plot.show()

    

def plot_label_counts(ds):

    plot.figure(figsize=small_fig_size)

    sns.countplot(x='label', data=ds)

    plot.show()

    

def plot_normalized(idx, ds, labels):

    plot.figure(figsize=small_fig_size)

    plot.imshow(ds[idx][:,:,0])

    plot.title(labels[idx].argmax()) 

    plot.show()

    

def plot_loss(history):

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    

    plot.figure(figsize=small_fig_size)

    plot.plot(epochs, loss, color='red', label='Training loss')

    plot.plot(epochs, val_loss, color='green', label='Validation loss')

    plot.title('Training and validation loss')

    plot.xlabel('Epochs')

    plot.ylabel('Loss')

    plot.legend()

    plot.show()

    

def plot_acc(history):

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    epochs = range(1, len(loss) + 1)    

    

    plot.figure(figsize=small_fig_size)

    plot.plot(epochs, acc, color='red', label='Training acc')

    plot.plot(epochs, val_acc, color='green', label='Validation acc')

    plot.title('Training and validation accuracy')

    plot.xlabel('Epochs')

    plot.ylabel('Loss')

    plot.legend()

    plot.show()

    

def display_activation(activations, col_size, row_size, act_index): 

    activation = activations[act_index]

    activation_index = 0

    

    plot.figure(figsize=(12,30))

    

    fig, ax = plot.subplots(row_size, col_size, figsize=(row_size * 1.5, col_size * 1.5))

    for row in range(0, row_size):

        for col in range(0, col_size):

            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')

            activation_index += 1



    plot.show()

    
show_some_numbers(shape_for_display(original_train_data), shape_for_display(original_test_data))
train_cnn = shape_for_cnn(original_train_data)

test_cnn = shape_for_cnn(original_test_data)
labels = extract_label(original_train_data)
def really_complicated():

    model = k.models.Sequential()

    

    model.add(k.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(k.layers.Dropout(0.5))

    model.add(k.layers.MaxPooling2D((2, 2)))

    model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(k.layers.Dropout(0.5))

    model.add(k.layers.MaxPooling2D((2, 2)))

    model.add(k.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(k.layers.Dropout(0.5))

    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(64, activation='relu'))

    model.add(k.layers.Dense(10, activation='softmax'))

    

    model.compile('rmsprop', 'categorical_crossentropy', ['accuracy'])

   

    return model



def another_really_complicated(optimizer='sgd'):

    model = k.models.Sequential()

    

    model.add(k.layers.Conv2D(64,(3,3), input_shape=(28, 28, 1), strides = (1,1), padding='same'))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.MaxPooling2D((2,2)))

    

    model.add(k.layers.Conv2D(64, (3,3), strides = (1,1), padding='same'))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.MaxPooling2D((2,2)))

    

    model.add(k.layers.Conv2D(64, (3,3), strides = (1,1), padding='same'))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.MaxPooling2D((2,2)))

    

    model.add(k.layers.Flatten())

    

    model.add(k.layers.Dense(64))

    model.add(k.layers.Activation('relu'))

    

    model.add(k.layers.Dropout(0.25))

    

    model.add(k.layers.Dense(32))

    model.add(k.layers.Activation('relu'))

    

    model.add(k.layers.Dropout(0.25))

    

    model.add(k.layers.Dense(10))

    model.add(k.layers.Activation('softmax'))

    

    if optimizer == 'sgd':   

        optim = k.optimizers.SGD(lr=0.01, momentum=.5, decay=0.0, nesterov=False)

        print('Using SGD optimizer')

    else:

        print('Using ADAM optimizer')        

        optim = k.optimizers.Adam(lr=0.01)

    

    model.compile(optim, 'categorical_crossentropy', ['accuracy'])

    

    return model



def really_basic():

    model = k.models.Sequential()



    model.add(k.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1)))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.AveragePooling2D((2,2)))

    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(10, activation='softmax'))  



    model.compile('adam', 'categorical_crossentropy', ['accuracy'])

    

def basic_with_sgd():

    model = k.models.Sequential()



    model.add(k.layers.Conv2D(64, (3, 3), input_shape=(28, 28, 1)))

    model.add(k.layers.BatchNormalization())

    model.add(k.layers.Activation('relu'))

    model.add(k.layers.AveragePooling2D((2,2)))

    model.add(k.layers.Flatten())

    model.add(k.layers.Dense(10))

    model.add(k.layers.Activation('softmax'))    

    

    sgd = k.optimizers.SGD(lr=0.01, momentum=.5, decay=0.0, nesterov=False)



    model.compile(sgd, 'categorical_crossentropy', ['accuracy'])

    

    return model



def get_generic_activation_model(model):

    layer_outputs = [layer.output for layer in model.layers]

    return k.models.Model(inputs=model.input, outputs=layer_outputs)    
def step_decay(epoch):

    initial_lrate = 0.1

    drop = 0.6

    epochs_drop = 3.0

    lrate = initial_lrate * np.power(drop, np.floor((1 + epoch) / epochs_drop))

    

    return lrate
checkpoint = k.callbacks.ModelCheckpoint('checkpoints')

learn_rate_sched = k.callbacks.LearningRateScheduler(step_decay)



X_train, X_test, y_train, y_test = train_test_split(train_cnn, labels)
model = another_really_complicated()
%%time



hist = model.fit(X_train, y_train, epochs=10, callbacks=[checkpoint, learn_rate_sched], validation_data=(X_test, y_test))
plot_loss(hist)
plot_acc(hist)
evaluation = model.evaluate(X_test, y_test)



print("Acc\t\t\t", str(evaluation[1]*100))

print("Total loss\t\t",str(evaluation[0]*100))
# Use this if you want to print out the activation layers.

act_model = get_generic_activation_model(model)

simple_pred = act_model.predict(X_train[10].reshape(1, 28, 28, 1))



display_activation(simple_pred, 8, 8, 1)
%%time



results = model.predict(test_cnn)
best_number = []



for i in range(results.shape[0]):

    best_number.append(np.argmax(results[i]))



result_df = p.DataFrame({ 'ImageId' : np.array(range(1, len(best_number) + 1)), 'Label' : np.array(best_number) })



result_df.to_csv('submission.csv', index=False)
# train_datagen = k.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255, \

#                                                          shear_range=0.2, zoom_range=0.2, \

#                                                          horizontal_flip=True)



# dg_flow = train_datagen.flow(X_train, y_train, batch_size=16)



# hist = model.fit_generator(dg_flow, epochs=10, steps_per_epoch=X_train.shape[0], \

#                            callbacks=[checkpoint, learn_rate_sched], \

#                            validation_data=(X_test, y_test))