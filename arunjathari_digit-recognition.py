import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



from keras.utils.np_utils import to_categorical # Label encoding



from sklearn.model_selection import train_test_split # Split Data



# Modelling

from tensorflow import keras

from keras import layers



from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



from keras import models



from sklearn.metrics import confusion_matrix
X_train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

X_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')



print('Shape of the training data: ', X_train.shape)

print('Shape of the test data: ', X_test.shape)
f = plt.figure(figsize=(10,6))

counts = X_train['label'].value_counts().sort_index()

sns.barplot(counts.index, counts.values)



for i in counts.index:

    plt.text(i,counts.values[i]+50,str(counts.values[i]),horizontalalignment='center',fontsize=14)





plt.tick_params(labelsize = 14)

plt.xticks(counts.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Frequency Graph training set",fontsize=20)



plt.show()
y_train = X_train['label']

X_train.drop(labels = ['label'], axis=1, inplace=True)
print('Null values in training data: ',X_train.isna().any().sum())

print('Null values in test data: ',X_test.isna().any().sum())



X_train = X_train / 255.0

X_test = X_test / 255.0
X_train = X_train.values.reshape(-1, 28, 28, 1)

X_test = X_test.values.reshape(-1, 28, 28, 1)
y_train = to_categorical(y_train, num_classes=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1)
model = keras.models.Sequential([layers.Conv2D(32, (3,3), activation="relu",padding='same', input_shape=(28,28,1)),

                                 layers.MaxPooling2D(2,2),

                                 layers.Dropout(0.25),

                                  layers.Conv2D(64, (3,3), activation="relu",padding='same'),

                                 layers.MaxPooling2D(2,2),

                                 layers.Dropout(0.25),

                                 layers.Flatten(),

                                  layers.BatchNormalization(),

                                 layers.Dense(128, activation="elu", kernel_initializer="he_normal"),

                                  layers.BatchNormalization(),

                                 layers.Dense(32, activation="elu", kernel_initializer="he_normal"),

                                  layers.BatchNormalization(),

                                 layers.Dense(10, activation="softmax")])

model.summary()
optimizer = Adam(learning_rate=0.001, epsilon=1e-07)



model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics=['accuracy'])



earlyStopping = EarlyStopping(monitor='val_accuracy', patience=10, verbose=0, mode='auto')

mcp = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_accuracy', mode='auto')

reduce_lr_loss = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='auto')
datagen = ImageDataGenerator(

    rotation_range=5,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=5,

    zoom_range=0.1)



datagen.fit(X_train)
Batch_size=100

Epochs = 100

history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=Batch_size),

                              epochs = Epochs, 

                              validation_data = (X_val,y_val),

                              verbose = 2, 

                              steps_per_epoch=X_train.shape[0]//Batch_size, 

                             callbacks = [earlyStopping, mcp, reduce_lr_loss])
f = plt.figure(figsize=(20,7))



f.add_subplot(121)

# plt.figure(figsize=(6, 4))

plt.plot(history.history['accuracy'], color='b', label="Train")

plt.plot(history.history['val_accuracy'], color='r',label="Validation")

plt.legend(loc='best',fontsize=18)

plt.title('Accuracy Curve',fontsize=25)







f.add_subplot(122)

# plt.figure(figsize=(6, 4))

plt.plot(history.history['loss'], color='b', label="Train")

plt.plot(history.history['val_loss'], color='r', label="validation")

plt.legend(loc='best',fontsize=18)

plt.title('Loss Curve',fontsize=25)

plt.show()
plt.figure(figsize=(10,8.5))

y_pred = model.predict(X_val)

y_pred = np.argmax(y_pred, axis=1)

y_val_test = np.argmax(y_val, axis=1)

cm = confusion_matrix(y_val_test, y_pred)

sns.heatmap(cm, annot=True, fmt='g', cbar=False)

plt.xlabel('Predicted')

plt.ylabel('True')

plt.title('Confusion Matrix for predicted and true labels')

plt.show()
model.load_weights(filepath = '.mdl_wts.hdf5')



scores = model.evaluate(X_val, y_val, callbacks = [earlyStopping, mcp, reduce_lr_loss])
cnt_error = []

for (a, b) in zip(y_val_test, y_pred):

    if a == b: continue

    cnt_error.append( a )



    

cnt_error = np.unique(cnt_error, return_counts = True)

cnt_error = pd.Series(cnt_error[1],index=cnt_error[0])

f = plt.figure(figsize=(10,6))

sns.barplot(cnt_error.index, cnt_error.values)



# for i in counts.index:

#     plt.text(i,cnt_error.values[i]+0.2,str(cnt_error.values[i]),horizontalalignment='center',fontsize=14)





plt.tick_params(labelsize = 14)

plt.xticks(cnt_error.index)

plt.xlabel("Digits",fontsize=16)

plt.ylabel("Frequency",fontsize=16)

plt.title("Worng Predictions",fontsize=20)



plt.show()
rows = 3

cols = 4



plt.figure(figsize=(10,7))

for i in range(rows*cols):

    ax = plt.subplot(rows, cols, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(X_val[i].reshape(28, 28), cmap='Blues')

    plt.title('Predicted: {}   True: {}'.format(y_pred[i], y_val_test[i])

              ,y=-0.15)
ls = np.array(y_pred - y_val_test) # Subtract true values from predicted --> All non-zero results were incorrectly predicted



nonzero_pred = np.nonzero(ls)[0]



rows = 3

cols = 4



plt.figure(figsize=(10,7))

for i in range(rows*cols):

    ax = plt.subplot(rows, cols, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(X_val[nonzero_pred[i]].reshape(28, 28), cmap='Reds')

    plt.title('Predicted: {}  True: {}'.format(y_pred[nonzero_pred[i]], y_val_test[nonzero_pred[i]]), y=-0.15)
test = X_test[0:30,:,:,:]

test_pred = np.argmax(model.predict(X_test), axis=1)

rows = 3

cols = 4

plt.figure(figsize=(10,7))

for i in range(rows*cols):

    ax = plt.subplot(rows,cols, i+1)

    plt.axis('off')

    ax.set_xticks([])

    ax.set_yticks([])

    plt.imshow(X_test[i].reshape(28, 28), cmap='bone')

    plt.title('Predicted: {}'.format(test_pred[i]),y=-0.15)
for layer in model.layers:

    if'conv' in layer.name:

        filters, biases = layer.get_weights()

        print('Layer: ', layer.name, filters.shape)

#         f_min, f_max = filters.min(), filters.max()

#         filters = (filters - f_min) / (f_max - f_min)

        

        print('Filter size: (', filters.shape[0], ',', filters.shape[1], ')')

        print('Channels in this layer: ', filters.shape[2])

        print('Number of filters: ', filters.shape[3])

        

        count = 1

        plt.figure(figsize = (18, 4))

        

        for i in range(filters.shape[3]):

            ax= plt.subplot(4, filters.shape[3]/4, count)

            ax.set_xticks([])

            ax.set_yticks([])

            plt.imshow(filters[:,:,0, i], cmap=plt.cm.binary)

            count+=1

            

        plt.show()
print('total number of layers',len(model.layers))
# Extract outputs of top 6 layers

layer_outputs = [layer.output for layer in model.layers[0:6]]



# Create a model that will return these outputs given the model input

activation_model = models.Model(inputs = model.input, outputs = layer_outputs)
plt.imshow(X_test[4].reshape(28, 28), cmap = plt.cm.binary)

plt.xticks([])

plt.yticks([])

plt.show()
img_tensor = X_test[4].reshape(-1, 28, 28, 1)

activations = activation_model.predict(img_tensor)
for i in range(len(activations)):

    print(activations[i].shape)
first_layer_activation = activations[0]



ax = plt.subplot(1, 2, 1)

ax.set_xticks([])

ax.set_yticks([])

plt.imshow(first_layer_activation[0,:,:,0], cmap = plt.cm.binary)

plt.title('1st Filter o/p')



ax = plt.subplot(1, 2, 2)

ax.set_xticks([])

ax.set_yticks([])

plt.imshow(first_layer_activation[0,:,:,16], cmap = plt.cm.binary)

plt.title('16th Filter o/p')

    

plt.show()
layer_names = []

for layer in model.layers[:6]:

    layer_names.append(layer.name)



for activation_layer, layer_name in zip(activations, layer_names):



    n_features = activation_layer.shape[3]

    feat_per_row = 16

    rows = n_features//feat_per_row

    size = activation_layer.shape[1]

    

    print(layer_name)

    plt.figure(figsize=(20, rows))

    for i in range(n_features):

        ax = plt.subplot(rows, feat_per_row, i+1)

        ax.set_xticks([])

        ax.set_yticks([])

        plt.imshow(activation_layer[:,:,:,i].reshape(size, size), cmap = plt.cm.binary)

    

    plt.show()
submissions = pd.DataFrame({"ImageId": list(range(1,len(test_pred)+1)),

    "Label": test_pred})

submissions.to_csv("submission.csv", index=False, header=True)