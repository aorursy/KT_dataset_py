import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
typeList = [

        'outlyingScatterPlot',

        'skewedScatterPlot',

        'clumpyScatterPlot',

        'sparsedScatterPlot',

        'striatedScatterPlot',

        'convexScatterPlot',

        'skinnyScatterPlot',

        'stringyScatterPlot',

        'monotonicScatterPlot']
from matplotlib import pyplot as plt
import json

# with open('/kaggle/input/typicalandrealwordscags/ScagnosticsTypicalData.json') as f:

#     typicalData = json.load(f)

# with open('/kaggle/input/typicalandrealwordscags/ScagnosticsTypicalData1.json') as f:

#     typicalData1 = json.load(f)    

# with open('/kaggle/input/typicalandrealwordscags/ScagnosticsTypicalData2.json') as f:

#     typicalData2 = json.load(f)    

# with open('/kaggle/input/typicalandrealwordscags/RealWorldData.json') as f:

#     realWorldData = json.load(f)

# with open('/kaggle/input/typicalandrealwordscags/RealWorldData10.json') as f:

#     realWorldData10 = json.load(f)
# def plotSampleDataOfType(df, numPlots, scagType):

#     ds = np.array([d['rectangularBins'] for d in df[typeList.index(scagType)]])

#     fig, axs = plt.subplots(1, numPlots, figsize=(20, 15))

#     counter = 0

#     for i in np.random.choice(range(len(ds)), numPlots):

#         axs[counter].imshow(ds[i], cmap='hot', interpolation='nearest')

#         axs[counter].grid(False)

#         counter += 1

#     plt.show()
# plotSampleDataOfType(typicalData, 10, 'outlyingScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'skewedScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'clumpyScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'sparsedScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'striatedScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'convexScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'skinnyScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'stringyScatterPlot')
# plotSampleDataOfType(typicalData, 10, 'monotonicScatterPlot')
numPoints = 0 # minimum number of bins

# X_typical = []

# y_typical = []

# y_typical_label = []

# for ds in typicalData:

#     for d in ds:

#         # filter out invalid data

#         if not ((np.array(d['scagnostics']) > 1).any() or (np.array(d['scagnostics']) < 0).any()) and np.sum(d['rectangularBins']) >= numPoints:

#             X_typical.append(d['rectangularBins'])

#             y_typical.append(d['scagnostics'])

#             y_typical_label.append([1 if tl == d['dataSource'] else 0 for tl in typeList])



# X_typical1 = []

# y_typical1 = []

# y_typical_label1 = []

# for ds in typicalData1:

#     for d in ds:

#         # filter out invalid data

#         if not ((np.array(d['scagnostics']) > 1).any() or (np.array(d['scagnostics']) < 0).any()) and np.sum(d['rectangularBins']) >= numPoints:

#             X_typical1.append(d['rectangularBins'])

#             y_typical1.append(d['scagnostics'])

#             y_typical_label1.append([1 if tl == d['dataSource'] else 0 for tl in typeList])



# X_typical2 = []

# y_typical2 = []

# y_typical_label2 = []

# for ds in typicalData2:

#     for d in ds:

#         # filter out invalid data

#         if not ((np.array(d['scagnostics']) > 1).any() or (np.array(d['scagnostics']) < 0).any()) and np.sum(d['rectangularBins']) >= numPoints:

#             X_typical2.append(d['rectangularBins'])

#             y_typical2.append(d['scagnostics'])

#             y_typical_label2.append([1 if tl == d['dataSource'] else 0 for tl in typeList])

            

# X_real = []

# y_real = []

# for ds in realWorldData:

#     for d in ds:

#         if not ((np.array(d['scagnostics']) > 1).any() or (np.array(d['scagnostics']) < 0).any()) and np.sum(d['rectangularBins']) >= numPoints:

#             X_real.append(d['rectangularBins'])

#             y_real.append(d['scagnostics'])



# X_real10 = []

# y_real10 = []

# for ds in realWorldData10:

#     for d in ds:

#         if not ((np.array(d['scagnostics']) > 1).any() or (np.array(d['scagnostics']) < 0).any()) and np.sum(d['rectangularBins']) >= numPoints:

#             X_real10.append(d['rectangularBins'])

#             y_real10.append(d['scagnostics'])



# # convert array type.

# X_typical = np.array(X_typical)

# y_typical = np.array(y_typical)

# y_typical_label = np.array(y_typical_label)



# X_typical1 = np.array(X_typical1)

# y_typical1 = np.array(y_typical1)

# y_typical_label1 = np.array(y_typical_label1)



# X_typical2 = np.array(X_typical2)

# y_typical2 = np.array(y_typical2)

# y_typical_label2 = np.array(y_typical_label2)



# X_real = np.array(X_real)

# y_real = np.array(y_real)



# X_real10 = np.array(X_real10)

# y_real10 = np.array(y_real10)
# def plotSampleData(ds, numPlots, labels = None):

#     fig, axs = plt.subplots(1, numPlots, figsize=(20, 15))

#     counter = 0

#     for i in np.random.choice(range(len(ds)), numPlots):

#         axs[counter].imshow(ds[i], cmap='hot', interpolation='nearest')

#         axs[counter].grid(False)

#         if labels is not None:

#             axs[counter].title.set_text(typeList[np.argmax(labels[i])].replace('ScatterPlot', ''))

#         counter += 1

#     plt.show()
# numPlots = 10

# plotSampleData(X_typical, numPlots, y_typical_label)
# numPlots = 10

# plotSampleData(X_typical1, numPlots, y_typical_label)
# numPlots = 10

# plotSampleData(X_real, numPlots)
# numPlots = 10

# plotSampleData(X_real10, numPlots)
# from sklearn.model_selection import train_test_split

# X = np.array(X_typical2)

# y = np.array(y_typical_label2)

# X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# X_real = X_real.reshape(X_real.shape[0], X_real.shape[1], X_real.shape[2], 1)
# # save data

# # dump the data too.

# import codecs, json

# def exportNPArrayToJSON(a, fileName):

#     b = a.tolist() # nested lists with same data, indices

#     json.dump(b, codecs.open(fileName, 'w', encoding='utf-8')) ### this saves the array in .json format
# exportNPArrayToJSON(X_train, "X_train_cls.json")

# exportNPArrayToJSON(X_test, "X_test_cls.json")

# exportNPArrayToJSON(y_train, "y_train_cls.json")

# exportNPArrayToJSON(y_test, "y_test_cls.json")
# # X_train = X_typical.reshape(X_typical.shape[0], X_typical.shape[1], X_typical.shape[2], 1)

# # y_train = y_typical_label

# # X_test = X_typical1.reshape(X_typical1.shape[0], X_typical1.shape[1],X_typical1.shape[2], 1)

# # y_test = y_typical_label1



# # X_real = X_real.reshape(X_real.shape[0], X_real.shape[1], X_real.shape[2], 1)



# from keras.models import Sequential

# from keras.layers import Dense

# from keras.layers import Flatten

# from keras.layers import Conv2D

# from keras.layers import MaxPooling2D

# from keras.layers import Dropout

# from keras.utils import plot_model

# from keras.regularizers import l2

# from keras.optimizers import SGD



# # new classification.

# # learned several reasons from here: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

# def create_cnn_cls():

#     model = Sequential()

#     # VGG Block 1

#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(40, 40, 1)))

# #     model.add(BatchNormalization())

#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# #     model.add(BatchNormalization())

#     model.add(MaxPooling2D((2, 2)))

#     model.add(Dropout(0.1))

#     # VGG Block 2

#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# #     model.add(BatchNormalization())

#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# #     model.add(BatchNormalization())

#     model.add(MaxPooling2D((2, 2)))

#     model.add(Dropout(0.1))

# #     # VGG Block 3

# #     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # #     model.add(BatchNormalization())

# #     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # #     model.add(BatchNormalization())

# #     model.add(MaxPooling2D((2, 2)))

# #     model.add(Dropout(0.1))



# # #     model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # # #     model.add(BatchNormalization())

# # #     model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # # #     model.add(BatchNormalization())

# # #     model.add(MaxPooling2D((2, 2)))

# # #     model.add(Dropout(0.1))



    

#     model.add(Flatten())

#     model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

# #     model.add(BatchNormalization())

#     model.add(Dropout(0.4))

    

#     opt = SGD(lr=0.001, momentum=0.9)

#     model.add(Dense(9, activation='softmax'))

#     # compile model

#     model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

#     return model



# def plotSamplePrediction(ds, numPlots, model, ds_label=None):

#     # sample data

#     sampled_data = []

#     sampled_labels = []

#     for i in np.random.choice(range(len(ds)), numPlots):

#         sampled_data.append(ds[i])

#         if ds_label is not None:

#             sampled_labels.append(ds_label[i])

#     sampled_data = np.array(sampled_data)

#     if ds_label is not None:

#         sampled_labels = np.array(sampled_labels)

#     # predict class

#     predicted_labels = model.predict_classes(sampled_data)

#     # draw

#     fig, axs = plt.subplots(1, numPlots, figsize=(20, 15))

#     for counter in range(numPlots):

#         axs[counter].imshow(sampled_data[counter].reshape(sampled_data[counter].shape[0], ds[counter].shape[1]), cmap='hot', interpolation='nearest')

#         axs[counter].grid(False)

#         if ds_label is not None:

#             axs[counter].title.set_text(typeList[np.argmax(sampled_labels[counter])].replace('ScatterPlot', '') + '/' + typeList[predicted_labels[counter]].replace('ScatterPlot', ''))

#         else:

#             axs[counter].title.set_text(typeList[predicted_labels[counter]].replace('ScatterPlot', ''))

#     plt.show()

    
# from keras.callbacks import EarlyStopping

# from keras.callbacks import ModelCheckpoint

# mc = ModelCheckpoint('best_model_cls.h5', monitor='val_accuracy', mode='max', save_best_only=True)

# es = EarlyStopping(monitor='val_accuracy', mode='max', patience=50)
# model = create_cnn_cls()

# # model = create_dense_model()

# plot_model(model=model, to_file='model.png', show_shapes=True, show_layer_names=True)
# history = model.fit(X_train, y_train, validation_split=0.33, epochs=500, callbacks=[mc, es])
# plt.plot(history.history['accuracy'])

# plt.plot(history.history['val_accuracy'])

# plt.title('Train/test accuracy')

# plt.xlabel('epochs')

# plt.ylabel('accuracy')
# ret = model.evaluate(X_test, y_test)

# print(rt)
# plotSamplePrediction(X_test, 10, model, y_test)
# plotSamplePrediction(X_real, 10, model)
# #For real-life data.

# from sklearn.model_selection import train_test_split

# X_train_test = X_real.reshape(X_real.shape[0], X_real.shape[1], X_real.shape[2], 1)

# y_train_test = y_real

# X_train, X_test, y_train, y_test = train_test_split(X_train_test, y_train_test, test_size=0.333)



# # For real-life + augmented data.

# X_train = X_real10.reshape(X_real10.shape[0], X_real10.shape[1], X_real10.shape[2], 1)

# y_train = y_real10

# X_test = X_real.reshape(X_real.shape[0], X_real.shape[1], X_real.shape[2], 1)

# y_test = y_real



# # For typical data

# X_train = X_typical.reshape(X_typical.shape[0], X_typical.shape[1], X_typical.shape[2], 1)

# y_train = y_typical

# X_test = X_typical1.reshape(X_typical1.shape[0], X_typical1.shape[1],X_typical1.shape[2], 1)

# y_test = y_typical1



# For both real and typical

# from sklearn.model_selection import train_test_split

# X = np.concatenate([X_real10, X_typical2])

# y = np.concatenate([y_real10, y_typical2])

# X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# To be consistent we keep the train/validation/test splits and re-train with different models.

# We will use the ones without filtering ther number of points as of experience models learns well and discard these scatter plots from learning too. As we train with the data filtered out the points => we get the same model with the same result.

import json

with open('/kaggle/input/typicalandrealwordscags/X_train.json') as f:

    X_train = json.load(f)

with open('/kaggle/input/typicalandrealwordscags/y_train.json') as f:

    y_train = json.load(f)

with open('/kaggle/input/typicalandrealwordscags/X_test.json') as f:

    X_test = json.load(f)

with open('/kaggle/input/typicalandrealwordscags/y_test.json') as f:

    y_test = json.load(f)
# len(X_train)
# len(X_test)
X_train = np.array(X_train)

y_train = np.array(y_train)

X_test = np.array(X_test)

y_test = np.array(y_test)
y_train_df = pd.DataFrame(y_train)

scagnosticScores = ["outlyingScore", "skewedScore", "clumpyScore", "sparseScore", "striatedScore", "convexScore", "skinnyScore", "stringyScore", "monotonicScore"]

y_train_df.columns = scagnosticScores

# !pip install pandas-profiling

import pandas_profiling

y_train_df.profile_report(style={'full_width': True})
y_test_df = pd.DataFrame(y_test)

scagnosticScores = ["outlyingScore", "skewedScore", "clumpyScore", "sparseScore", "striatedScore", "convexScore", "skinnyScore", "stringyScore", "monotonicScore"]

y_test_df.columns = scagnosticScores

# !pip install pandas-profiling

import pandas_profiling

y_test_df.profile_report(style={'full_width': True})
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Flatten

from keras.layers import Conv2D

from keras.layers import MaxPooling2D

from keras.layers import Dropout

from keras.utils import plot_model

from keras.regularizers import l2

from keras.optimizers import SGD

from keras.layers import BatchNormalization

# learned several reasons from here: https://machinelearningmastery.com/how-to-develop-a-cnn-from-scratch-for-cifar-10-photo-classification/

def create_cnn_model():

    model = Sequential()

    # VGG Block 1

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(40, 40, 1)))

#     model.add(BatchNormalization())

    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

#     model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.1))

    # VGG Block 2

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

#     model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

#     model.add(BatchNormalization())

    model.add(MaxPooling2D((2, 2)))

    model.add(Dropout(0.1))

    # VGG Block 3

#     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# #     model.add(BatchNormalization())

#     model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# #     model.add(BatchNormalization())

#     model.add(MaxPooling2D((2, 2)))

# #     model.add(Dropout(0.1))



# #     model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # #     model.add(BatchNormalization())

# #     model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))

# # #     model.add(BatchNormalization())

# #     model.add(MaxPooling2D((2, 2)))

# #     model.add(Dropout(0.1))



    

    model.add(Flatten())

    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))

#     model.add(BatchNormalization())

    model.add(Dropout(0.1))

    

    opt = SGD(lr=0.001, momentum=0.9)

    model.add(Dense(9, activation='relu'))

    # compile model

    model.compile(optimizer=opt, loss='mse')

    return model
model = create_cnn_model()

plot_model(model=model, to_file='model.png', show_shapes=True, show_layer_names=True)
from keras.callbacks import EarlyStopping

from keras.callbacks import ModelCheckpoint

mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True)

es = EarlyStopping(monitor='val_loss', mode='min', patience=50)
history = model.fit(X_train, y_train, validation_split=0.33, epochs=500, callbacks=[mc, es])
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Training/validation loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['training loss', 'validation loss'], loc='upper left')
# predict on the real data.

print(model.evaluate(X_test, y_test))
y_predicted = model.predict(X_test)
def plotResult(i):

    print(typeList[i])

    plt.figure(figsize=(15, 5))

    sortOrder = np.argsort(y_test[:, i])

    sortPredicted = np.array([y_predicted[:, i][idx] for idx in sortOrder])

    plt.scatter(np.arange(len(y_test)), sorted(y_test[:,i]), label='actual')

    plt.scatter(np.arange(len(y_predicted)), sortPredicted, label='predicted')

    plt.legend()

    plt.show()
plotResult(0)
plotResult(1)
plotResult(2)
plotResult(3)
plotResult(4)
plotResult(5)
plotResult(6)
plotResult(7)
plotResult(8)
# # dump the data too.

# import codecs, json

# def exportNPArrayToJSON(a, fileName):

#     b = a.tolist() # nested lists with same data, indices

#     json.dump(b, codecs.open(fileName, 'w', encoding='utf-8')) ### this saves the array in .json format
# exportNPArrayToJSON(X_train, "X_train.json")

# exportNPArrayToJSON(y_train, "y_train.json")

# exportNPArrayToJSON(X_test, "X_test.json")

# exportNPArrayToJSON(y_test, "y_test.json")