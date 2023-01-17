import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import seaborn as sns

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

%matplotlib inline



from sklearn.model_selection import train_test_split

import itertools



from keras.utils.np_utils import to_categorical # convert to one-hot-encoding

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, GlobalMaxPooling2D, BatchNormalization

from keras.optimizers import RMSprop, Adadelta

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import VGG16, preprocess_input

from keras.preprocessing import image

from keras.preprocessing.image import img_to_array, array_to_img

from keras.utils import np_utils

import keras.backend as K



import cv2

import scipy



from random import randint

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.preprocessing import StandardScaler

from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn.model_selection import KFold
SEED = 42
data = pd.read_csv('../input/fashionmnist/fashion-mnist_train.csv', sep = ',')
data.head()
X,y = data.drop(['label'], axis = 1), data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
g = sns.countplot(y_train)



y_train.value_counts()
X_train.isnull().any().describe()
X_train = X_train / 255.

X_test = X_test / 255.
label_dictionnary = {0:'T-shirt/top', 1:'Trouser', 2:'Pullover', 

                     3:'Dress', 4:'Coat', 5:'Sandal', 6:'Shirt', 

                     7:'Sneaker', 8:'Bag', 9:'Ankle boot' }

def true_label(x):

    return label_dictionnary[x]
example = 16

g = plt.imshow(X_train.values.reshape(-1,28,28,1)[example][:,:,0])

print('this represent a : ' + true_label(y_train[example]))
# Taking only the first N rows to speed things up

X_PCA = X_train[:3000].values



# Call the PCA method with 5 components. 

pca = PCA(n_components=5)

pca.fit(X_PCA)

X_5d = pca.transform(X_PCA)



# For cluster coloring in our Plotly plots, remember to also restrict the target values 

Target_name = y_train[:3000].apply(true_label)

Target = y_train[:3000]
trace0 = go.Scatter(

    x = X_5d[:,0],

    y = X_5d[:,1],

    mode = 'markers',

    text = Target_name,

    showlegend = False,

    marker = dict(

        size = 8,

        color = Target,

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [trace0]



layout = go.Layout(

    title= 'Principal Component Analysis (PCA)',

    hovermode= 'closest',

    xaxis= dict(

         title= 'First Principal Component',

        ticklen= 5,

        zeroline= False,

        gridwidth= 2,

    ),

    yaxis=dict(

        title= 'Second Principal Component',

        ticklen= 5,

        gridwidth= 2,

    ),

    showlegend= True

)





fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
# Invoking the t-SNE method

tsne = TSNE(n_components=2)

tsne_results = tsne.fit_transform(X_PCA) 
traceTSNE = go.Scatter(

    x = tsne_results[:,0],

    y = tsne_results[:,1],

    text = Target_name,

    mode = 'markers',

    showlegend = True,

    marker = dict(

        size = 8,

        color = Target,

        colorscale ='Jet',

        showscale = False,

        line = dict(

            width = 2,

            color = 'rgb(255, 255, 255)'

        ),

        opacity = 0.8

    )

)

data = [traceTSNE]



layout = dict(title = 'TSNE (T-Distributed Stochastic Neighbour Embedding)',

              hovermode= 'closest',

              yaxis = dict(zeroline = False),

              xaxis = dict(zeroline = False),

              showlegend= False,



             )



fig = dict(data=data, layout=layout)

py.iplot(fig, filename='styled-scatter')
y_predicted = [randint(0, 9) for p in range(0, len(y_test))]
print(accuracy_score(y_test,y_predicted))
kfold = KFold(n_splits=3, shuffle=True)



X_train = X_train.reset_index(drop=True)



y_train.index = X_train.index

performance_accuracy = []



fold_count = 1



pipe = Pipeline([('pca',PCA(n_components=100)),

                 ('svm',RandomForestClassifier(n_jobs=-1, n_estimators=20))])
for (train_index, test_index) in kfold.split(X_train):

    print('fold '+str(fold_count)+': ')

    print('-- training --')

    X_kfold_train, y_kfold_train = X_train.iloc[train_index], y_train.loc[train_index]

    X_kfold_test, y_kfold_test = X_train.iloc[test_index], y_train.loc[test_index]

    pipe.fit(X_kfold_train, y_kfold_train)

    y_kfold_pred = pipe.predict(X_kfold_test)

    print('-- train ok ! --')

    perf = accuracy_score(y_kfold_test, y_kfold_pred)

    performance_accuracy.append(perf)

    

    fold_count = fold_count + 1

    

print(np.mean(performance_accuracy))
print(performance_accuracy)
pipe.fit(X_train, y_train)

y_predicted = pipe.predict(X_test)



print(accuracy_score(y_test, y_predicted))
batch_size = 128

epochs = 30
Y_train = np_utils.to_categorical(y_train, 10)

Y_test = np_utils.to_categorical(y_test, 10)
def first_nn():

    model = Sequential()

    model.add(Dense(512, activation='relu', input_shape=(784,)))

    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    return model
model = first_nn()

model.summary()
model.compile(loss='categorical_crossentropy',

                 optimizer=RMSprop(),

                 metrics=['accuracy'])
fold_count = 1

performance_accuracy = []

for (train_index, test_index) in kfold.split(X_train):

    print('fold '+str(fold_count)+': ')

    print('-- training --')

    X_kfold_train, y_kfold_train = X_train.iloc[train_index], y_train.loc[train_index]

    X_kfold_test, y_kfold_test = X_train.iloc[test_index], y_train.loc[test_index]

    Y_kfold_train = np_utils.to_categorical(y_kfold_train,10)

    Y_kfold_test = np_utils.to_categorical(y_kfold_test,10)

    first_nn_crossval = first_nn()

    first_nn_crossval.compile(loss='categorical_crossentropy',

                              optimizer=RMSprop(),

                              metrics=['accuracy'])

    history = first_nn_crossval.fit(X_kfold_train, Y_kfold_train,

                                    batch_size=batch_size,

                                    epochs=epochs,

                                    verbose=10,

                                    validation_data=(X_kfold_test, Y_kfold_test))

    y_kfold_pred = first_nn_crossval.predict_classes(X_kfold_test)

    print('-- train ok ! --')

    perf = accuracy_score(y_kfold_test, y_kfold_pred)

    performance_accuracy.append(perf)

    

    fold_count = fold_count + 1

    

print(np.mean(performance_accuracy))
history = model.fit(X_train, Y_train,

                    batch_size=batch_size,

                    epochs=epochs,

                    verbose=1)
y_pred_nn = model.predict_classes(X_test)
perf = accuracy_score(y_test, y_pred_nn)
print(perf)
img_rows, img_cols = 28, 28

batch_size = 256

epochs = 15

input_shape = (img_rows, img_cols, 1)
def first_cnn(input_shape=input_shape):

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3),

                     activation='relu',

                     input_shape=input_shape))

    model.add(BatchNormalization())

    model.add(Conv2D(64, (3, 3), activation='relu'))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))

    model.add(BatchNormalization())

    model.add(Dropout(0.5))

    model.add(Dense(10, activation='softmax'))

    return model
model = first_cnn()

model.compile(loss='categorical_crossentropy',

              optimizer=Adadelta(),

              metrics=['accuracy'])
print(model.summary())
x_train = np.array(X_train).reshape(X_train.shape[0], img_rows, img_cols, 1)

x_test = np.array(X_test).reshape(X_test.shape[0], img_rows, img_cols, 1)
fold_count = 1

performance_accuracy = []

for (train_index, test_index) in kfold.split(X_train):

    print('fold '+str(fold_count)+': ')

    print('-- training --')

    X_kfold_train, y_kfold_train = X_train.iloc[train_index], y_train.loc[train_index]

    X_kfold_test, y_kfold_test = X_train.iloc[test_index], y_train.loc[test_index]

    X_kfold_train = np.array(X_kfold_train).reshape(X_kfold_train.shape[0], img_rows, img_cols, 1)

    X_kfold_test = np.array(X_kfold_test).reshape(X_kfold_test.shape[0], img_rows, img_cols, 1)

    Y_kfold_train = np_utils.to_categorical(y_kfold_train,10)

    Y_kfold_test = np_utils.to_categorical(y_kfold_test,10)

    

    first_nn_crossval = first_cnn()

    first_nn_crossval.compile(loss='categorical_crossentropy',

                              optimizer=Adadelta(),

                              metrics=['accuracy'])

    history = first_nn_crossval.fit(X_kfold_train, Y_kfold_train,

                                    batch_size=batch_size,

                                    epochs=epochs,

                                    verbose=10,

                                    validation_data=(X_kfold_test, Y_kfold_test))

    y_kfold_pred = first_nn_crossval.predict_classes(X_kfold_test)

    print('-- train ok ! --')

    perf = accuracy_score(y_kfold_test, y_kfold_pred)

    performance_accuracy.append(perf)

    

    fold_count = fold_count + 1

    

print(np.mean(performance_accuracy))
model.fit(x_train, Y_train,

          batch_size=batch_size,

          epochs=epochs,

          verbose=1,

          validation_data=(x_test, Y_test))

score = model.evaluate(x_test, Y_test, verbose=0)

y_pred_cnn = model.predict_classes(x_test)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
batch_size = 256

epochs = 5
def resize_all(x, shape = (48,48)):

    band_shape = x.shape

    x_resize = np.zeros(shape = (band_shape[0],shape[0],shape[1]))

    for i in range(band_shape[0]):

        x_resize[i] = scipy.misc.imresize(x[i],shape)

    return x_resize





def transform_input_vgg(x):

    x_vgg = np.array(x).reshape(-1,28,28)

    x_vgg = resize_all(x_vgg, (48,48))

    x_vgg = np.repeat(x_vgg[:, :, :, np.newaxis], 3, axis=3)

#    x_vgg = preprocess_input(x_vgg)

    return x_vgg
def vgg16_model():

    vgg_conv = VGG16(weights= None , include_top=False, 

                     input_shape=(48, 48, 3))

    vgg_conv.load_weights('../input/keras-pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    vgg_conv.trainable = False

    model = Sequential()



    # Add the vgg convolutional base model

    model.add(vgg_conv)



    # Add new layers

    model.add(Flatten())

    model.add(Dense(1024, activation='relu'))   

    model.add(Dense(256, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    return model
fold_count = 1

performance_accuracy = []

for (train_index, test_index) in kfold.split(X_train):

    print('fold '+str(fold_count)+': ')

    print('-- training --')

    X_kfold_train, y_kfold_train = X_train.iloc[train_index], y_train.loc[train_index]

    X_kfold_test, y_kfold_test = X_train.iloc[test_index], y_train.loc[test_index]

    X_kfold_train = transform_input_vgg(X_kfold_train)

    X_kfold_test = transform_input_vgg(X_kfold_test)

    Y_kfold_train = np_utils.to_categorical(y_kfold_train,10)

    Y_kfold_test = np_utils.to_categorical(y_kfold_test,10)

    

    first_nn_crossval = vgg16_model()

    first_nn_crossval.compile(loss='categorical_crossentropy',

                              optimizer='adam',

                              metrics=['accuracy'])

    history = first_nn_crossval.fit(X_kfold_train, Y_kfold_train,

                                    batch_size=batch_size,

                                    epochs=epochs,

                                    verbose=1,

                                    validation_data=(X_kfold_test, Y_kfold_test))

    y_kfold_pred = first_nn_crossval.predict_classes(X_kfold_test)

    print('-- train ok ! --')

    perf = accuracy_score(y_kfold_test, y_kfold_pred)

    performance_accuracy.append(perf)

    

    fold_count = fold_count + 1

    

print(np.mean(performance_accuracy))
y_pred_series = pd.Series(y_pred_cnn, index = y_test.index)
cm = confusion_matrix(y_test, y_pred_series)

plt.figure(figsize=(9,9))

plt.imshow(cm, interpolation='nearest', cmap='Pastel1')

plt.title('Confusion matrix', size = 15)

plt.colorbar()

tick_marks = np.arange(10)

plt.xticks(tick_marks, ['T-shirt/top', 'Trouser', 'Pullover', 

                        'Dress', 'Coat', 'Sandal', 'Shirt', 

                        'Sneaker', 'Bag', 'Ankle boot'], rotation=45, size = 10)

plt.yticks(tick_marks, ['T-shirt/top', 'Trouser', 'Pullover', 

                        'Dress', 'Coat', 'Sandal', 'Shirt', 

                        'Sneaker', 'Bag', 'Ankle boot'], size = 10)

plt.tight_layout()

plt.ylabel('Actual label', size = 15)

plt.xlabel('Predicted label', size = 15)

width, height = cm.shape

for x in range(width):

    for y in range(height):

        plt.annotate(str(cm[x][y]), xy=(y, x), 

        horizontalalignment='center',

        verticalalignment='center')
plt.figure(figsize=(20,10))

for index, (image, label) in enumerate(zip(np.array(X_test)[1:10], y_pred_series[1:10])):

    plt.subplot(1, 10, index + 1)

    plt.imshow(np.reshape(np.array(image), (28,28)), cmap=plt.cm.gray)

    plt.title('Predicted: ' + str(true_label(label)) , fontsize = 10)
y_test.apply(true_label).loc[y_test!=y_pred_series].value_counts()
j = 56



x = x_train[j].reshape(1, 28, 28, 1)



preds = model.predict(x)

class_idx = np.argmax(preds[0])

class_output = model.output[:, class_idx]

last_conv_layer = model.layers[-6]
print(model.layers[-6].output_shape)
grads = K.gradients(class_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(64):

    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)

heatmap /= np.max(heatmap)
img = x_train[j].reshape(28, 28, 1)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
fig = plt.figure()

a = fig.add_subplot(1, 2, 1)

imgplot = plt.imshow(img.reshape(28,28))

a.set_title('image')

a = fig.add_subplot(1, 2, 2)

imgplot = plt.imshow(heatmap)

a.set_title('heatmap')

print('predicted label: '+ true_label(class_idx) + ' VS real label :' + true_label(y_train[j]))