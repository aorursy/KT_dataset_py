# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

!ls -l /kaggle/input/traffic-signs-classification
# Visualization

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline

from tensorflow.keras.utils import plot_model



# Splitting data

from sklearn.model_selection import train_test_split



# Metrics 

from sklearn.metrics import confusion_matrix, classification_report



# Deep Learning

import tensorflow as tf

print('TensoFlow Version: ', tf.__version__)

from tensorflow.keras.preprocessing.image import ImageDataGenerator



from tensorflow.keras.models import Model

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout

from tensorflow.keras.applications.resnet import ResNet50



from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
path = '/kaggle/input/traffic-signs-classification'

lab = pd.read_csv('/kaggle/input/traffic-signs-classification/labels.csv')

# Count PLot of the samples/observations w.r.t the classes

d = dict()

class_labels = dict()

for dirs in os.listdir(path + '/myData'):

    count = len(os.listdir(path+'/myData/'+dirs))

    d[dirs+' => '+lab[lab.ClassId == int(dirs)].values[0][1]] = count

    class_labels[int(dirs)] = lab[lab.ClassId == int(dirs)].values[0][1]



plt.figure(figsize = (20, 50))

sns.barplot(y = list(d.keys()), x = list(d.values()), palette = 'Set3')

plt.ylabel('Label')

plt.xlabel('Count of Samples/Observations');
# input image dimensions

img_rows, img_cols = 32, 32

# The images are RGB.

img_channels = 3

nb_classes = len(class_labels.keys())



datagen = ImageDataGenerator()

data = datagen.flow_from_directory('/kaggle/input/traffic-signs-classification/myData',

                                    target_size=(32, 32),

                                    batch_size=73139,

                                    class_mode='categorical',

                                    shuffle=True )
X , y = data.next()
# Labels are one hot encoded

print(f"Data Shape   :{X.shape}\nLabels shape :{y.shape}")
fig, axes = plt.subplots(10,10, figsize=(18,18))

for i,ax in enumerate(axes.flat):

    r = np.random.randint(X.shape[0])

    ax.imshow(X[r].astype('uint8'))

    ax.grid(False)

    ax.axis('off')

    ax.set_title('Label: '+str(np.argmax(y[r])))

    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=11)
print("Train Shape: {}\nTest Shape : {}".format(X_train.shape, X_test.shape))
resnet = ResNet50(weights= None, include_top=False, input_shape= (img_rows,img_cols,img_channels))
x = resnet.output

x = GlobalAveragePooling2D()(x)

x = Dropout(0.5)(x)

predictions = Dense(nb_classes, activation= 'softmax')(x)

model = Model(inputs = resnet.input, outputs = predictions)
model.summary()
plot_model(model, show_layer_names=True, show_shapes =True, to_file='model.png', dpi=350)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_check = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')



early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=0, mode='max', restore_best_weights=True)



reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)



csv_logger = CSVLogger('train_log.csv', separator=',')

n_epochs = 10

history =  model.fit(X_train, y_train,  batch_size = 32, epochs = n_epochs, verbose = 1, 

              validation_data = (X_test, y_test), callbacks = [model_check, early, reduce_lr, csv_logger])
# Saving the model

model.save('TSC_model.h5')
loss, acc = model.evaluate(X_test, y_test)

print('Accuracy: ', acc, '\nLoss    : ', loss)
q = len(list(history.history['loss']))

plt.figure(figsize=(12, 6))

sns.lineplot(x = range(1, 1+q), y = history.history['accuracy'], label = 'Accuracy')

sns.lineplot(x = range(1, 1+q), y = history.history['loss'], label = 'Loss')

plt.xlabel('#epochs')

plt.ylabel('Training')

plt.legend();
plt.figure(figsize=(12, 6))

sns.lineplot(x = range(1, 1+q), y = history.history['accuracy'], label = 'Train')

sns.lineplot(x = range(1, 1+q), y = history.history['val_accuracy'], label = 'Validation')

plt.xlabel('#epochs')

plt.ylabel('Accuracy')

plt.legend();
plt.figure(figsize=(12, 6))

sns.lineplot(x = range(1, 1+q), y = history.history['loss'], label = 'Train')

sns.lineplot(x = range(1, 1+q), y = history.history['val_loss'], label = 'Validation')

plt.xlabel('#epochs')

plt.ylabel('Loss')

plt.legend();
%%time

pred = np.argmax(model.predict(X_test), axis = 1)
labels = [class_labels[i] for i in range(43)]

print(classification_report(np.argmax(y_test, axis = 1), pred, target_names = labels))
cmat = confusion_matrix(np.argmax(y_test, axis=1), pred)

plt.figure(figsize=(16,16))

sns.heatmap(cmat, annot = True, cbar = False, cmap='Paired', fmt="d", xticklabels=labels, yticklabels=labels);
classwise_acc = cmat.diagonal()/cmat.sum(axis=1) * 100 

cls_acc = pd.DataFrame({'Class_Label':[class_labels[i] for i in range(43)], 'Accuracy': classwise_acc.tolist()}, columns = ['Class_Label', 'Accuracy'])

cls_acc.style.format({"Accuracy": "{:,.2f}",}).hide_index().bar(subset=["Accuracy"], color='tomato')
fig, axes = plt.subplots(5,5, figsize=(18,18))

for i,ax in enumerate(axes.flat):

    r = np.random.randint(X_test.shape[0])

    ax.imshow(X_test[r].astype('uint8'))

    ax.grid(False)

    ax.axis('off')

    ax.set_title('Original: {} Predicted: {}'.format(np.argmax(y_test[r]), np.argmax(model.predict(X_test[r].reshape(1, 32, 32, 3)))))