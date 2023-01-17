import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.listdir('../input/volcanoes_train/')
train_images = pd.read_csv('../input/volcanoes_train/train_images.csv', header = None)
train_labels = pd.read_csv('../input/volcanoes_train/train_labels.csv')
test_images = pd.read_csv('../input/volcanoes_test/test_images.csv', header = None)
test_labels = pd.read_csv('../input/volcanoes_test/test_labels.csv')
train_images.shape, train_labels.shape
train_images.head()
train_labels.head()
train_counts = train_labels['Volcano?'].value_counts()
test_counts = test_labels['Volcano?'].value_counts()

plt.figure(figsize = (8,4))
plt.subplot(121)
sns.barplot(train_counts.index, train_counts.values)
plt.title('volcanos in training set')
plt.subplot(122)
sns.barplot(test_counts.index, test_counts.values)
plt.title('volcanos in testing set')

pos_samples = train_images[train_labels['Volcano?'] == 1].sample(5)
neg_samples = train_images[train_labels['Volcano?'] == 0].sample(5)

plt.subplots(figsize = (15,6))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(pos_samples.iloc[i,:].values.reshape((110, 110)), cmap = 'gray')
    if i == 0: plt.ylabel('Volcano')
for i in range(5):
    plt.subplot(2,5,i+6)
    if i == 0: plt.ylabel('No Volcano')
    plt.imshow(neg_samples.iloc[i,:].values.reshape((110,110)), cmap = 'gray')
Xtrain_raw = train_images/256
ytrain_raw = train_labels['Volcano?']
Xtest_raw = test_images/256
ytest_raw = test_labels['Volcano?']
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

Xtrain, Xvali, ytrain, yvali = train_test_split(Xtrain_raw, ytrain_raw, test_size = 0.2, random_state = 3)
Xtest, ytest = Xtest_raw, ytest_raw
modelLR = LogisticRegression()

from time import time
start = time()
modelLR.fit(Xtrain, ytrain)
end = time()
print('training time: {} mins.'.format((end-start)/60))
from sklearn.metrics import classification_report
predVali = modelLR.predict(Xvali)
predTest = modelLR.predict(Xtest)
print('validation report:','\n',classification_report(yvali, predVali))
print('testing report:', '\n', classification_report(ytest, predTest))
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, MaxPool2D, Dropout
#  can be used to fix the random seed to get reproducable result
# from numpy.random import seed
# from tensorflow import set_random_seed
# seed(42)
# set_random_seed(42)
img_rows, img_cols = 110, 110

X = Xtrain_raw.values.reshape((-1, img_rows, img_cols, 1))
y = ytrain_raw.values
X_train, X_vali, y_train, y_vali = train_test_split(X, y, test_size = 0.2, random_state = 3)

X_test = Xtest_raw.values.reshape((-1, img_rows, img_cols, 1))
y_test = ytest_raw.values


# kernel_initializer can be tuned for the first conv2D layer
init = keras.initializers.RandomNormal(mean=0, stddev=0.1 )
modelCNN1 = Sequential()
modelCNN1.add(Conv2D(6, kernel_size = (3,3),kernel_initializer=init, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
modelCNN1.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN1.add(Dropout(0.5))
modelCNN1.add(Conv2D(12, kernel_size = (3,3), activation = 'relu'))
modelCNN1.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN1.add(Dropout(0.5))
modelCNN1.add(Conv2D(24, kernel_size = (3,3), activation = 'relu'))
modelCNN1.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN1.add(Dropout(0.5))
modelCNN1.add(Flatten())
modelCNN1.add(Dense(1, activation = 'sigmoid'))

modelCNN1.summary()
# the line bolow can be used for tuning the adam optimizer, e.g. different initial learning rate
# adam = keras.optimizers.Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
modelCNN1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# the callBack parameter can be added to model.fit as 'callbacks = [callBack]' for early termination
from keras.callbacks import EarlyStopping
callBack = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=0, mode='auto')


def reset_weights(model):
    session = keras.backend.get_session()
    for layer in model.layers: 
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)

epochs = 100
batch_size = 64
                        
reset_weights(modelCNN1)            
history = modelCNN1.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_vali, y_vali),
                        callbacks=[callBack]
                       )
def report(model):
    predVali = model.predict_classes(X_vali)
    predTest = model.predict_classes(X_test)
    print('validation report:','\n',classification_report(y_vali, predVali))
    print('validation accuracy:', accuracy_score(y_vali, predVali))
    print('testing report:', '\n', classification_report(y_test, predTest))
    print('test accuracy:', accuracy_score(y_test, predTest))

def plotLearningCurves(history):
    fig, ax = plt.subplots(1,2, figsize = (14,6))
    ax[0].plot(history.epoch, history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.epoch, history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    ax[0].legend(loc='best', shadow=True)
    ax[0].set_title('loss vs epoch')

    ax[1].plot(history.epoch, history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.epoch, history.history['val_acc'], color='r',label="Validation accuracy")
    ax[1].legend(loc='best', shadow=True)
    ax[1].set_title('accuracy vs epoch')

plotLearningCurves(history)    
report(modelCNN1)
modelCNN2 = keras.models.clone_model(modelCNN1)
modelCNN2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
reset_weights(modelCNN2)

# the block below computes the class weights from the training set
from collections import Counter
counter = Counter(y_train) 
max_val = float(max(counter.values()))       
class_weight = {class_id : max_val/num_images for class_id, num_images in counter.items()}

epochs = 80
batch_size = 64
history2 = modelCNN2.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                         validation_data=(X_vali, y_vali),
                         # callbacks=[callBack],
                         class_weight = class_weight
                        )
plotLearningCurves(history2)
report(modelCNN2)
from keras.preprocessing.image import ImageDataGenerator


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = False, # Randomly zoom image 
        width_shift_range= False,  # randomly shift images horizontally (fraction of total width)
        height_shift_range= False,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)  # randomly flip images


datagen.fit(X_train)
generator = datagen.flow(X_train, y_train, batch_size= batch_size)

modelCNN3 = keras.models.clone_model(modelCNN1)
modelCNN3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

epochs = 60
reset_weights(modelCNN3)
history3 = modelCNN3.fit_generator(generator,epochs = epochs, validation_data = (X_vali,y_vali),
                                   class_weight = class_weight,
                                   #callbacks=[callBack]
                                  )
plotLearningCurves(history3)
report(modelCNN3)
init = keras.initializers.RandomNormal(mean=0, stddev=0.1 )
modelCNN4 = Sequential()
modelCNN4.add(Conv2D(32, kernel_size = (3,3),kernel_initializer=init, activation = 'relu', input_shape = (img_rows, img_cols, 1)))
modelCNN4.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN4.add(Dropout(0.5))
modelCNN4.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
modelCNN4.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN4.add(Dropout(0.5))
modelCNN4.add(Conv2D(128, kernel_size = (3,3), activation = 'relu'))
modelCNN4.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN4.add(Dropout(0.5))
modelCNN4.add(Conv2D(64, kernel_size = (3,3), activation = 'relu'))
modelCNN4.add(MaxPool2D(pool_size=(2,2), strides=2))
modelCNN4.add(Dropout(0.5))
modelCNN4.add(Flatten())
modelCNN4.add(Dense(1, activation = 'sigmoid'))

modelCNN4.summary()
modelCNN4.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

reset_weights(modelCNN4)
epochs = 80
batch_size = 64
history4 = modelCNN4.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                         validation_data=(X_vali, y_vali),
                         callbacks=[callBack],
                         class_weight = class_weight
                        )
plotLearningCurves(history4)
report(modelCNN4)

