import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from tensorflow import keras

import tensorflow as tf

import os

%matplotlib inline

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_imgs = pd.read_csv("../input/train.csv")

test_imgs = pd.read_csv("../input/test.csv")
label_train = train_imgs["label"]
img_train = train_imgs.drop(labels= "label", axis = 1)

del train_imgs
sns.countplot(label_train)
img_train.max().max()
test_imgs.max().max()
img_train = img_train/255.

test_imgs = test_imgs/255.
print(img_train.max().max())

print(test_imgs.max().max())
img_train = img_train.values.reshape(-1,28,28,1)

test_imgs = test_imgs.values.reshape(-1,28,28,1)
label_train = keras.utils.to_categorical(label_train, num_classes= 10)
label_train[0]
train_x , test_x , train_y , test_y = train_test_split(img_train , label_train , 

                                            test_size = 0.2 ,

                                            random_state = 42)
plt.imshow(train_x[10][:,:,0])
def CNN(n_conv):

    """

    Build a Convolutional neural network for n_conv number of convolutional layers.

    n_conv: Integer.

    """

    model = keras.Sequential()

    

    for _ in range(n_conv):

        model.add(keras.layers.Conv2D(64, kernel_size= 2, activation=tf.nn.relu, input_shape=(28,28,1)))

        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(keras.layers.Dropout(0.2))

        

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(256, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu))

    model.add(keras.layers.Dense(10, activation=tf.nn.softmax))



    return model
model = CNN(n_conv=3)

model.compile(optimizer= keras.optimizers.Adam(), 

              loss= keras.losses.categorical_crossentropy,

              metrics=['accuracy'])
#Input Layer

visible = keras.layers.Input(shape=(28,28,1))

#Convolutional layer

conv1 = keras.layers.Conv2D(32, kernel_size=4, activation='relu')(visible)

pool1 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

flat1 = keras.layers.Flatten()(pool1)



#Convolutional layer

conv2 = keras.layers.Conv2D(32, kernel_size=4, activation='relu')(visible)

pool2 = keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

flat2 = keras.layers.Flatten()(pool2)

#Merge

merge = keras.layers.concatenate([flat1, flat2])

#Dense layer

hidden1 = keras.layers.Dense(256, activation='relu')(merge)

drop = keras.layers.Dropout(0.5)(hidden1)

#Output layer

output = keras.layers.Dense(10, activation='softmax')(drop)

model = keras.Model(inputs=visible, outputs=output)

print(model.summary())

model.compile(optimizer= keras.optimizers.Adam(), 

              loss= keras.losses.categorical_crossentropy,

              metrics=['accuracy'])

earlystop = keras.callbacks.EarlyStopping(monitor='val_acc',

                                          min_delta=0.001,

                                          patience=7, 

                                          mode='min')



reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.2,

                              patience=5, min_lr=0.001)
history = model.fit(train_x,

                    train_y,epochs=20,

                    batch_size=200,

                    validation_data= (test_x, test_y),

                    verbose = 1,

                    callbacks= [reduce_lr])
#accuracy

train_accuracy = history.history['acc']

validation_accuracy = history.history['val_acc']



#loss 

train_loss = history.history['loss']

validation_loss = history.history['val_loss']



#Epochs

epoch_range = range(1,len(train_accuracy)+1)



#Plot

fig, ax = plt.subplots(1, 2, figsize=(12,5))



ax[0].set_title('Accuracy per Epoch')

sns.lineplot(x=epoch_range,y=train_accuracy,marker='o',ax=ax[0])

sns.lineplot(x=epoch_range,y=validation_accuracy,marker='o',ax=ax[0])

ax[0].legend(['training','validation'])

ax[0].set_xlabel('Epoch')

ax[0].set_ylabel('Accuracy')

ax[1].set_title('Loss per Epoch')

sns.lineplot(x=epoch_range,y=train_loss,marker='o',ax=ax[1])

sns.lineplot(x=epoch_range,y=validation_loss,marker='o',ax=ax[1])

ax[1].legend(['training','validation'])

ax[1].set_xlabel('Epoch')

ax[1].set_ylabel('Loss')

plt.show()
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,

                          normalize=False,

                          title=None,

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if not title:

        if normalize:

            title = 'Normalized confusion matrix'

        else:

            title = 'Confusion matrix, without normalization'



    # Compute confusion matrix

    cm = confusion_matrix(y_true, y_pred)

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots()

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...

    ax.set(xticks=np.arange(cm.shape[1]),

           yticks=np.arange(cm.shape[0]),

           # ... and label them with the respective list entries

           xticklabels=classes, yticklabels=classes,

           title=title,

           ylabel='True label',

           xlabel='Predicted label')



    # Rotate the tick labels and set their alignment.

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",

             rotation_mode="anchor")



    # Loop over data dimensions and create text annotations.

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            ax.text(j, i, format(cm[i, j], fmt),

                    ha="center", va="center",

                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()

    return ax

Predict = model.predict(test_x)

Predict_classes = np.argmax(Predict,axis = 1) 

True_classes = np.argmax(test_y, axis = 1)

plot_confusion_matrix(True_classes, Predict_classes, classes= range(10), normalize=False,

                      title='Confusion Matrix')
Predict = model.predict(test_imgs)
number_predict = []

for i in Predict:

    number_predict.append(np.argmax(i))
sub = pd.read_csv('../input/sample_submission.csv')
sub['Label'] = number_predict
sub.head()
sub.to_csv('Final.csv', index = False)