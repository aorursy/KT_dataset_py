# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import pandas_profiling as pdp

import numpy as np

np.random.seed(2)



import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

import itertools



import keras

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

from keras.optimizers import RMSprop

from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
!ls ../input/digit-recognizer

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')

sample_submission = pd.read_csv('../input/digit-recognizer/sample_submission.csv')



y_train = train["label"]

# Drop 'label' column

X_train = train.drop(labels = ["label"],axis = 1) 



# free some space

del train 

g = sns.countplot(y_train)

y_train.value_counts()
X_train.isnull().any().describe()

test.isnull().any().describe() 

#unique:種類 top:最頻値 freq:最頻値の出現回数
# 0~1正則化

X_train = X_train / 255.0

test = test / 255.0



# 28×28 のチャネル1(grayscale)にreshape

# Reshape:Kerasの入力データの形式は(ミニバッチサイズ、横幅、縦幅、チャネル数)である必要があるので、reshape()を使って形式を変換する。

# 784次元あったものを28px*28pxにする。

# MNISTの画像はグレースケールなので、チャネル数は1。RGB画像の場合はチャネル数は3なので784pxのベクトルを28x28x3の3D行列にreshapeする。

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



# 0~9の10分割する

y_train = to_categorical(y_train, num_classes=10)



# 分割する

random_seed = 2

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)
#画像確認

print(X_train.shape)

test_show = plt.imshow(X_train[4][:,:,0])
# kerasでモデル構築

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu', input_shape=(28, 28, 1)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='Same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(10, activation='softmax'))
# コンパイル、コールバック等パラメータを定義する

model.compile(optimizer=RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0), loss='categorical_crossentropy', metrics=['accuracy'])

# learning rate を設定する

reduce_lr = ReduceLROnPlateau(monitor='val_acc',

                              patience=3,

                              verbose=1,

                              factor=0.5,

                              min_lr=0.00001)

epochs = 20

batch_size = 86

early_stopping = EarlyStopping(patience=0, verbose=1) 
# augmantation.



datagen = ImageDataGenerator(

        featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, # Randomly zoom image 

        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

        horizontal_flip=False,  # randomly flip images

        vertical_flip=False)  # randomly flip images





datagen.fit(X_train)
# 学習



history = model.fit_generator(

    datagen.flow(X_train, y_train, batch_size=batch_size),

    epochs=epochs,

    validation_data=(X_test, y_test),

    verbose=1,

    steps_per_epoch=X_train.shape[0] // batch_size,

    callbacks=[reduce_lr]

)
# Plot the loss and accuracy curves for training and validation 

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best', shadow=True)
# Look at confusion matrix 



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



# Predict the values from the validation dataset

y_pred = model.predict(X_test)

# Convert predictions classes to one hot vectors 

y_pred_classes = np.argmax(y_pred,axis = 1) 

# Convert validation observations to one hot vectors

y_true = np.argmax(y_test,axis = 1) 

# compute the confusion matrix

confusion_mtx = confusion_matrix(y_true, y_pred_classes) 

# plot the confusion matrix

plot_confusion_matrix(confusion_mtx, classes = range(10)) 
# Display some error results 



# Errors are difference between predicted labels and true labels

errors = (y_pred_classes - y_true != 0)



y_pred_classes_errors = y_pred_classes[errors]

y_pred_errors = y_pred[errors]

y_true_errors = y_true[errors]

X_test_errors = X_test[errors]



def display_errors(errors_index,img_errors,pred_errors, obs_errors):

    """ This function shows 6 images with their predicted and real labels"""

    n = 0

    nrows = 2

    ncols = 3

    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)

    for row in range(nrows):

        for col in range(ncols):

            error = errors_index[n]

            ax[row,col].imshow((img_errors[error]).reshape((28,28)))

            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))

            n += 1



# 予測した数字が間違っている確率

y_pred_errors_prob = np.max(y_pred_errors,axis = 1)



# 誤差集合内の真の値の予測確率

true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))



# 予測されたラベルと真のラベルの確率の差

delta_pred_true_errors = y_pred_errors_prob - true_prob_errors



# the delta prob errors　の sort list

sorted_dela_errors = np.argsort(delta_pred_true_errors)



# Top 6 errors 

most_important_errors = sorted_dela_errors[-6:]



# Show the top 6 errors

display_errors(most_important_errors, X_test_errors, y_pred_classes_errors,y_true_errors)
# predict results

results = model.predict(test)



# 10ラベルから一番でかいのを選ぶ（axis=1はたくさんのファイルが存在する為、一層落としてそれぞれのデータから見る）

results = np.argmax(results,axis = 1)



#pandasで整形

results = pd.Series(results,name="Label")



results
submission = pd.concat([pd.Series(range(1,28001), name="ImageId"), results], axis=1)

submission.to_csv('sample_submission.csv', index=False)