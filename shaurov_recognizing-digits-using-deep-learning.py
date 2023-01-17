import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer

import re

%pylab inline

import matplotlib.image as mpimg

from keras.models import  Sequential

from keras.layers.core import  Lambda , Dense, Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.layers import BatchNormalization, Convolution2D , MaxPooling2D



from keras.preprocessing import image



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df=pd.read_csv('../input/digit-recognizer/train.csv')

train = df[:]

print(train.shape)

train.head()
test= pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

print(test.shape)

test.head()
X_train_dataset = (train.iloc[:, 1:].values).astype('float32')

y_train_dataset = train.iloc[:, 0].values.astype('int32')



x_test_dataset = test.values.astype('float32')
X_train_dataset = X_train_dataset.reshape(

                        X_train_dataset.shape[0], 28, 28)

print(X_train_dataset.shape)

print(y_train_dataset.shape)



x_test_dataset = x_test_dataset.reshape(x_test_dataset.shape[0], 28, 28)

x_test_dataset.shape


fig=plt.figure(figsize=(7, 7))

columns = 3

rows = 2

for i in range(1, columns*rows +1):

    fig.add_subplot(rows, columns, i)

    img = X_train_dataset[i]

    plt.imshow(img)

    # if want to show gray image

    # plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))

    plt.title(y_train_dataset[i])

plt.show()

X_train_dataset = X_train_dataset.reshape(

                    X_train_dataset.shape[0], 28, 28, 1)

print(X_train_dataset.shape)



x_test_dataset = x_test_dataset.reshape(x_test_dataset.shape[0], 28, 28, 1)

x_test_dataset.shape
import tensorflow as tf



fig=plt.figure(figsize=(7, 7))

columns = 3

rows = 2

for i in range(1, columns*rows +1):

    fig.add_subplot(rows, columns, i)

    img = X_train_dataset[i]

    plt.imshow(img.squeeze(),cmap=plt.get_cmap('gray'))

    plt.title(y_train_dataset[i])

    

plt.show()

mean_px = X_train_dataset.mean().astype(np.float32)

std_px = X_train_dataset.std().astype(np.float32)



def standardize(x): 

    return (x-mean_px)/std_px
y_train_dataset
from sklearn.preprocessing import LabelEncoder

print(y_train_dataset[:3])

lb = LabelEncoder()

lb.fit(y_train_dataset)

y_train_dataset = lb.transform(y_train_dataset)



data = pd.get_dummies(y_train_dataset, columns = ['label'])

y_train_dataset = data[:]

print(y_train_dataset[:3])
# fixing random seed for reproducibility

seed = 43

np.random.seed(seed)
gen = image.ImageDataGenerator()

from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(featurewise_center=False,  # set input mean to 0 over the dataset

        samplewise_center=False,  # set each sample mean to 0

        featurewise_std_normalization=False,  # divide inputs by std of the dataset

        samplewise_std_normalization=False,  # divide each input by its std

        zca_whitening=False,  # apply ZCA whitening

        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)

        zoom_range = 0.1, 

        width_shift_range=0.1,  

        height_shift_range=0.1, 

        horizontal_flip=False,  

        vertical_flip=False)



datagen.fit(X_train_dataset)

from sklearn.model_selection import train_test_split

X = X_train_dataset

y = y_train_dataset

X_train, X_val, y_train, y_val = train_test_split(X_train_dataset, y_train_dataset, test_size=0.10, random_state=42)



print('X_train {},X_val {}, y_train {}, y_val {}'.format(X_train.shape, 

                                                         X_val.shape, 

                                                         y_train.shape, y_val.shape))

from keras.layers import Convolution2D, MaxPooling2D



def cnn_model():

    model = Sequential([

        Lambda(standardize, input_shape=(28,28,1)),

        Convolution2D(32,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(32,(3,3), activation='relu'),

        MaxPooling2D(pool_size=(2,2)),

        BatchNormalization(axis=1),

        

        Convolution2D(64,(3,3), activation='relu'),

        BatchNormalization(axis=1),

        Convolution2D(64,(3,3), activation='relu'),

        MaxPooling2D(pool_size=(2,2)),

        Flatten(),

        BatchNormalization(),

        

        Dense(512, activation='relu'),

        BatchNormalization(),

        Dense(10, activation='softmax')

        ])

    

    model.compile(optimizer='adam' , loss='categorical_crossentropy',

                  metrics=['accuracy'])

    return model
model = cnn_model()

model.optimizer.lr=0.0001



history=model.fit_generator(datagen.flow(X_train_dataset,

                            y_train_dataset, batch_size=84),

                            steps_per_epoch=42000, 

                            epochs=5, 

                            validation_data = (X_val,y_val)

                           )
# Predict the values from the validation dataset

Y_pred = model.predict(X_val)

# Convert predictions classes to one hot vectors 

Y_pred_classes = np.argmax(Y_pred,axis = 1) 

Y_pred_classes[:5]



x = y_val.stack()

Y_true = pd.Series(pd.Categorical(x[x!=0].index.get_level_values(1)))

from sklearn.metrics import confusion_matrix

# compute the confusion matrix

confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

confusion_mtx
import seaborn as sns

def plot_cm(y_true, y_pred, figsize=(20,10)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    cm_sum = np.sum(cm, axis=1, keepdims=True)

    cm_perc = cm / cm_sum.astype(float) * 100

    annot = np.empty_like(cm).astype(str)

    nrows, ncols = cm.shape

    for i in range(nrows):

        for j in range(ncols):

            c = cm[i, j]

            p = cm_perc[i, j]

            if i == j:

                s = cm_sum[i]

                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)

            elif c == 0:

                annot[i, j] = ''

            else:

                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))

    cm.index.name = 'Actual'

    cm.columns.name = 'Predicted'

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)



plot_cm(Y_true, Y_pred_classes)
prediction = model.predict_classes(x_test_dataset)
my_submission = pd.DataFrame({"ImageId": list(range(1,len(prediction)+1)),

                         "Label": prediction})

my_submission.to_csv("submission.csv", index=False, header=True)
var = pd.read_csv("./submission.csv")
var[:5]