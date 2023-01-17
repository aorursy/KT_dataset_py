import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



%matplotlib inline



import os

print(os.listdir("../input"))



from keras.models import Model

from keras.layers import Input

from keras.layers import Conv2D, MaxPool2D, Dense, Flatten

from keras.callbacks import EarlyStopping



from keras import backend as K



from sklearn.preprocessing import MinMaxScaler
train_df = pd.read_csv('../input/sign_mnist_train.csv')

test_df = pd.read_csv('../input/sign_mnist_test.csv')
train_df.head(8)
train_df.info()
y_train = train_df['label']

X_train = train_df.drop(columns=['label'])

y_test = test_df['label']

X_test = test_df.drop(columns=['label'])
y_train.unique()
## ONE HOT ENCODE

y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)
x = Input(shape=(784, ))
layer1 = Dense(784, activation='relu')(x)

layer2 = Dense(500, activation='relu')(layer1)

layer3 = Dense(300, activation='relu')(layer2)

layer4 = Dense(100, activation='relu')(layer3)

layer5 = Dense(25, activation='relu')(layer4)

predictions = Dense(24, activation='softmax')(layer5)
model = Model(inputs=x, outputs=predictions)
model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
earlyStopper = EarlyStopping(monitor='acc', patience=1, restore_best_weights=True)
hist = model.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])
plt.plot(hist.history['acc'])

plt.plot(hist.history['loss'])

plt.legend(['accuracy', 'loss'], loc='right')

plt.title('accuracy and loss');

plt.xlabel('epoch');

plt.ylabel('accuracy/loss');
K.clear_session()
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
X_train = X_train.reshape((X_train.shape[0], 28,28,1))

X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
def show_images(images, cols = 1, titles = None):

    """Display a list of images in a single figure with matplotlib.

    

    Parameters

    ---------

    images: List of np.arrays compatible with plt.imshow.

    

    cols (Default = 1): Number of columns in figure (number of rows is 

                        set to np.ceil(n_images/float(cols))).

    

    titles: List of titles corresponding to each image. Must have

            the same length as titles.

    """

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: print('Serial title'); titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image.reshape((28,28)), cmap=None)

        a.set_title(title, fontsize=50)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

plt.show()
samples = np.random.choice(len(X_train), 8)

sample_images = []

sample_labels = []

for sample in samples:

    sample_images.append(X_train[sample])

    sample_labels.append(np.argmax(y_train.iloc[sample]))
show_images(sample_images, 2, titles=sample_labels)
inputs = Input(shape=(28,28,1))
l1 = Conv2D(10, kernel_size=[3,3], activation='relu', padding='valid')(inputs)

l2 = Conv2D(50, kernel_size=[3,3], activation='relu', padding='same')(l1)

l3 = MaxPool2D(pool_size=[2,2])(l2)

l4 = Conv2D(100, kernel_size=[5,5], activation='relu', padding='valid')(l3)

l5 = Conv2D(100, kernel_size=[3,3], activation='relu', padding='same')(l4)

l6 = MaxPool2D(pool_size=[5,5])(l5)

l7 = Flatten()(l5)

l8 = Dense(1024, activation='relu')(l7)

l9 = Dense(512, activation='relu')(l8)

predictions = Dense(24, activation='softmax')(l9)
modelCNN = Model(inputs=inputs, outputs=predictions)
modelCNN.summary()
modelCNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hist = modelCNN.fit(x=X_train, y=y_train, batch_size=100, epochs=100, callbacks=[earlyStopper])
plt.plot(hist.history['acc'])

plt.plot(hist.history['loss'])

plt.legend(['accuracy', 'loss'], loc='right')

plt.title('accuracy and loss');

plt.xlabel('epoch');

plt.ylabel('accuracy/loss');
predicts = modelCNN.predict(x=X_test)
def show_test_images(images, cols = 1, true_label = None, pred_label=None):

    n_images = len(images)

    fig = plt.figure()

    for n, (image, label, pred) in enumerate(zip(images, true_label, pred_label)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image.reshape((28,28)))

        a.set_title("{}\n{}".format(label, pred), fontsize=50)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

plt.show()
# samples = np.random.choice(len(X_test), 8)

sample_images = []

sample_labels = []

sample_pred_labels = []

for sample in range(8):

    sample_images.append(X_test[sample])

    sample_pred_labels.append(np.argmax(predicts[sample]))

    sample_labels.append(np.argmax(y_test.iloc[sample]))
show_test_images(sample_images, 2, sample_labels, sample_pred_labels)