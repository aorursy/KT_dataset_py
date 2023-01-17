import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Dense,Flatten,Dropout

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Conv2D,MaxPooling2D,Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.utils import to_categorical,plot_model



from IPython.display import Image



import time

import sys

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

df_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")
df_submission
print("Train File")

df_train.head()
print("Test File")

df_test.head()
print("Train File", df_train.shape)

print("Test File", df_test.shape)
print("Train File", df_train.isnull().any().sum())

print("Test File", df_test.isnull().any().sum())
print("Train File")

df_train.describe()
print("Test File")

df_test.describe()
sns.countplot(df_train['label'])

plt.show()
y_train = df_train['label'].astype('float32')

X_train = df_train.drop(['label'], axis=1).astype('int32')

X_test = df_test.astype('float32')

X_train.shape, y_train.shape, X_test.shape
X = np.array(X_train).reshape(df_train.shape[0],28,28,1)

Y = np.array(y_train).reshape(df_train.shape[0],1)

f, axes = plt.subplots(2, 10, sharey=True,figsize=(20,20))

for i,ax in enumerate(axes.flat):

    ax.axis('off')

    ax.imshow(X[i,:,:,0],cmap="gray")
X_train = X_train/255

X_test = X_test/255
X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

X_train.shape , X_test.shape
y_train = to_categorical(y_train, num_classes = 10)

y_train.shape
x = tf.Variable(5.0)



with tf.GradientTape() as tape:

    y = x**3
# dy = 3x * dx

dy_dx = tape.gradient(y, x)

dy_dx.numpy()
# tf.GradientTape works on any tensor:



w = tf.Variable(tf.random.normal((3, 2)), name='w')

b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')

x = [[1., 2., 3.]]



with tf.GradientTape(persistent=True) as tape:

    y = x @ w + b

    loss = tf.reduce_mean(y**2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(w.shape)

print(dl_dw.shape)
x = tf.constant(3.0)

with tf.GradientTape() as g:

    g.watch(x)

    with tf.GradientTape() as gg:

        gg.watch(x)

        y = x * x

        dy_dx = gg.gradient(y, x)

        print(dy_dx.numpy())

    d2y_dx2 = g.gradient(dy_dx, x)

    print(d2y_dx2.numpy())
def build_model(width, height, depth, classes):

    # initialize the input shape and channels dimension to be

    # "channels last" ordering

    inputShape = (height, width, depth)

    chanDim = -1

    

    # build the model using Keras' Sequential API

    model = Sequential([

        # CONV => RELU => BN => POOL layer set

        Conv2D(16, (3, 3), padding="same", input_shape=inputShape),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        MaxPooling2D(pool_size=(2, 2)),

        

        # (CONV => RELU => BN) * 2 => POOL layer set

        Conv2D(32, (3, 3), padding="same"),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        Conv2D(32, (3, 3), padding="same"),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        MaxPooling2D(pool_size=(2, 2)),

        

        # (CONV => RELU => BN) * 3 => POOL layer set

        Conv2D(64, (3, 3), padding="same"),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        Conv2D(64, (3, 3), padding="same"),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        Conv2D(64, (3, 3), padding="same"),

        Activation("relu"),

        BatchNormalization(axis=chanDim),

        MaxPooling2D(pool_size=(2, 2)),

        

        # first (and only) set of FC => RELU layers

        Flatten(),

        Dense(256),

        Activation("relu"),

        BatchNormalization(),

        Dropout(0.5),

        # softmax classifier

        Dense(classes),

        Activation("softmax")

    ])



    # return the built model to the calling function

    return model
def step(X, y):

    # keep track of our gradients

    with tf.GradientTape() as tape:

        # make a prediction using the model and then calculate the loss

        pred = model(X)

        loss = categorical_crossentropy(y, pred)

    

    # calculate the gradients using our tape and then update the model weights

    grads = tape.gradient(loss, model.trainable_variables)

    opt.apply_gradients(zip(grads, model.trainable_variables))
# initialize the number of epochs to train for, batch size, and initial learning rate

EPOCHS = 50

BS = 32

INIT_LR = 1e-3





model = build_model(28, 28, 1, 10)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)



# compute the number of batch updates per epoch

numUpdates = int(X_train.shape[0] / BS)



# loop over the number of epochs

for epoch in range(0, EPOCHS):

    # show the current epoch number

    print("[INFO] starting epoch {}/{}...".format(epoch + 1, EPOCHS), end="")

    sys.stdout.flush()

    epochStart = time.time()



    # loop over the data in batch size increments

    for i in range(0, numUpdates):

        # determine starting and ending slice indexes for the current batch

        start = i * BS

        end = start + BS



        # take a step

        step(X_train[start:end], y_train[start:end])



    # show timing information for the epoch

    epochEnd = time.time()

    elapsed = (epochEnd - epochStart) / 60.0

    print("took {:.4} minutes".format(elapsed))
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

Image("model.png")
# in order to calculate accuracy using Keras' functions we first need to compile the model

model.compile(optimizer=opt, loss=categorical_crossentropy,metrics=["acc"])



# now that the model is compiled we can compute the accuracy

# (loss, acc) = model.evaluate(X_test, y_test)

# print("[INFO] test accuracy: {:.4f}".format(acc))
y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)

my_submission = pd.DataFrame({'ImageId': list(range(1, len(y_pred)+1)), 'Label': y_pred})

my_submission.to_csv('submission.csv', index=False)