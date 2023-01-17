# Import libraries and modules

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

# For reproducibility

np.random.seed(42)



from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Flatten

from keras.layers import Conv2D, MaxPooling2D

from keras.utils import np_utils, to_categorical

from keras.optimizers import Adam



import tensorflow as tf



from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
import os

os.listdir("../input/digit-recognizer")
DATA_PATH = "../input/digit-recognizer/"

digits = pd.read_csv(DATA_PATH + "train.csv")
digits.head()
digits.shape
target = np.array(digits["label"])

data = digits.drop(columns = "label")
def transform_raw_data(data):

    '''returns the scaled and normalized data reshaped to proper 2D format'''

    data = np.array(data)

    shape = (data.shape[0], 28,28, 1)

    X = data.reshape(shape)

    X = X / 255 # rescale

    X = (X - .5).astype("float16")  # normalize (optimization for CNN)

    return X
X = transform_raw_data(data)
X.shape
def show_image(image, ax = plt, title = None):

    '''displays a single image using a given axes'''

    ax.imshow(image.reshape((28,28)), cmap="gray")

    if title:

        ax.set_title(title)

    ax.tick_params(bottom = False, left = False, labelbottom = False, labelleft = False)
def show_images(images, titles = None, ncols = 4, height = 2):

    '''displays a list/array of images in a grid format'''

    nrows = int(np.ceil(len(images)/ncols))

    f, ax = plt.subplots(nrows=nrows,ncols=ncols, figsize=(10,nrows * height))

    ax = ax.flatten()

    for i, image in enumerate(images):

        if titles:

            show_image(image, ax = ax[i], title = titles[i])

        else:

            show_image(image, ax = ax[i], title = None)

    plt.tight_layout()

    plt.show()
show_images(np.asarray(data[:4]).reshape((4,28,28)))
X_train, X_test, y_train, y_test = train_test_split(X, target, stratify = target, test_size=.1)
y_train_cat = np_utils.to_categorical(y_train, num_classes=10)

y_test_cat = np_utils.to_categorical(y_test, num_classes=10)
def graph_loss(history):

    '''graphs training and testing loss given a keras History object'''

    # Check out our train loss and test loss over epochs.

    train_loss = history.history['loss']

    test_loss = history.history['val_loss']

    xticks = np.array(range(len(train_loss)))

    # Set figure size.

    plt.figure(figsize=(12, 8))



    # Generate line plot of training, testing loss over epochs.

    plt.plot(train_loss, label='Training Loss', color='#185fad')

    plt.plot(test_loss, label='Testing Loss', color='orange')



    # Set title

    plt.title('Training and Testing Loss by Epoch', fontsize = 25)

    plt.xlabel('Epoch', fontsize = 18)

    plt.ylabel('Categorical Crossentropy', fontsize = 18)

    plt.xticks(xticks, xticks+1)



    plt.legend(fontsize = 18);
# setup model

model = Sequential([



    Conv2D(32, input_shape = (28,28, 1), kernel_size = 5, activation="relu", padding="same"),

    Conv2D(32, kernel_size = 5, activation="relu", padding = 'same'),

    MaxPooling2D((2,2)),

    Dropout(.25),

    

    Conv2D(64, kernel_size = 3, activation="relu", padding = 'same'),

    Conv2D(64, kernel_size = 3, activation="relu", padding = 'same'),

    MaxPooling2D((2,2), strides=(2,2)),

    Dropout(.25),



    Flatten(),

    Dense(256, activation="relu"),

    Dropout(.5),

    

    Dense(64, activation="relu"),

    Dropout(.5),



    Dense(10, activation="softmax"),

])
# compile model

model.compile(

    loss = "categorical_crossentropy",

    optimizer = "adam", # Adam(lr = .0001, decay= 1e-5),

    metrics = ["acc"]

)
# fit model

history = model.fit(

    X_train,

    y_train_cat,

    validation_data=(X_test, y_test_cat),

    epochs = 50,

    batch_size= 64

)
graph_loss(history)
preds = model.predict_classes(X_test)
results = pd.DataFrame({"actual":y_test,"pred":preds,"is_correct":y_test == preds})

errors = results[results["is_correct"] == False]

errors.head()
results["is_correct"].value_counts(normalize=True)
titles = ["Actual:{}\nPred:{}".format(act,errors["pred"].iloc[i]) for i,act in enumerate(errors["actual"][:12])]
show_images(((X_test + .5) * 255)[errors.index][:12].astype(int), titles = titles, ncols = 6, height=3)
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()
kaggle_X = transform_raw_data(test)



kaggle_X.shape
kaggle_preds = model.predict_classes(kaggle_X)
final = pd.DataFrame({"ImageId":test.index + 1,"Label":kaggle_preds})



final.head()
final.to_csv("submission.csv", index = False)
digits = pd.read_csv(DATA_PATH + "train.csv")
digits.head()
data = np.asarray(digits.drop(columns = "label"))

target = to_categorical(digits["label"], num_classes = 10) # turns into matrix for us!
data.shape
target.shape
def transform_raw_data_TF(data):    

    # rescales and normalizes (optimization for FFNN)

    return (data / 255 - .5).astype("float32")
data = transform_raw_data_TF(data)
X_train, X_test, y_train, y_test = train_test_split(data, target, random_state = 42)
tf.reset_default_graph()



### START OF NETWORK ###

X_scaffold = tf.placeholder(dtype = tf.float32, shape = (None, X_train.shape[1]))

y_scaffold = tf.placeholder(dtype = tf.float32, shape = (None, y_train.shape[1]))



h1 = tf.layers.dense(X_scaffold, 128, activation = tf.nn.relu)



# DROPOUT CAUSING DIFFERENT TEST OUTPUT, NEED TO TURN OFF DURING INFERENCING

prob = tf.placeholder_with_default(0.0, shape=())



d1 = tf.layers.dropout(h1, rate = prob)

h2 = tf.layers.dense(d1, 64, activation = tf.nn.relu)

d2 = tf.layers.dropout(h2, rate = prob)



y_hat = tf.layers.dense(d2, y_train.shape[1], activation = None)



ouput = tf.nn.softmax(y_hat)

### END OF NETWORK ###



loss = tf.losses.softmax_cross_entropy(y_scaffold, y_hat)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

training_epoch = optimizer.minimize(loss)



saver = tf.train.Saver()
init = tf.global_variables_initializer()

n_epochs = 500



sess = tf.Session()

with sess:

    sess.run(init)

    train_loss = []

    test_loss  = []

    for epoch in range(n_epochs):

        sess.run(training_epoch, feed_dict={

            X_scaffold : X_train, 

            y_scaffold : y_train,

            prob       : .1     # DROPOUT CAUSING DIFFERENT TEST OUTPUT, NEED TO TURN OFF DURING INFERENCING

        }) 

        train_loss.append(sess.run(loss, feed_dict={X_scaffold : X_train, y_scaffold : y_train}))

        test_loss.append(sess.run(loss, feed_dict={X_scaffold: X_test, y_scaffold: y_test}))

    pred = sess.run(y_hat, feed_dict={X_scaffold: X_test})

    saver.save(sess, './sess.ckpt')   
def graph_loss(train_loss, test_loss):

    '''Graphing function to visualize loss'''

    xticks = np.array(range(len(train_loss)))

    # Set figure size.

    plt.figure(figsize=(12, 8))



    # Generate line plot of training, testing loss over epochs.

    plt.plot(train_loss, label='Training Loss', color='#185fad')

    plt.plot(test_loss, label='Testing Loss', color='orange')



    # Set title

    plt.title('Training and Testing Loss by Epoch', fontsize = 25)

    plt.xlabel('Epoch', fontsize = 18)

    plt.ylabel('Categorical Crossentropy', fontsize = 18)

    plt.xticks(xticks[::10], (xticks+1)[::10])



    plt.legend(fontsize = 18);
graph_loss(train_loss, test_loss)
y_pred = pred.argmax(axis = 1)

y_true = y_test.argmax(axis = 1)

cm = confusion_matrix(y_true, y_pred)

pd.DataFrame(cm,

             index=["Actual {}".format(num+1) for num in range(10)],

             columns=["Pred. {}".format(num+1) for num in range(10)])
np.mean(y_true == y_pred)
test = pd.read_csv(DATA_PATH + "test.csv")
test.head()
kaggle_X = transform_raw_data_TF(np.asarray(test))



kaggle_X.shape
with tf.Session() as sess:

    saver.restore(sess, './sess.ckpt')

    kaggle_preds = sess.run(y_hat, feed_dict={X_scaffold: kaggle_X})
kaggle_preds_final = kaggle_preds.argmax(axis = 1)

kaggle_preds_final
# sneak peek at kaggle answers

show_images(np.asarray(test).reshape((-1, 28,28,1))[:8], titles = ["Pred: {}".format(i) for i in kaggle_preds_final[:8]])
final = pd.DataFrame({"ImageId":test.index + 1,"Label":kaggle_preds_final})



final.head()
final.to_csv("submission_tf.csv", index = False)