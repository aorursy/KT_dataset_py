import numpy as np

import pandas as pd

import tensorflow as tf

import time

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
tf.random.set_random_seed(0)

np.random.seed(0)
train_df = pd.read_csv("../input/fashion-mnist_train.csv")

test_df = pd.read_csv("../input/fashion-mnist_test.csv")


train_df.tail()
test_df.tail()
train_df["label"].value_counts()
test_df["label"].value_counts()
print("Training set contains NaN values:", train_df.isnull().any().any())

print("Test set contains NaN values:", test_df.isnull().any().any())
def transform(x, nrows, ncols):

    # rearrange a numpy array for visualization

    assert nrows * ncols == x.shape[2]

    h, w = x.shape[0], x.shape[1]

    x = np.transpose(x, [2, 0, 1]).reshape(nrows, ncols, h, w)

    x = np.transpose(x, [0, 2, 1, 3]).reshape(nrows*h, ncols*w)

    return x



categories = ["T-shirt", "Trousers", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

sample = train_df.sort_values(by="label", ascending=True).iloc[:, 1:].values

sample = sample.reshape(10, 6000, 784)[:, :10, :].reshape(100, 28, 28)

sample = np.transpose(sample, [1, 2, 0])

fig, ax = plt.subplots(figsize=(10, 10))

ax.imshow(transform(sample, 10, 10), cmap="Greys")

ax.set(yticks=np.arange(15, 280, 28), yticklabels=categories)

ax.get_xaxis().set_visible(False)

plt.tight_layout()
train_df, validate_df = train_test_split(train_df, train_size=50000, random_state=8888, stratify=train_df.iloc[:, 0])
train_y, train_X = train_df.iloc[:, 0].values.astype(np.int32), train_df.iloc[:, 1:].values

val_y, val_X = validate_df.iloc[:, 0].values.astype(np.int32), validate_df.iloc[:, 1:].values

test_y, test_X = test_df.iloc[:, 0].values.astype(np.int32), test_df.iloc[:, 1:].values
data = train_df.iloc[:, 1:].values

print("Min value:", data.min())

print("Max value:", data.max())
def feature_scaling(arr, min_val=0, max_val=255):

    return arr/max_val

train_X, val_X, test_X = [feature_scaling(x).astype(np.float32) for x in (train_X, val_X, test_X)]
class CNN_beta:

    def __init__(self, params):

        print("Building CNN_beta")

        self.conv1 = tf.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)

        self.pool1 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")

        self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

        self.pool2 = tf.layers.MaxPooling2D(pool_size=2, strides=2, padding="same")

        self.flatten = tf.layers.Flatten()

        self.dense = tf.layers.Dense(units=128, activation=tf.nn.relu)       

        self.dropout = tf.layers.Dropout(rate=0.5)

        self.output = tf.layers.Dense(units=10)

    

    def __call__(self, x, training):

        x = tf.reshape(x, [-1, 28, 28, 1])

        conv1 = self.conv1(x)

        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)

        pool2 = self.pool2(conv2)

        flatten = self.flatten(pool2)

        dense = self.dropout(self.dense(flatten), training=training)

        logits = self.output(dense)

        probs = tf.nn.softmax(logits, axis=-1)

        return {"image": x, "conv1": conv1, "conv2": conv2, "pool1": pool1, "pool2": pool2, "flatten": flatten, "dense": dense, "logits": logits, "probs": probs}
def compute_loss(labels, logits):

    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))



def predict(labels, probs):

    preds = tf.argmax(probs, axis=-1)

    _, accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)

    return preds, accuracy



class Model:

    def __init__(self, architecture, params, train_data, val_data):

        forward = architecture(params)

        #For training

        self.train_iter = self.create_iter(train_data, True, params["batch_size"])

        train_X, train_y = self.train_iter.get_next()

        train_logits = forward(train_X, training=True)["logits"]

        self.train_loss = compute_loss(labels=train_y, logits=train_logits)

        optimizer = tf.train.AdamOptimizer(learning_rate=params["lr"])

        self.train_op = optimizer.minimize(self.train_loss)

        #For validation

        self.val_iter = self.create_iter(val_data, False)        

        val_X, val_y = self.val_iter.get_next()

        val_logits = forward(val_X, training=False)["logits"]

        self.val_loss = compute_loss(labels=val_y, logits=val_logits)

        with tf.variable_scope("validate"):

            _, self.val_accuracy = predict(val_y, val_logits)

        #For evaluation

        self.X_holder = tf.placeholder(tf.float32, shape=[None, 784])

        self.y_holder = tf.placeholder(tf.int32, shape=[None])

        output_dict = forward(self.X_holder, training=False)

        test_logits = output_dict["logits"]

        with tf.variable_scope("test"):

            self.test_preds, self.test_accuracy = predict(self.y_holder, test_logits)

        #Extract layers' outputs for visualization

        self.outputs = output_dict

        for k, v in self.outputs.items():

            self.outputs[k] = tf.squeeze(v, axis=0)



    def train_and_validate(self, sess, p=10):

        # train and validate with early stopping.

        # patience is set to 10 by default.

        start = time.time()

        sess.run(tf.global_variables_initializer())

        j = 0

        i = 0

        n_epochs = 0

        best_loss = float("inf")

        train_losses = []

        val_losses = []

        while j < p:

            train_outs = self.train(sess)

            val_outs = self.validate(sess)

            i += 1

            train_loss = train_outs["loss"]

            train_losses.append(train_loss)

            val_loss = val_outs["loss"]

            val_losses.append(val_loss)

            val_accuracy = val_outs["accuracy"]

            print("Epoch {}:\nTraining Loss: {}\nValidation Loss: {}\nValidation Accuracy: {}\n".format(i, train_loss, val_loss, val_accuracy))

            if val_loss < best_loss:

                j = 0

                n_epochs = i

                best_loss = val_loss

                weights = sess.run(tf.trainable_variables())

            else:

                j += 1

        print("Training stopped after {} epochs.\nOptimal number of epochs: {}\nLowest validation loss: {}\n".format(i, n_epochs, best_loss))

        revert_ops = [var.assign(weight) for var, weight in zip(tf.trainable_variables(), weights)]

        sess.run(revert_ops)

        print("Parameters reverted to epoch %d."%n_epochs)

        print("Training time: %.2f sec."%(time.time()-start))

        self.train_losses = np.array(train_losses)

        self.val_losses = np.array(val_losses)

    

    def train(self, sess):

        sess.run(self.train_iter.initializer)

        loss = 0

        i = 0

        while True:

            try:

                b_loss, _ = sess.run([self.train_loss, self.train_op])

                loss += b_loss

                i += 1

            except tf.errors.OutOfRangeError:

                break;

        return {"loss": loss/i}

    

    def validate(self, sess):

        sess.run(self.val_iter.initializer)

        sess.run(tf.variables_initializer(tf.local_variables(scope="validate")))

        loss = 0

        i = 0

        while True:

            try:

                b_loss, accuracy = sess.run([self.val_loss, self.val_accuracy])

                loss += b_loss

                i += 1

            except tf.errors.OutOfRangeError:

                break;

        return {"loss": loss/i, "accuracy": accuracy}

    

    def evaluate(self, sess, X, y):

        sess.run(tf.variables_initializer(tf.local_variables(scope="test")))

        preds = []

        batch_size = 128

        s = 0

        e = batch_size

        while s < y.shape[0]:

            b_preds, accuracy = sess.run([self.test_preds, self.test_accuracy], feed_dict={self.X_holder: X[s:e], self.y_holder: y[s:e]})

            s = e

            e += batch_size

            preds.append(b_preds)

        print("Test Accuracy: {}\n".format(accuracy))

        self.preds = np.concatenate(preds)

    

    def extract_layer_outputs(self, sess, x):

        outputs = sess.run(self.outputs, feed_dict={self.X_holder: x})

        return outputs

    

    def create_iter(self, data, shuffle, batch_size=128):

        dataset = tf.data.Dataset.from_tensor_slices(data)

        dataset = dataset.batch(batch_size)

        if shuffle:

            dataset = dataset.shuffle(100000)

        return tf.data.make_initializable_iterator(dataset)
params = {"lr": 0.001, "batch_size": 128}

model = Model(CNN_beta, params, (train_X, train_y), (val_X, val_y))

sess = tf.Session()

model.train_and_validate(sess, p=10)
def plot_learning_curves(train_losses, val_losses):

    n_epochs = len(train_losses)

    fig, ax = plt.subplots(figsize=(10,10))

    ax.plot(np.arange(n_epochs)+1, train_losses, color="red", label="Training Loss")

    ax.plot(np.arange(n_epochs)+1, val_losses, color="blue", label="Validation Loss")

    ax.set_title("Learning Curves", fontsize=20, pad=20)

    ax.set_xlabel("Epoch", fontsize=15, labelpad=15)

    ax.set_ylabel("Loss", fontsize=15, labelpad=15)

    plt.legend()

    plt.tight_layout()



plot_learning_curves(model.train_losses, model.val_losses)
model.evaluate(sess, test_X, test_y)

sess.close()
class CNN:

    def __init__(self, params):

        print("Building CNN")

        self.conv1a = tf.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation=tf.nn.relu)

        self.conv1b = tf.layers.Conv2D(filters=32, kernel_size=5, padding="same", activation=tf.nn.relu)

        self.pool1 = tf.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")

        self.dropout1 = tf.layers.Dropout(rate=0.5)

        self.conv2 = tf.layers.Conv2D(filters=64, kernel_size=5, padding="same", activation=tf.nn.relu)

        self.pool2 = tf.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")

        self.dropout2 = tf.layers.Dropout(rate=0.5)

        self.flatten = tf.layers.Flatten()

        self.dense = tf.layers.Dense(units=1024, activation=tf.nn.relu)       

        self.dropout = tf.layers.Dropout(rate=0.7)

        self.output = tf.layers.Dense(units=10)

    

    def __call__(self, x, training):

        x = tf.reshape(x, [-1, 28, 28, 1])

        conv1a = self.conv1a(x)

        conv1b = self.conv1b(conv1a)

        pool1 = self.pool1(conv1b)

        conv2 = self.conv2(self.dropout1(pool1, training=training))

        pool2 = self.pool2(conv2)

        flatten = self.flatten(self.dropout2(pool2, training=training))

        dense = self.dropout(self.dense(flatten), training=training)

        logits = self.output(dense)

        probs = tf.nn.softmax(logits, axis=-1)

        return {"image": x, "conv1a": conv1a, "conv1b": conv1b, "conv2": conv2, "pool1": pool1, "pool2": pool2, "flatten": flatten, "dense": dense, "logits": logits, "probs": probs}
tf.reset_default_graph()

params = {"lr": 0.001, "batch_size": 128}

model = Model(CNN, params, (train_X, train_y), (val_X, val_y))

sess = tf.Session()

model.train_and_validate(sess, p=15)
plot_learning_curves(model.train_losses, model.val_losses)
model.evaluate(sess, test_X, test_y)



cm = confusion_matrix(y_true=test_y, y_pred=model.preds)

fig, ax = plt.subplots(figsize=(10,10))

im = ax.imshow(cm)

ax.figure.colorbar(im, ax=ax)

ax.set(xticks=np.arange(10), yticks=np.arange(10), xticklabels=categories, yticklabels=categories)

ax.set_title("Confusion Matrix", fontsize=20, pad=20)

ax.set_xlabel("Predictions", fontsize=15, labelpad=15)

ax.set_ylabel("Ground truth", fontsize=15, labelpad=15)

th = cm.max() / 2.

for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] > th else "white")

plt.tight_layout()
def plot_layers(layers):

    cmap = "Greys"

    fs = 30

    pad = 30

    fig, ax = plt.subplots(ncols=8,figsize=(40, 10))

    ax[0].imshow(np.squeeze(layers["image"]), cmap=cmap)

    ax[0].set_title("Shirt", fontsize=fs, pad=pad)

    ax[1].imshow(transform(layers["conv1a"], 8, 4), cmap=cmap)

    ax[1].set_title("conv1a", fontsize=fs, pad=pad)

    ax[2].imshow(transform(layers["conv1b"], 8, 4), cmap=cmap)

    ax[2].set_title("conv1b", fontsize=fs, pad=pad)

    ax[3].imshow(transform(layers["pool1"], 8, 4), cmap=cmap)

    ax[3].set_title("pool1", fontsize=fs, pad=pad)

    ax[4].imshow(transform(layers["conv2"], 8, 8), cmap=cmap)

    ax[4].set_title("conv2", fontsize=fs, pad=pad)

    ax[5].imshow(transform(layers["pool2"], 8, 8), cmap=cmap)

    ax[5].set_title("pool2", fontsize=fs, pad=pad)

    ax[6].imshow(np.expand_dims(layers["logits"], 1), cmap=cmap)

    ax[6].set_title("logits", fontsize=fs, pad=pad)

    ax[7].barh(np.arange(10)[::-1], layers["probs"], tick_label=categories)

    ax[7].set_title("Probability Distribution", fontsize=fs, pad=pad)

    ax[7].tick_params(labelsize=20)

    for i in range(len(ax)-1):

        ax[i].axis("off")

    plt.tight_layout()
x1 = test_X[(test_y == model.preds) & (test_y == 6)][:1]

x2 = test_X[(test_y != model.preds) & (test_y == 6)][:1]

for x in x1, x2:

    layers = model.extract_layer_outputs(sess, x)

    plot_layers(layers)