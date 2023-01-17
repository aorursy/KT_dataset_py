# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

sns.set(style="darkgrid")



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot figures

%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14

plt.rcParams['xtick.labelsize'] = 12

plt.rcParams['ytick.labelsize'] = 12



import warnings

warnings.filterwarnings(action="ignore")
def load_digit_data(filename, house_path):

    csv_path = os.path.join(house_path, filename)

    return pd.read_csv(csv_path)
train_data = load_digit_data('train.csv',"../input")
digits = train_data.copy()
digits.head()
X_train = digits.drop(['label'], axis=1).values

y_train = digits['label']
X_train.shape
y_train.shape
X_valid, X_train = X_train[:5000], X_train[5000:]

y_valid, y_train = y_train[:5000], y_train[5000:]
y_train[:5]
some_digit = X_train[3]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,

           interpolation="nearest")

plt.axis("off")

plt.show()
# EXTRA

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = matplotlib.cm.binary, **options)

    plt.axis("off")
plt.figure(figsize=(9,9))

example_images = np.r_[X_train[:12000:600], X_train[13000:30600:600], X_train[30600:60000:590]]

plot_digits(example_images, images_per_row=10)

plt.show()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

X_valid_scaled = scaler.transform(X_valid.astype(np.float64))
import tensorflow as tf

tf.__version__
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.exceptions import NotFittedError



class CNNClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, optimizer_class=tf.train.AdamOptimizer,

                 batch_size=20, activation=tf.nn.relu, dropout_rate=None, random_state=None):

        """Initialize the CNNClassifier by simply storing all the hyperparameters."""

        self.optimizer_class = optimizer_class

        self.batch_size = batch_size

        self.activation = activation

        self.dropout_rate = dropout_rate

        self.random_state = random_state

        self._session = None



    def _cnn(self, inputs):

        """Build the hidden layers, with dropout."""

        

        conv1 = tf.layers.conv2d(inputs, filters=32, kernel_size=5, strides=1, padding="SAME", activation=self.activation, name="conv1")

        conv2 = tf.layers.conv2d(conv1, filters=64, kernel_size=5, strides=1, padding="SAME", activation=self.activation, name="conv2")

        

        pool1_fmaps = 64

        

        with tf.name_scope("pool1"):

            pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

            pool1_flat = tf.reshape(pool1, shape=[-1, pool1_fmaps * 14 * 14])

            if self.dropout_rate:

                pool1_flat = tf.layers.dropout(pool1_flat, self.dropout_rate, training=self._training)

                

        n_fc1 = 128

                

        with tf.name_scope("fc1"):

            fc1 = tf.layers.dense(pool1_flat, n_fc1, activation=tf.nn.relu, name="fc1")

            if self.dropout_rate:

                fc1 = tf.layers.dropout(fc1, self.dropout_rate, training=self._training)

            

        return fc1



    def _build_graph(self, n_outputs):

        """Build the graph"""

        if self.random_state is not None:

            tf.set_random_seed(self.random_state)

            np.random.seed(self.random_state)

        

        height = 28

        width = 28

        channels = 1

        n_inputs = height * width

        

        if self.dropout_rate:

            self._training = tf.placeholder_with_default(False, shape=[], name="training")

        else:

            self._training = None



        with tf.name_scope("inputs"):

            X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")

            X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])

            y = tf.placeholder(tf.int32, shape=[None], name="y")



        cnn_outputs = self._cnn(X_reshaped)

        

        with tf.name_scope("output"):

            logits = tf.layers.dense(cnn_outputs, n_outputs, name="output")

            Y_proba = tf.nn.softmax(logits, name="Y_proba")

            

        with tf.name_scope("train"):

            xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)

            loss = tf.reduce_mean(xentropy)

            optimizer = self.optimizer_class()

            training_op = optimizer.minimize(loss)

            

        with tf.name_scope("eval"):

            correct = tf.nn.in_top_k(logits, y, 1)

            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        

        with tf.name_scope("init_and_save"):

            init = tf.global_variables_initializer()

            saver = tf.train.Saver()



        # Make the important operations available easily through instance variables

        self._X, self._y = X, y

        self._Y_proba, self._loss = Y_proba, loss

        self._training_op, self._accuracy = training_op, accuracy

        self._init, self._saver = init, saver



    def close_session(self):

        if self._session:

            self._session.close()



    def _get_model_params(self):

        """Get all variable values (used for early stopping, faster than saving to disk)"""

        with self._graph.as_default():

            gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        return {gvar.op.name: value for gvar, value in zip(gvars, self._session.run(gvars))}



    def _restore_model_params(self, model_params):

        """Set all variables to the given values (for early stopping, faster than loading from disk)"""

        gvar_names = list(model_params.keys())

        assign_ops = {gvar_name: self._graph.get_operation_by_name(gvar_name + "/Assign")

                      for gvar_name in gvar_names}

        init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}

        feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}

        self._session.run(assign_ops, feed_dict=feed_dict)



    def fit(self, X, y, n_epochs=5, X_valid=None, y_valid=None):

        """Fit the model to the training set. If X_valid and y_valid are provided, use early stopping."""

        self.close_session()



        # infer n_outputs from the training set.

        self.classes_ = np.unique(y)

        n_outputs = len(self.classes_)

        

        # Translate the labels vector to a vector of sorted class indices, containing

        # integers from 0 to n_outputs - 1.

        # For example, if y is equal to [8, 8, 9, 5, 7, 6, 6, 6], then the sorted class

        # labels (self.classes_) will be equal to [5, 6, 7, 8, 9], and the labels vector

        # will be translated to [3, 3, 4, 0, 2, 1, 1, 1]

        self.class_to_index_ = {label: index for index, label in enumerate(self.classes_)}

        y = np.array([self.class_to_index_[label] for label in y], dtype=np.int32)

        

        self._graph = tf.Graph()

        with self._graph.as_default():

            self._build_graph(n_outputs)



        # needed in case of early stopping

        best_loss_val = np.infty

        check_interval = 500

        checks_since_last_progress = 0

        max_checks_without_progress = 10

        best_model_params = None

        

        # Now train the model!

        self._session = tf.Session(graph=self._graph)

        with self._session.as_default() as sess:

            self._init.run()

            for epoch in range(n_epochs):

                rnd_idx = np.random.permutation(len(X))

                iteration = 0

                for rnd_indices in np.array_split(rnd_idx, len(X) // self.batch_size):

                    X_batch, y_batch = X[rnd_indices], y[rnd_indices]

                    feed_dict = {self._X: X_batch, self._y: y_batch}

                    if self._training is not None:

                        feed_dict[self._training] = True

                    sess.run(self._training_op, feed_dict=feed_dict)

                    

                    if X_valid is not None and y_valid is not None:

                        if iteration % check_interval == 0:

                            loss_val = self._loss.eval(feed_dict={self._X: X_valid, self._y: y_valid})

                            if loss_val < best_loss_val:

                                best_loss_val = loss_val

                                checks_since_last_progress = 0

                                best_model_params = self._get_model_params()

                            else:

                                checks_since_last_progress += 1

                    iteration += 1

                    

                if X_valid is not None and y_valid is not None:

                    acc_train = self._accuracy.eval(feed_dict={self._X: X_batch, self._y: y_batch})

                    acc_val = self._accuracy.eval(feed_dict={self._X: X_valid, self._y: y_valid})

                    

                    print("{}\tTrain Accuracy: {:.2f}\tValidation Accuracy: {:.2f}%\tBest loss: {:.6f}".format(

                        epoch, acc_train * 100, acc_val * 100, best_loss_val))

                    if checks_since_last_progress > max_checks_without_progress:

                        print("Early stopping!")

                        break

                else:

                    loss_train, acc_train = sess.run([self._loss, self._accuracy],

                                                     feed_dict={self._X: X_batch, self._y: y_batch})

                    print("{}\tLast training batch loss: {:.6f}\tAccuracy: {:.2f}%".format(

                        epoch, loss_train, acc_train * 100))

                    

            if best_model_params:

                self._restore_model_params(best_model_params)

            return self



    def predict_proba(self, X):

        if not self._session:

            raise NotFittedError("This %s instance is not fitted yet" % self.__class__.__name__)

        with self._session.as_default() as sess:

            return self._Y_proba.eval(feed_dict={self._X: X})



    def predict(self, X):

        class_indices = np.argmax(self.predict_proba(X), axis=1)

        return np.array([[self.classes_[class_index]] for class_index in class_indices], np.int32)



    def save(self, path):

        self._saver.save(self._session, path)
cnn_clf = CNNClassifier(random_state=42, batch_size=100, dropout_rate=0.15)

cnn_clf.fit(X_train_scaled, y_train, n_epochs=1000, X_valid=X_valid_scaled, y_valid=y_valid)
test_data = load_digit_data('test.csv','../input')

test_data.head()
len(test_data)
y_pred = []

for chunk in np.array_split(test_data, 1000):

    X_test = chunk.values

    X_test_scaled = scaler.transform(X_test.astype(np.float64))

    y_pred.extend(cnn_clf.predict(X_test_scaled))



y_pred = np.array(y_pred)
print(y_pred.shape)

y_pred
test_data['Label'] = y_pred.ravel()

test_data['ImageId']= np.arange(1, len(test_data)+1)

test_data[['ImageId','Label']].head(10)
some_digit = test_data.drop(['ImageId','Label'], axis=1).values[4]

some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,

           interpolation="nearest")

plt.axis("off")

plt.show()
test_data[['ImageId','Label']].to_csv('submission.csv', index=False)