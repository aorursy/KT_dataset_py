import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.attacks import SaliencyMapMethod
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import other_classes, set_log_level
from cleverhans.utils import pair_visual, grid_visual, AccuracyReport
from cleverhans.utils_tf import model_eval, model_argmax
from cleverhans.train import train
from cleverhans_tutorials.tutorial_models import ModelBasicCNN
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import ProjectedGradientDescent
from cleverhans.attacks import BasicIterativeMethod
import keras

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
# define parameters
train_start=0
train_end=60000
test_start=0
test_end=10000
nb_epochs=6
batch_size=128
source_samples=10
learning_rate=.001

train_params = {
  'nb_epochs': nb_epochs,
  'batch_size': batch_size,
  'learning_rate': learning_rate
}

def get_bim_params(clip_min=0.,clip_max=1., eps_iter = 0.01, nb_iter= 100):
    bim_params = {'eps_iter': eps_iter,
                  'nb_iter': nb_iter,
                  'clip_min': clip_min,
                  'clip_max': clip_max}

    return bim_params
tf.reset_default_graph()
# Object used to keep track of (and return) key accuracies
report = AccuracyReport()

# Set TF random seed to improve reproducibility
tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
sess = tf.Session()
print("Created TensorFlow session.")

set_log_level(logging.DEBUG)

# Get MNIST test data
mnist = MNIST(train_start=train_start, train_end=train_end,
            test_start=test_start, test_end=test_end)
x_train, y_train = mnist.get_set('train')
x_test, y_test = mnist.get_set('test')

# Obtain Image Parameters
img_rows, img_cols, nchannels = x_train.shape[1:4]
nb_classes = y_train.shape[1]

# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                    nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes))

nb_filters = 64
# Define TF model graph
model = ModelBasicCNN('model1', nb_classes, nb_filters)
preds = model.get_logits(x)
loss = CrossEntropy(model, smoothing=0.1)
print("Defined TensorFlow model graph.")


sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss, x_train, y_train, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': batch_size}
accuracy = model_eval(sess, x, y, preds, x_test, y_test, args=eval_params)
assert x_test.shape[0] == test_end - test_start, x_test.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
report.clean_train_clean_eval = accuracy
bim_op = BasicIterativeMethod(model, sess=sess)
advs = bim_op.generate_np(x_test[:2000], **get_bim_params(nb_iter=20, eps_iter=0.01))
advs_bis = bim_op.generate_np(x_test[2000:4000], **get_bim_params(nb_iter=30, eps_iter=0.01))
pred_class_adv=model_argmax(sess, x, preds, advs)
pred_class_orig=model_argmax(sess, x, preds, x_test[:2000])
res_advs = np.array(pred_class_adv ==np.argmax(y_test[:2000], axis=1)).astype(int)
accuracy_advs=res_advs.sum()/len(res_advs)
accuracy_advs
y_train.shape
X_advs=np.append(x_train[:20000], advs,axis=0)
y_advs=keras.utils.to_categorical(np.append(np.argmax(y_train[:20000], axis=1), 10*np.ones((2000,1))))
X_train_advs, X_test_advs, y_train_advs, y_test_advs = train_test_split( X_advs, y_advs, test_size=0.33, random_state=42)
y_train_advs.shape
tf.reset_default_graph()

tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
sess = tf.Session()

# Obtain Image Parameters
nb_classes_ = y_train.shape[1]+1


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                    nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes_))

nb_filters = 64
# Define TF model graph
model2 = ModelBasicCNN('model1', 11, nb_filters)
preds2 = model2.get_logits(x)
loss2 = CrossEntropy(model2, smoothing=0.1)
print("Defined TensorFlow model graph.")


sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss2, X_train_advs, y_train_advs, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': batch_size}
accuracy = model_eval(sess, x, y, preds2, X_test_advs, y_test_advs, args=eval_params)
#assert x_test.shape[0] == test_end - test_start, X_test_advs.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
report.clean_train_clean_eval = accuracy
from sklearn.metrics import f1_score
#f1_score(y_true, y_pred, average='weighted')  
#advs_on_test=X_test_advs[np.argmax(y_test_advs,axis=1)==10]
pred_class_adv=model_argmax(sess, x, preds2, X_test_advs)
f1_score(pred_class_adv,np.argmax(y_test_advs,axis=1), average="weighted")
print('Confusion Matrix')
print(confusion_matrix(pred_class_adv, np.argmax(y_test_advs,axis=1)))
print('Classification Report')
print(classification_report(pred_class_adv, np.argmax(y_test_advs,axis=1), labels=range(11)))
pred_class_advbis=model_argmax(sess, x, preds2, advs_bis)
y_test_advs_bis=np.zeros((2000,11))
y_test_advs_bis[:,10]=1
accuracy_on_advs = model_eval(sess, x, y, preds2, advs_bis, y_test_advs_bis, args=eval_params)
accuracy_on_orig = model_eval(sess, x, y, preds2, x_test[2000:4000], np.append(y_test[2000:4000],np.zeros((2000,1)),axis=1), args=eval_params)
print("accuracy on advs {}".format(accuracy_on_advs))
print("accuracy on origs {}".format(accuracy_on_orig))
print('Confusion Matrix')
print(confusion_matrix(pred_class_advbis, 10*np.ones((2000,1))))
print('Classification Report')
print(classification_report(pred_class_advbis,10*np.ones((2000,1)), labels=range(11)))
#X_advs=np.append(x_train[:20000], advs,axis=0)
y_advs2=keras.utils.to_categorical(np.append(np.zeros((20000,1)), np.ones((2000,1))))
X_train_advs2, X_test_advs2, y_train_advs2, y_test_advs2 = train_test_split( X_advs, y_advs2, test_size=0.33, random_state=42)

tf.reset_default_graph()

tf.set_random_seed(1234)

# Create TF session and set as Keras backend session
sess = tf.Session()

# Obtain Image Parameters
nb_classes_ = y_train_advs2.shape[1]


# Define input TF placeholder
x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                    nchannels))
y = tf.placeholder(tf.float32, shape=(None, nb_classes_))

nb_filters = 64
# Define TF model graph
model3 = ModelBasicCNN('model1', nb_classes_, nb_filters)
preds3 = model3.get_logits(x)
loss3 = CrossEntropy(model3, smoothing=0.1)
print("Defined TensorFlow model graph.")


sess.run(tf.global_variables_initializer())
rng = np.random.RandomState([2017, 8, 30])
train(sess, loss3, X_train_advs2, y_train_advs2, args=train_params, rng=rng)

# Evaluate the accuracy of the MNIST model on legitimate test examples
eval_params = {'batch_size': batch_size}
accuracy = model_eval(sess, x, y, preds3, X_test_advs2, y_test_advs2, args=eval_params)
#assert x_test.shape[0] == test_end - test_start, X_test_advs.shape
print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
report.clean_train_clean_eval = accuracy

