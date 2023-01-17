# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
from skimage import io, transform, exposure
from tqdm import tqdm
from sklearn import utils
from sklearn.metrics import roc_curve, auc

plt.style.use("seaborn")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# Any results you write to the current directory are saved as output.
train_path = "../input/chest-xray-pneumonia/chest_xray/train/"
validation_path = "../input/chest-xray-pneumonia/chest_xray/val/"
normal_path = "NORMAL/"
pneumonia_path = "PNEUMONIA/"
normal_trainset = [train_path+normal_path+element for element in os.listdir(train_path+normal_path)]
pneumonia_trainset = [train_path+pneumonia_path+element for element in os.listdir(train_path+pneumonia_path)]
normal_validationset = [validation_path+normal_path+element for element in os.listdir(validation_path+normal_path)]
pneumonia_validationset = [validation_path+pneumonia_path+element for element in os.listdir(validation_path+pneumonia_path)]
plt.figure(figsize=[16, 8])
for i in range(3):
    plt.subplot(2,3,i+1)
    im = io.imread(normal_trainset[np.random.randint(0, len(normal_trainset))])
    plt.imshow(im)
    plt.title("NORMAL {}".format(i+1))
for i in range(3):
    plt.subplot(2,3,(3+i)+1)
    im = io.imread(pneumonia_trainset[np.random.randint(0, len(pneumonia_trainset))])
    plt.imshow(im)
    plt.title("PNEUMONIA {}".format(i+1))
plt.subplots_adjust(wspace=0, hspace=0.5)
plt.figure(figsize=[12, 4])
plt.subplot(1, 2, 1)
plt.bar([0, 1], [len(normal_trainset), len(pneumonia_trainset)])
plt.xticks([0, 1],["NORMAL", "PNEUMONIA"])
plt.title("Traininng set")
plt.subplot(1, 2, 2)
plt.bar([0, 1], [len(normal_validationset), len(pneumonia_validationset)])
plt.xticks([0, 1],["NORMAL", "PNEUMONIA"])
plt.title("Validation set")
def random_rotation(img):
    random_int = np.random.randint(-15, 15)
    img = transform.rotate(img, random_int)
    return img

def exposure_image(img):
    v_min, v_max = np.percentile(img, (1, 98))
    return exposure.rescale_intensity(img, in_range=(v_min, v_max))

def flip_image(img):
    return img[:, ::-1]
img = io.imread(normal_trainset[0])
plt.figure(figsize=[16, 12])
plt.subplot(1, 4, 1)
plt.imshow(img)
plt.title("Original Image")
plt.subplot(1, 4, 2)
plt.imshow(random_rotation(img))
plt.title("Rotated Image")
plt.subplot(1, 4, 3)
plt.imshow(exposure_image(img))
plt.title("Exposure Image")
plt.subplot(1, 4, 4)
plt.imshow(flip_image(img))
plt.title("Flip image")
def create_numpy_array(input_list, length, resize_shape, label, augmentation_function=None):
    _X = np.zeros((length, resize_shape[0], resize_shape[1]))
    if label == 'NORMAL':
        _y = np.zeros(length)
    elif label == 'PNEUMONIA':
        _y = np.ones(length)
    else:
        raise Exception("not valid label")
    _index = 0
    for _element in tqdm(input_list):
        _img = io.imread(_element, as_gray=True)
        if augmentation_function is not None:
            _img = augmentation_function(_img)
        _img = transform.resize(_img, resize_shape)
        _X[_index, :, :] = _img
        _index = _index + 1
    return _X, _y
X_1, y_1 = create_numpy_array(normal_trainset, len(normal_trainset), (256, 256), "NORMAL")
# select random images from 'Normal' training set
# we want 500 new images selected in the 'Normal' training length
random_index = np.random.randint(0, len(X_1), 500)
X_2, y_2 = create_numpy_array([normal_trainset[i] for i in random_index], 500, (256, 256), "NORMAL", augmentation_function=random_rotation)
random_index = np.random.randint(0, len(X_1), 500)
X_3, y_3 = create_numpy_array([normal_trainset[i] for i in random_index], 500, (256, 256), "NORMAL", augmentation_function=exposure_image)
random_index = np.random.randint(0, len(X_1), 500)
X_4, y_4 = create_numpy_array([normal_trainset[i] for i in random_index], 500, (256, 256), "NORMAL", augmentation_function=flip_image)
# now we want 'Pneumonia' training set with length similar the the sum of the previous ones.
X_5, y_5 = create_numpy_array(pneumonia_trainset[:len(normal_trainset)+500+500+500], len(normal_trainset)+500+500+500, (256, 256), "PNEUMONIA")
# contatenate
X_training = np.concatenate((X_1, X_2, X_3, X_4, X_5))
y_training = np.concatenate((y_1, y_2, y_3, y_4, y_5))
#shuffle
X_training, y_training = utils.shuffle(X_training, y_training)
# same for validation
X_v1, y_v1 = create_numpy_array(normal_validationset, len(normal_validationset), (256, 256), "NORMAL")
X_v2, y_v2 = create_numpy_array(pneumonia_validationset, len(pneumonia_validationset), (256, 256), "PNEUMONIA")
X_validation = np.concatenate((X_v1, X_v2))
y_validation = np.concatenate((y_v1, y_v2))
X_validation, y_validation = utils.shuffle(X_validation, y_validation)
tf.reset_default_graph()
X = tf.placeholder(dtype = tf.float32, shape= [None, 256, 256, 1], name= 'input')
y = tf.placeholder(dtype= tf.float32, shape=[None, 1], name="prediction")
w1 = tf.get_variable(name="w1", shape=[3, 3, 1, 16], initializer=tf.glorot_normal_initializer(), trainable=True)
b1 = tf.get_variable(name="b1", shape=[16], initializer=tf.constant_initializer(), trainable=True)
pre_1 = tf.nn.conv2d(input= X, filter=w1, strides=[1, 2, 2, 1], padding='SAME')
post_1 = tf.nn.bias_add(tf.nn.leaky_relu(pre_1), b1)
print(post_1.shape)
w2 = tf.get_variable(name="w2", shape=[3, 3, 16, 32], initializer=tf.glorot_normal_initializer(), trainable=True)
b2 = tf.get_variable(name="b2", shape=[32], initializer=tf.constant_initializer(), trainable=True)
pre_2 = tf.nn.conv2d(input= post_1, filter=w2, strides=[1, 2, 2, 1], padding='SAME')
post_2 = tf.nn.bias_add(tf.nn.leaky_relu(pre_2), b2)
print(post_2.shape)
w3 = tf.get_variable(name="w3", shape=[3, 3, 32, 64], initializer=tf.glorot_normal_initializer(), trainable=True)
b3 = tf.get_variable(name="b3", shape=[64], initializer=tf.constant_initializer(), trainable=True)
pre_3 = tf.nn.conv2d(input= post_2, filter=w3, strides=[1, 2, 2, 1], padding='SAME')
post_3 = tf.nn.bias_add(tf.nn.leaky_relu(pre_3), b3)
print(post_3.shape)
w4 = tf.get_variable(name="w4", shape=[3, 3, 64, 64], initializer=tf.glorot_normal_initializer(), trainable=True)
b4 = tf.get_variable(name="b4", shape=[64], initializer=tf.constant_initializer(), trainable=True)
pre_4 = tf.nn.conv2d(input= post_3, filter=w4, strides=[1, 2, 2, 1], padding='SAME')
post_4 = tf.nn.bias_add(tf.nn.leaky_relu(pre_4), b4)
print(post_4.shape)
flattened = tf.reshape(post_4, [-1, 16*16*64])
flattened_shape = flattened.shape
wfc1 = tf.get_variable(name="wfc1", shape = [flattened_shape[1], 1000], initializer=tf.glorot_normal_initializer(), trainable=True)
bfc1 = tf.get_variable(name="bfc1", shape=[1000], initializer=tf.constant_initializer(), trainable=True)
pre_fc1 = tf.matmul(flattened, wfc1)
post_fc1 = tf.nn.bias_add(tf.nn.leaky_relu(pre_fc1), bfc1)
print(post_fc1.shape)
wfc2 = tf.get_variable(name="wfc2", shape = [1000, 100], initializer=tf.glorot_normal_initializer(), trainable=True)
bfc2 = tf.get_variable(name="bfc2", shape=[100], initializer=tf.constant_initializer(), trainable=True)
pre_fc2 = tf.matmul(post_fc1, wfc2)
post_fc2 = tf.nn.bias_add(tf.nn.leaky_relu(pre_fc2), bfc2)
print(post_fc2.shape)
wfc3 = tf.get_variable(name="wfc3", shape = [100, 1], initializer=tf.glorot_normal_initializer(), trainable=True)
bfc3 = tf.get_variable(name="bfc3", shape=[1], initializer=tf.constant_initializer(), trainable=True)
logits = tf.matmul(post_fc2, wfc3) + bfc3
print(logits.shape)
outputs = tf.nn.sigmoid(logits)
learning_rate = 1e-5
batch_size = 16
epochs = 5
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
correct_pred = tf.equal(tf.round(outputs), y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
if not os.path.exists("CHECKPOINT"):
    os.makedirs("CHECKPOINT")
len(X_training)//batch_size
print(355*16)
loss_trend = []
accuracy_trend = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 1
    for e in range(epochs):
        loss_trend_per_epoch = []
        for batch_len in range(len(X_training)//batch_size):
            try:
                start = batch_len*batch_size
                end = (batch_len+1)*batch_size
                x_t = X_training[start:end]
                y_t = y_training[start:end]
                x_t = x_t.reshape([-1, 256, 256, 1])
                y_t = y_t.reshape([-1, 1])
            except Exception as ex:
                print(ex)
                continue
            fd_t = {X: x_t, y: y_t}
            _, batch_loss = sess.run([optimizer, loss], feed_dict = fd_t)
            i = i + 1
            loss_trend.append(batch_loss)
            loss_trend_per_epoch.append(batch_loss)
        x_v = X_validation.reshape([-1, 256, 256, 1])
        y_v = y_validation.reshape([-1, 1])
        
        fd_v = {X: x_v, y: y_v}
        validation_accuracy = sess.run([accuracy], feed_dict = fd_v)
        accuracy_trend.append(validation_accuracy)
        print("Epoch - {}) Avg loss: {:20} - Val Accuracy: {:.2%}".format(e, sum(loss_trend_per_epoch)/len(loss_trend_per_epoch), validation_accuracy[0]))
        del loss_trend_per_epoch
        saver.save(sess, "CHECKPOINT/model_at_"+str(e))
plt.plot(loss_trend)
plt.xlabel("no of iter")
plt.ylabel("loss value")
plt.title("Loss train over iteration")
test_path = "../input/chest-xray-pneumonia/chest_xray/test/"
normal_path = "NORMAL/"
pneumonia_path = "PNEUMONIA/"
normal_testset = [test_path+normal_path+element for element in os.listdir(test_path+normal_path)]
pneumonia_testset = [test_path+pneumonia_path+element for element in os.listdir(test_path+pneumonia_path)]
X_t1, y_t1 = create_numpy_array(normal_testset, len(normal_testset), (256, 256), "NORMAL")
X_t2, y_t2 = create_numpy_array(pneumonia_testset, len(pneumonia_testset), (256, 256), "PNEUMONIA")
X_test = np.concatenate((X_t1, X_t2))
y_test = np.concatenate((y_t1, y_t2))
X_test, y_test = utils.shuffle(X_test, y_test)
testing_trend = []
#fpr = {}
#tpr = {}
#roc_auc = {}
#y_predicted = np.zeros(len(y_test))
with tf.Session() as sess:
    saver.restore(sess, "CHECKPOINT/model_at_4")
    i=1
    for batch_len in range(len(X_test)//batch_size):
            x_t = X_test[(i-1)*batch_size:(i)*batch_size]
            y_t = y_test[(i-1)*batch_size:(i)*batch_size]
            x_t = x_t.reshape([-1, 256, 256, 1])
            y_t = y_t.reshape([-1, 1])
            fd_t = {X: x_t, y: y_t}
            testing_accuracy, output_res = sess.run([accuracy, outputs], feed_dict = fd_t)
            #y_predicted[(i-1)*batch_size:(i)*batch_size] = np.round(output_res.reshape([-1]))
            testing_trend.append(testing_accuracy)
            i = i + 1
print("Accuracy on Testing Set: {:.2%}".format(sum(testing_trend)/len(testing_trend)))
'''TP = np.count_nonzero(y_predicted * y_test)
TN = np.count_nonzero((y_predicted - 1) * (y_test - 1))
FP = np.count_nonzero(y_predicted * (y_test - 1))
FN = np.count_nonzero((y_predicted - 1) * y_test)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1 = 2 * precision * recall / (precision + recall)
print(precision)
print(recall)
print(f1)'''