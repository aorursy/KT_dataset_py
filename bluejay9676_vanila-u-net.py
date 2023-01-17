import os
import sys
import numpy as np
import tensorflow as tf
import random
import math
import warnings
import pandas as pd
import cv2
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed
# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../input/stage1_train/'
TEST_PATH = '../input/stage1_test/'
MODEL_PATH = '../pretrained/'
# Get train and test IDs
train_ids = [p for p in os.listdir(TRAIN_PATH) if p.startswith('.') == False]
test_ids = [p for p in os.listdir(TEST_PATH) if p.startswith('.') == False]
# Get and resize train images and masks
images = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
labels = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
# for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
for n, id_ in tqdm(enumerate(train_ids), total=30):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    images[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', 
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    labels[n] = mask

X_train = images
Y_train = labels

# Get and resize test images
submission_id = []
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    submission_id.append(id_)

print('Done!')
def shuffle():
    global images, labels
    p = np.random.permutation(len(X_train))
    images = X_train[p]
    labels = Y_train[p]
def next_batch(batch_s, iters):
    if(iters == 0):
        shuffle()
    count = batch_s * iters
    return images[count:(count + batch_s)], labels[count:(count + batch_s)]
from tensorflow.python.ops import array_ops

def conv2d_3x3(filters, name):
    return tf.layers.Conv2D(filters=filters, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name=name)

def max_pool():
    return tf.layers.MaxPooling2D((2,2), strides=2, padding='same') 

def drop_out(rate):
    return tf.layers.Dropout(rate)

def conv2d_transpose_2x2(filters, name):
    return tf.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same', name=name)

def concatenate(branches):
    return array_ops.concat(branches, 3)

tf.set_random_seed(1234)
X = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS], name='X')
Y = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 1], name='Y')
lr = tf.placeholder(tf.float32, name='lr')
with tf.device("/gpu:0"):
    s = X / 255 # convert image to 0 .. 1.0

    c1 = conv2d_3x3(32, "c1") (s)
    c1 = drop_out(0.1) (c1)
    c1 = conv2d_3x3(32, "c1") (c1)
    p1 = max_pool() (c1)

    c2 = conv2d_3x3(64, "c2") (p1)
    c2 = drop_out(0.1) (c2)
    c2 = conv2d_3x3(64, "c2") (c2)
    p2 = max_pool() (c2)

    c3 = conv2d_3x3(128, "c3") (p2)
    c3 = drop_out(0.1) (c3)
    c3 = conv2d_3x3(128, "c3") (c3)
    p3 = max_pool() (c3)

    c4 = conv2d_3x3(256, "c4") (p3)
    c4 = drop_out(0.2) (c4)
    c4 = conv2d_3x3(256, "c4") (c4)
    p4 = max_pool() (c4)

    c5 = conv2d_3x3(512, "c5") (p4)
    c5 = drop_out(0.1) (c5)
    c5 = conv2d_3x3(512, "c5") (c5)

    u6 = conv2d_transpose_2x2(256, "u6") (c5)
    u6 = concatenate([u6, c4])
    c6 = conv2d_3x3(256, "c6") (u6)
    c6 = drop_out(0.2) (c6)
    c6 = conv2d_3x3(256, "c6") (c6)

    u7 = conv2d_transpose_2x2(128, "u7") (c6)
    u7 = concatenate([u7, c3])
    c7 = conv2d_3x3(128, "c7") (u7)
    c7 = drop_out(0.1) (c7)
    c7 = conv2d_3x3(128, "c7") (c7)

    u8 = conv2d_transpose_2x2(64, "u8") (c7)
    u8 = concatenate([u8, c2])
    c8 = conv2d_3x3(64, "c8") (u8)
    c8 = drop_out(0.1) (c8)
    c8 = conv2d_3x3(64, "c8") (c8)

    u9 = conv2d_transpose_2x2(32, "u9") (c8)
    u9 = concatenate([u9, c1])
    c9 = conv2d_3x3(32, "c9") (u9)
    c9 = drop_out(0.1) (c9)
    c9 = conv2d_3x3(32, "c9") (c9)

    logits = tf.layers.Conv2D(1, (1, 1)) (c9)

loss = tf.losses.sigmoid_cross_entropy(Y, logits)
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver()
epoch = 12000
batch_iter = 67
batch_size = 10
learning_rate = 1e-4
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
TRAIN = True

if TRAIN:
    batch_count = 0
    for i in range(epoch):
        if(batch_count > batch_iter):
            batch_count = 0    
        batch_X, batch_Y = next_batch(batch_size, batch_count)
        batch_count += 1
        feed_dict = {X: batch_X, Y: batch_Y, lr: learning_rate}
        loss_value, _ = sess.run([loss, optimizer], feed_dict=feed_dict)

        if(i % 100 == 0):
            print(str(i) + " training loss:", str(loss_value))
        if(i%1000 == 0):
            saver.save(sess, './u_net_kaggle', global_step=i)

    saver.save(sess, './u_net_kaggle', global_step=10000)
    print("Training finished.")
ix = random.randint(0, 600)
check_data = np.expand_dims(np.array(images[ix]), axis=0)
check_train = {X:check_data}
check_train_mask = sess.run(logits,feed_dict=check_train)
true_mask = labels[ix]
print("original image")
imshow(images[ix])
plt.show()
print("true mask")
print(true_mask.shape)
imshow(true_mask.squeeze().astype(np.uint8))
plt.show()
print("produced mask")
print(check_train_mask.shape)
imshow(check_train_mask.squeeze().astype(np.uint8))
plt.show()
ix = random.randint(0, len(X_test)) #len(X_test) - 1 = 64
test_image = X_test[ix].astype(float)
imshow(test_image)
plt.show()
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
#print(ix)
test_image = np.reshape(test_image, [-1, IMG_HEIGHT , IMG_WIDTH, IMG_CHANNELS])
test_data = {X:test_image}

test_mask = sess.run([logits],feed_dict=test_data)
test_mask = np.reshape(np.squeeze(test_mask), [IMG_WIDTH , IMG_WIDTH, 1])
for i in range(IMG_WIDTH):
    for j in range(IMG_HEIGHT):
            test_mask[i][j] = int(sigmoid(test_mask[i][j])*255)
print(test_mask.shape)
imshow(test_mask.squeeze().astype(np.uint8))
plt.show()
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

img_masks = sess.run(logits,feed_dict= {X:X_test})
# Create list of upsampled test masks
preds_test_upsampled = []
for i in range(len(img_masks)):
    preds_test_upsampled.append(resize(np.squeeze(img_masks[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))
    
# Create submission DataFrame
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-dsbowl2018-1.csv', index=False)


sub.head()