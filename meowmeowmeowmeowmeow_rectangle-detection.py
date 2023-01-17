import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import tensorflow as tf
import json
import functools
import matplotlib.pyplot as plt
import time
import cv2
import random

dataset_dir = '../input/rect_dataset/rect_dataset/'
train_file = 'train_ann.json'
validation_file = 'validation_ann.json'
def to_np_array(data):
    return np.array(list(map(lambda x: np.array(x), data)))


def draw_sample(file, bbox_gt, bbox_pred=None):
    if type(file) is not np.ndarray:
        img = cv2.imread(file)
    else:
        img = file
    pt1 = lambda x: (int(x[0]*img.shape[1]), int(x[1]*img.shape[0]))
    pt2 = lambda x: (int(x[2]*img.shape[1]), int(x[3]*img.shape[0]))
    img = cv2.rectangle(img, pt1(bbox_gt), pt2(bbox_gt), (50,255,50), 3)
    if bbox_pred is not None:
        img = cv2.rectangle(img, pt1(bbox_pred), pt2(bbox_pred), (255,50,50), 3)
        
    return img


train_data = pd.read_json(os.path.join(dataset_dir, train_file))
validation_data = pd.read_json(os.path.join(dataset_dir, validation_file))

x_train, y_train = to_np_array(train_data['fname']), to_np_array(train_data['box'])
x_test, y_test = to_np_array(validation_data['fname']), to_np_array(validation_data['box'])

x_train = np.array(list(map(lambda x: os.path.join(dataset_dir, x), x_train)))
x_test = np.array(list(map(lambda x: os.path.join(dataset_dir, x), x_test)))

y_train = np.array(list(map(lambda x: np.array([x[0]/320, x[1]/240, x[2]/320, x[3]/240]), y_train)))
y_test = np.array(list(map(lambda x: np.array([x[0]/320, x[1]/240, x[2]/320, x[3]/240]), y_test)))

hist_x = np.hstack((y_train[:, 0]+(y_train[:, 2]-y_train[:, 0])/2, y_test[:, 0]+(y_test[:, 2]-y_test[:, 0])/2))
hist_y = np.hstack((y_train[:, 1]+(y_train[:, 3]-y_train[:, 1])/2, y_test[:, 1]+(y_test[:, 3]-y_test[:, 1])/2))
hist_w = np.hstack((y_train[:, 2]-y_train[:, 0], y_test[:, 2]-y_test[:, 0]))
hist_h = np.hstack((y_train[:, 3]-y_train[:, 1], y_test[:, 3]-y_test[:, 1]))

fig, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 5), dpi=80)
axs[0].hist(hist_x, bins=20, label='X-center distibution', color='#68a0f9')
axs[0].legend()
axs[1].hist(hist_y, bins=20, label='Y-center distibution', color='#f96868')
axs[1].legend()
axs[2].hist(hist_w, bins=20, label='Width distibution', color='#68f9cb')
axs[2].legend()
axs[3].hist(hist_h, bins=20, label='Height distibution', color='#bff968')
axs[3].legend()

rows = 4
cols = 4

fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(20, 15), dpi=80)

for i in range(rows):
    for j in range(cols):
        idx = random.randint(0, len(x_train)-1)
        img = draw_sample(x_train[idx], y_train[idx])
        axs[i, j].imshow(img)
def parse_function(filename, label):
        image_string = tf.read_file(filename)
        image = tf.image.decode_png(image_string, channels=3)
        return image, label

def train_preprocess(image, label):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def build_dataset(files, labels, batch_size, epochs, prefetch_count = 1):
        def build_dataset_lambda():
            dataset = tf.data.Dataset.from_tensor_slices((files, labels))
            dataset = dataset.shuffle(len(files))
            dataset = dataset.repeat(epochs)
            dataset = dataset.map(parse_function, num_parallel_calls=4)
            dataset = dataset.map(train_preprocess, num_parallel_calls=4)
            dataset = dataset.batch(batch_size)
            dataset = dataset.prefetch(prefetch_count)
            return dataset

        return build_dataset_lambda
    
def make_dataset_minibatch_generator(x, y, batch_size, epochs, sess):
    dataset = build_dataset(x, y, batch_size, epochs, 1)()
    iterator = dataset.make_initializable_iterator()
    generator = iterator.get_next()
    sess.run(iterator.initializer)
    return generator

def load_feed_dict(sess, generator, X, Y):
    batch = sess.run(train_batch_generator)
    feed_dict = {X:batch[0], Y:reshape_y_tensor(batch[1])}
    return feed_dict

def reshape_y_tensor(x_tensor):
    return x_tensor.reshape([len(x_tensor), -1])
def bb_intersection_over_union(boxA, boxB):
    xA = tf.maximum(boxA[0], boxB[0])
    yA = tf.maximum(boxA[1], boxB[1])
    xB = tf.minimum(boxA[2], boxB[2])
    yB = tf.minimum(boxA[3], boxB[3])
 
    interArea = tf.maximum(0.0, xB - xA + 1.0) * tf.maximum(0.0, yB - yA + 1.0)
 
    boxAArea = (boxA[2] - boxA[0] + 1.0) * (boxA[3] - boxA[1] + 1.0)
    boxBArea = (boxB[2] - boxB[0] + 1.0) * (boxB[3] - boxB[1] + 1.0)

    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

tf.reset_default_graph()

X = tf.placeholder(dtype=tf.float32, shape=[None, 240, 320, 3], name='X_input')
Y = tf.placeholder(dtype=tf.float32, shape=[None, 4], name='Y_input')

net = X

net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=4, activation=tf.nn.relu)
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=4, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=(2,2), strides=(2,2))
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=8, activation=tf.nn.relu)
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=8, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=(2,2), strides=(2,2))
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=16, activation=tf.nn.relu)
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=16, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=(2,2), strides=(2,2))
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=32, activation=tf.nn.relu)
net = tf.layers.conv2d(inputs=net, kernel_size=(3,3), strides=(1,1), filters=32, activation=tf.nn.relu)
net = tf.layers.max_pooling2d(inputs=net, pool_size=(2,2), strides=(2,2))

net = tf.layers.flatten(inputs=net)

net = tf.layers.dropout(inputs=net, rate=0.5)
net = tf.layers.dense(inputs=net, units=5000, activation=tf.nn.relu)
net = tf.layers.dropout(inputs=net, rate=0.4)
net = tf.layers.dense(inputs=net, units=320, activation=tf.nn.relu)
net = tf.layers.dropout(inputs=net, rate=0.2)
net = tf.layers.dense(inputs=net, units=64, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=net, units=4)

iou = bb_intersection_over_union(Y, logits)
loss = tf.losses.mean_squared_error(labels=Y, predictions=logits)

opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)
epochs = 60
train_batch_size = 64
test_batch_size = 64
delay_show = 7
iter_per_epoch = len(x_train)//train_batch_size
iter_per_epoch_test = len(x_test)//test_batch_size
best_acc = 0
restore = False

with tf.Session() as sess:
    saver = tf.train.Saver()
    
    print ("Load dataset...", end='')
    train_batch_generator = make_dataset_minibatch_generator(x=x_train,
                                                                         y = y_train,
                                                                         batch_size=train_batch_size,
                                                                         epochs=epochs*2,
                                                                         sess=sess)
    test_batch_generator =  make_dataset_minibatch_generator(x=x_test,
                                                                        y = y_test,
                                                                        batch_size=test_batch_size,
                                                                        epochs=epochs*2,
                                                                        sess=sess)
    print ("done")
    
    print("Initializing...", end='')
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    if restore:
        saver.restore(sess, "./model.ckpt")
    
    print('done')
    
    for e in range(epochs):
        loss_stat = []
        acc_stat = []
        loss_stat_test = []
        acc_stat_test = []
        
        next_time = time.time()-1
        for i in range(iter_per_epoch):
            feed_dict = load_feed_dict(X=X, Y=Y, generator=train_batch_generator, sess=sess)
            _, _loss, _iou = sess.run([opt, loss, iou], feed_dict=feed_dict)
            loss_stat.append(_loss)
            acc_stat.append(_iou)
            
            if time.time() > next_time:
                print("Epoch {} | Iteration {}/{} [{:0.2f}%] | Loss: {} IOU: {}".format(e,
                                                                                        i,
                                                                                        iter_per_epoch,
                                                                                        (i+1)/iter_per_epoch*100,
                                                                                        np.mean(loss_stat),
                                                                                        np.mean(acc_stat)))
                next_time = time.time()+delay_show
            
        print('End epoch. Testing...')
        
        for i in range(iter_per_epoch_test):
            feed_dict = load_feed_dict(X=X, Y=Y, generator=test_batch_generator, sess=sess)
            _, _loss, _iou = sess.run([opt, loss, iou], feed_dict=feed_dict)
            loss_stat_test.append(_loss)
            acc_stat_test.append(_iou)
            
            if time.time() > next_time:
                print("[TEST] Epoch {} | Iteration {}/{} [{:0.2f}%] | Loss: {} IOU: {}".format(e,
                                                                                                i,
                                                                                                iter_per_epoch_test,
                                                                                                (i+1)/iter_per_epoch_test*100,
                                                                                                np.mean(loss_stat_test),
                                                                                                np.mean(acc_stat_test)))
                next_time = time.time()+delay_show
                
        loss_stat = np.mean(loss_stat)
        acc_stat = np.mean(acc_stat)
        loss_stat_test = np.mean(loss_stat_test)
        acc_stat_test = np.mean(acc_stat_test)
        print("***EPOCH SUMMARY*** Loss: {} Acc: {} | Test Loss: {} Test IOU {}".format(loss_stat, acc_stat, loss_stat_test, acc_stat_test))
    
        if acc_stat > best_acc:
            best_acc = loss_stat_test
            save_path = saver.save(sess, "./model.ckpt")
            print("Model saved in path: %s" % save_path)
rows = 4
cols = 4

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    saver.restore(sess, "./model.ckpt")
    
    dataset = build_dataset(x_test, y_test, 1, 1, 1)()
    iterator = dataset.make_initializable_iterator()
    generator = iterator.get_next()
    sess.run(iterator.initializer)
    
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=(20, 15), dpi=80)

    for i in range(rows):
        for j in range(cols):
            
            sample = sess.run(generator)
            feed_dict = {X:sample[0], Y:sample[1]}
            _bbox, _loss = sess.run([logits, loss], feed_dict=feed_dict)
            
            
            _img = (sample[0][0,:,:,:]*255).astype(np.uint8)
            img = draw_sample(_img, sample[1][0], _bbox[0])
            axs[i, j].imshow(img)


