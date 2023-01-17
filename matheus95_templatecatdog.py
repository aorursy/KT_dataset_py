!pip install tensorflow==1.15.4
!pip install tensorflow-gpu==1.15.4
import tensorflow.compat.v1 as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as ran
import PIL
import PIL.Image
import pathlib
from IPython.display import display
import csv
IMAGE_SIZE = 64

def load_dataset():
    tr_path = '/kaggle/input/entre-caes-e-gatos/training_set/training_set/'
    tr_data_dir = pathlib.Path(tr_path)
    ts_path = '/kaggle/input/entre-caes-e-gatos/test_set/test_set/'
    ts_data_dir = pathlib.Path(ts_path)
    
    cats = list(tr_data_dir.glob('cats/*.jpg')) + list(ts_data_dir.glob('cats/*.jpg'))
    dogs = list(tr_data_dir.glob('dogs/*.jpg')) + list(ts_data_dir.glob('dogs/*.jpg'))
    
    dataset = []
    
    print(f'Loading cats...')
    for c in cats:
        im = PIL.Image.open(str(c))
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('L')
        dataset.append(('cat', np.asarray(im)))
        del im
    
    print(f'Loading dogs...')
    for d in dogs:
        im = PIL.Image.open(str(d))
        im = im.resize((IMAGE_SIZE, IMAGE_SIZE)).convert('L')
        dataset.append(('dog', np.asarray(im)))
        del im
    del cats
    del dogs

    print('Shuffle dataset')
    
    # shuffle dataset
    np.random.seed(0)
    np.random.shuffle(dataset)
    
    print('List to np array')
    
    # list to array
    samples = np.array(list(map(lambda x: x[1], dataset))) / 255.0
    labels = np.array(list(map(lambda x: x[0], dataset)))
    del dataset
    
    print('Dataset loaded')
    
    return samples, labels


images, labels_ = load_dataset()
class Batch:
    
    def __init__(self, dataset, labels, batch_size=128):
        self.dataset = dataset
        self.labels = labels
        self.batch_size = batch_size
        
    def next_batch(self):
        start = ran.randint(0, self.dataset.shape[0] - self.batch_size)
        result = self.dataset[start:(start+batch_size)], labels[start:(start+batch_size)]
        return result
def preprocess_labels(labels):
    ### START CODE HERE ### (≈ 1 lines of code)
    new_labels = []
    ### END CODE HERE ###
    return np.array(new_labels) 
    
    
def split_dataset(images, labels, num_train):
    if num_train > images.shape[0]:
        raise Exception(f'A quantidade de amostras no conjunto de treino deve ser menor do que {images.shape[0]}')
    
    num_test = images.shape[0] - num_train
    ### START CODE HERE ### (≈ 4 lines of code)
    train_X = None
    train_y = None
    test_X = None
    test_y = None
    ### END CODE HERE ###
    
    return train_X, train_y, test_X, test_y

def conv2d(x, W, b, strides=1):
    ### START CODE HERE ### (≈ 3 lines of code)
    x = None
    result = None
    ### END CODE HERE ###
    return result

def maxpool2d(x, k=2):
    ### START CODE HERE ### (≈ 1 lines of code)
    result = None
    ### END CODE HERE ###
    return result

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, IMAGE_SIZE, IMAGE_SIZE, 1])
    
    ### START CODE HERE ### (≈ 9 lines of code)
    conv1 = None
    
    conv2 = None
    
    fc1 = None
    
    out = None
    ### END CODE HERE ###
    return out
print(images.shape)
print(labels_.shape)

index = ran.randint(0, images.shape[0])
print(f'index: {index}, class: {labels_[index]}')
display_image(images[index])

labels = preprocess_labels(labels_)
print(f'old_label: {labels_[index]}, new label: {labels[index]}')
### START CODE HERE ### (≈ 1 lines of code)
num_train = 1
### END CODE HERE ###
train_X, train_y, test_X, test_y = split_dataset(images, labels, num_train)
print(f'train shape - X: {train_X.shape}, y: {train_y.shape}')
print(f'test shape - X: {test_X.shape}, y: {test_y.shape}')
### START CODE HERE ### (≈ 4 lines of code)
learning_rate = 0.0
epochs = 1
batch_size = 1
dropout = 0.0
### END CODE HERE ###
display_epoch = 10

n_input_rows = train_X.shape[1]
n_input_cols =  train_X.shape[2]
n_classes = 2

batches = Batch(train_X, train_y, batch_size)
train_loss = []
train_acc = []
test_acc = []
    
with tf.Session() as sess:
    ### START CODE HERE ### (≈ 11 lines of code)
    x = None
    y = None
    keep_prob = None
    
    weights = {'wc1': None,
        'wc2': None,
        'wd1': None,
        'out': None}
    
    biases = {'bc1': None,
        'bc2': None,
        'bd1': None,
        'out': None}
    ### END CODE HERE ###
    
    pred = conv_net(x, weights, biases, keep_prob)
    
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = None
    ### END CODE HERE ###
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    pred_max = tf.argmax(pred, 1)
    
    init = tf.global_variables_initializer()
    
    sess.run(init)
    epoch = 1
    print(f'Start training...')
    while epoch <= epochs:
        batch_x, batch_y = batches.next_batch()
        sess.run(optimizer, feed_dict={x:batch_x, y: batch_y, keep_prob: dropout})
        
        if epoch % display_epoch == 0:
            loss_train, acc_train = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            
            acc_test, pred_values = sess.run([accuracy, pred_max], feed_dict={x: test_X, y: test_y, keep_prob: 1.0})
            
            print(f'Iter {epoch}, Minibatch Loss= {loss_train:.2f}, Training Accuracy= {acc_train:.2f}, Testing Accuracy: {acc_test:.2f}')
            
            train_loss.append(loss_train)
            train_acc.append(acc_train)
            test_acc.append(acc_test)
            
        epoch += 1
def write_csv(name, pred):
    with open(f'{name}.csv', mode='w') as pred_file:
        pred_writer = csv.writer(pred_file, delimiter=',')
        pred_writer.writerow(['Id', 'Expected'])
        for idx,value in enumerate(pred):
            pred_writer.writerow([f'{idx}', f'{float(value)}'])
write_csv('pred', pred_values)
eval_indices = range(0, epochs, display_epoch)

plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per iteration')
plt.xlabel('Iteration')
plt.ylabel('Softmax')
plt.show()

plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train and Test Accuracy')
plt.xlabel('Geration')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()