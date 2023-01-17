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

import tensorflow as tf
import keras.preprocessing.image
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble
import os;
import datetime  
import cv2 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm  

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train['label'].value_counts().sort_index()
width = height = 28
channel = 1
data_train = train.iloc[:,1:].values.reshape(-1,width,height,channel).astype(np.float)
data_train_labels = train.iloc[:,0].values
data_train=data_train/data_train.max()
plt.figure(figsize=(15,9))
for i in range(50):
    plt.subplot(5,10,1+i)
    plt.title(data_train_labels[i])
    plt.imshow(data_train[i].reshape(28,28), cmap=cm.binary)
nlabels = data_train_labels.shape[0]
nclasses = np.unique(data_train_labels).shape[0]
data_train_labels_one_hot = np.zeros((nlabels,nclasses))
data_train_labels_one_hot.flat[np.arange(nlabels)*nclasses+data_train_labels.ravel()]=1
image_generator = keras.preprocessing.image.ImageDataGenerator(rotation_range = 10, 
                                                               width_shift_range = 0.1 , 
                                                               height_shift_range = 0.1, 
                                                               zoom_range = 0.1)
class cnn:
    def __init__(self, args):
        #hyperparameters
        self.conv1_filter_size = args['conv1_filter'][1]
        self.conv1_filter_num = args['conv1_filter'][0]
        self.conv2_filter_size = args['conv2_filter'][1]
        self.conv2_filter_num = args['conv2_filter'][0]
        self.conv3_filter_size = args['conv3_filter'][1]
        self.conv3_filter_num = args['conv3_filter'][0]
        self.fullconn1_filter_num = args['fullconn1_filter_num']
    
        self.batch_size = args['batch_size']
        self.keep_prob_ = args['keep_prob']
        self.learn_rate_array = args['learn_rate_array']
        self.learn_rate_change_every = args['learn_rate_change_every'] #learn_rate_step_size
        
        #parameters
        self.learn_rate_ = self.learn_rate_array[0]
        self.current_learn_rate_index = 0 #learn_rate_pos
        self.current_index_in_epoch = 0 #index_in_epoch
        self.current_progress_in_epoch = 0 #current_epoch
        self.current_batch = 0 #n_log_step
        self.log_every_progress = args['log_every_progress'] #log_step
        self.use_tb_summary = args['use_tb_summary'] 
        self.use_tf_saver = args['use_tf_saver'] 
        self.name = args['name']
        
        # permutation array
        self.perm_array = np.array([])
    
        self.image_generator = keras.preprocessing.image.ImageDataGenerator(
                                rotation_range = 10, width_shift_range = 0.1 , height_shift_range = 0.1,
                                zoom_range = 0.1)
        
    def next_batch(self):
        start = self.current_index_in_epoch
        self.current_index_in_epoch += self.batch_size
        self.current_progress_in_epoch += self.batch_size/len(self.x_train)
        
        if len(self.perm_array) != len(self.x_train):
            self.perm_array = np.arange(len(self.x_train))
        
        if start == 0:
            np.random.shuffle(self.perm_array)
        
        if self.current_index_in_epoch > self.x_train.shape[0]:
            np.random.shuffle(self.perm_array)
            start = 0
            self.current_index_in_epoch = self.batch_size
            
            if self.augmented:
                self.x_train_aug = self.augment(self.x_train)
                self.x_train_aug = self.x_train_aug/self.x_train_aug.max()
                self.y_train_aug = self.y_train
            
        end = self.current_index_in_epoch
        
        if self.augmented:
            x_train_ = self.x_train_aug[self.perm_array[start:end]]
            y_train_ = self.y_train_aug[self.perm_array[start:end]]
        else:
            x_train_ = self.x_train[self.perm_array[start:end]]
            y_train_ = self.y_train[self.perm_array[start:end]]
        
        return x_train_, y_train_
    
    def augment(self,imgs):
        print('Augmenting images')
        imgs_ = self.image_generator.flow(imgs.copy(), np.zeros(len(imgs)),
                                    batch_size=len(imgs), shuffle = False).next()
        return imgs_[0]
    
    def init_weight(self, shape, name = None):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name = name)
    
    def init_bias(self, shape, name = None):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name = name)
    
    def conv2d(self, x, W, name = None):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name = name)
    
    def max_pool(self, x, name = None):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                              padding='SAME', name = name) 
    
    def summary_variable(self, var, var_name):
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
    
    def create_graph(self):
        tf.reset_default_graph()
        
        self.x_data = tf.placeholder(dtype=tf.float32, shape=[None,28,28,1], name = 'x_data')
        self.y_data = tf.placeholder(dtype=tf.float32, shape=[None,10], name = 'y_data')
        
        #layer 1: conv -> max pool
        self.conv1_W = self.init_weight([self.conv1_filter_size, self.conv1_filter_size, 1, self.conv1_filter_num], name = 'conv1_W')
        self.conv1_b = self.init_bias([self.conv1_filter_num], name = 'conv1_b')
        self.conv1_h = tf.nn.relu(self.conv2d(self.x_data, self.conv1_W)+self.conv1_b, name = 'conv1_h')
        self.pool1_h = self.max_pool(self.conv1_h, name = 'pool1_h')
        
        #layer 2: conv -> max pool
        self.conv2_W = self.init_weight([self.conv2_filter_size, self.conv2_filter_size, self.conv1_filter_num, self.conv2_filter_num], name = 'conv2_W')
        self.conv2_b = self.init_bias([self.conv2_filter_num], name = 'conv2_b')
        self.conv2_h = tf.nn.relu(self.conv2d(self.pool1_h, self.conv2_W)+self.conv2_b, name = 'conv2_h')
        self.pool2_h = self.max_pool(self.conv2_h, name = 'pool2_h')
        
        #layer 3: conv -> max pool
        self.conv3_W = self.init_weight([self.conv3_filter_size, self.conv3_filter_size, self.conv2_filter_num, self.conv3_filter_num], name = 'conv3_W')
        self.conv3_b = self.init_bias([self.conv3_filter_num], name = 'conv3_b')
        self.conv3_h = tf.nn.relu(self.conv2d(self.pool2_h, self.conv3_W)+self.conv3_b, name = 'conv3_h')
        self.pool3_h = self.max_pool(self.conv3_h, name = 'pool3_h')
        
        #layer 4: fully connected
        self.fullconn1_W = self.init_weight([4*4*self.conv3_filter_num,self.fullconn1_filter_num],name='fullconn1_W')
        self.fullconn1_b = self.init_bias([self.fullconn1_filter_num], name='fullconn1_b')
        self.pool3_h_flat = tf.reshape(self.pool3_h, [-1, 4*4*self.conv3_filter_num], name = 'pool3_h_flat')
        self.fullconn1_h = tf.nn.relu(tf.matmul(self.pool3_h_flat,self.fullconn1_W)+self.fullconn1_b, name = 'fullconn1_h')

        #dropout
        self.keep_prob = tf.placeholder(dtype=tf.float32, name = 'keep_prob')
        self.fullconn1_h_dropout = tf.nn.dropout(self.fullconn1_h, self.keep_prob, name = 'fullconn1_h_dropout')

        #layer 5: fully connected
        self.fullconn2_W = self.init_weight([self.fullconn1_filter_num,10], name = 'fullconn2_W')
        self.fullconn2_b = self.init_bias([10], name = 'fullconn2_b')
        self.pred = tf.add(tf.matmul(self.fullconn1_h_dropout, self.fullconn2_W),self.fullconn2_b, name = 'pred')
        
        #cost function
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_data, logits=self.pred), name = 'cross_entropy')
        
        #training
        self.learn_rate = tf.placeholder(dtype=tf.float32,name = 'learn_rate')
        self.train_step = tf.train.AdamOptimizer(self.learn_rate).minimize(self.cross_entropy, name = 'train_step')
        
        self.y_pred_prob = tf.nn.softmax(self.pred, name = 'y_pred_prob')
        
        self.y_pred_correct = tf.equal(tf.argmax(self.y_pred_prob,1), tf.argmax(self.y_data,1), name = 'y_pred_correct')
        
        self.accuracy = tf.reduce_mean(tf.cast(self.y_pred_correct, dtype=tf.float32), name = 'accuracy')
        
        self.train_loss = tf.Variable(np.array([]),dtype=tf.float32, validate_shape=False, name = 'train_loss')
        self.valid_loss = tf.Variable(np.array([]),dtype=tf.float32, validate_shape=False, name = 'valid_loss' )
        self.train_acc = tf.Variable(np.array([]),dtype=tf.float32, validate_shape=False, name = 'train_acc')
        self.valid_acc = tf.Variable(np.array([]),dtype=tf.float32, validate_shape=False, name = 'valid_acc' )
        
        return None
    
    def attach_summary(self, session):
        self.use_tb_summary = True
        self.summary_variable(self.conv1_W, 'conv1_W')
        self.summary_variable(self.conv1_b, 'conv1_b')
        self.summary_variable(self.conv2_W, 'conv2_W')
        self.summary_variable(self.conv2_b, 'conv2_b')
        self.summary_variable(self.conv3_W, 'conv3_W')     
        self.summary_variable(self.conv3_b, 'conv3_b')
        self.summary_variable(self.fullconn1_W, 'fullconn1_W')
        self.summary_variable(self.fullconn1_b, 'fullconn1_b')
        self.summary_variable(self.fullconn2_W, 'fullconn2_W')
        self.summary_variable(self.fullconn2_b, 'fullconn2_b')
        
        tf.summary.scalar('cross_entropy', self.cross_entropy)
        tf.summary.scalar('accuracy', self.accuracy)
        
        self.merged_summary = tf.summary.merge_all()
        
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), 'logs', (self.name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), session.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), session.graph)
        
    def attach_saver(self):
        self.use_tf_saver=True
        self.saver=tf.train.Saver()
    
    def train_graph(self, session, x_train, y_train, x_valid, y_valid, n_epoch=1, augmented=False):
        
        self.augmented = augmented
        
        self.x_train=x_train
        self.y_train=y_train
        self.x_valid=x_valid
        self.y_valid=y_valid
        
        if self.augmented:
            print('Augmenting images')
            self.x_train_aug=self.augment(self.x_train)
            self.x_train_aug=self.x_train_aug/self.x_train_aug.max()
            self.y_train_aug=self.y_train
        
        batch_per_epoch=self.x_train.shape[0]/self.batch_size
        train_loss, train_acc, valid_loss, valid_acc = [],[],[],[]
        
        start = datetime.datetime.now();
        print(datetime.datetime.now().strftime('%d-%m-%Y %H:%M:%S'),': start training')
        print('learnrate = ',self.learn_rate_,', n_epoch = ', n_epoch, ', batch_size = ', self.batch_size)
        
        for i in range(int(n_epoch*batch_per_epoch)+1):
            
            self.current_learn_rate_index=int(self.current_progress_in_epoch//self.learn_rate_change_every)
            if self.learn_rate_ != self.learn_rate_array[self.current_learn_rate_index]:
                self.learn_rate_ = self.learn_rate_array[self.current_learn_rate_index]
                print(datetime.datetime.now()-start,': set learn rate to %.6f'%self.learn_rate_)
            
            x_batch, y_batch = self.next_batch()
            
            session.run(self.train_step, feed_dict={self.x_data: x_batch, self.y_data: y_batch, 
                                                    self.keep_prob: self.keep_prob_, self.learn_rate: self.learn_rate_})
        
            if i%int(self.log_every_progress*batch_per_epoch) == 0 or i == int(n_epoch*batch_per_epoch):
                self.current_batch +=1
                
                feed_dict_train = {
                    self.x_data: self.x_train[self.perm_array[:len(self.x_valid)]],
                    self.y_data: self.y_train[self.perm_array[:len(self.y_valid)]],
                    self.keep_prob: 1.0
                }
                
                feed_dict_valid = {
                    self.x_data: self.x_valid,
                    self.y_data: self.y_valid,
                    self.keep_prob: 1.0
                }
                
                if self.use_tb_summary:
                    train_summary = session.run(self.merged_summary, feed_dict=feed_dict_train)
                    valid_summary = session.run(self.merged_summary, feed_dict=feed_dict_valid)
                    self.train_writer.add_summary(train_summary, self.current_batch)
                    self.valid_writer.add_summary(valid_summary, self.current_batch)
                    
                train_loss.append(session.run(self.cross_entropy, feed_dict=feed_dict_train))
                train_acc.append(self.accuracy.eval(session=session, feed_dict=feed_dict_train))  
                
                valid_loss.append(session.run(self.cross_entropy, feed_dict=feed_dict_valid))
                valid_acc.append(self.accuracy.eval(session=session, feed_dict=feed_dict_valid))
                
                print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(
                    self.current_progress_in_epoch, train_loss[-1], valid_loss[-1], train_acc[-1], valid_acc[-1]))
        
        tl_c = np.concatenate([self.train_loss.eval(session=session), train_loss], axis = 0)
        vl_c = np.concatenate([self.valid_loss.eval(session=session), valid_loss], axis = 0)
        ta_c = np.concatenate([self.train_acc.eval(session=session), train_acc], axis = 0)
        va_c = np.concatenate([self.valid_acc.eval(session=session), valid_acc], axis = 0)
   
        session.run(tf.assign(self.train_loss, tl_c, validate_shape = False))
        session.run(tf.assign(self.valid_loss, vl_c , validate_shape = False))
        session.run(tf.assign(self.train_acc, ta_c , validate_shape = False))
        session.run(tf.assign(self.valid_acc, va_c , validate_shape = False))
        
        print('running time for training: ', datetime.datetime.now() - start)
        return None
  
    def save_model(self, session):
        if self.use_tf_saver:
            filepath = os.path.join(os.getcwd(), self.name)
            self.saver.save(session,filepath)
        
        if self.use_tb_summary:
            self.train_writer.close()
            self.valid_writer.close()
        return None
    
    def predict(self, session, x_data):
        return self.y_pred_prob.eval(session=session, feed_dict={self.x_data: x_data, self.keep_prob:1.0})
    
    def load_tensors(self, graph):
        
        self.x_data = graph.get_tensor_by_name("x_data:0")
        self.y_data = graph.get_tensor_by_name("y_data:0")
        
        self.conv1_W = graph.get_tensor_by_name("conv1_W:0")
        self.conv1_b = graph.get_tensor_by_name("conv1_b:0")
        self.conv2_W = graph.get_tensor_by_name("conv2_W:0")
        self.conv2_b = graph.get_tensor_by_name("conv2_b:0")
        self.conv3_W = graph.get_tensor_by_name("conv3_W:0")
        self.conv3_b = graph.get_tensor_by_name("conv3_b:0")
        self.fullconn1_W = graph.get_tensor_by_name("fullconn1_W:0")
        self.fullconn1_b = graph.get_tensor_by_name("fullconn1_b:0")
        self.fullconn2_W = graph.get_tensor_by_name("fullconn2_W:0")
        self.fullconn2_b = graph.get_tensor_by_name("fullconn2_b:0")
        
        # activation tensors
        self.conv1_h = graph.get_tensor_by_name('conv1_h:0')  
        self.pool1_h = graph.get_tensor_by_name('pool1_h:0')
        self.conv2_h = graph.get_tensor_by_name('conv2_h:0')  
        self.pool2_h = graph.get_tensor_by_name('pool2_h:0')
        self.conv3_h = graph.get_tensor_by_name('conv3_h:0')  
        self.pool3_h = graph.get_tensor_by_name('pool3_h:0')
        
        self.fullconn1_h = graph.get_tensor_by_name('fullconn1_h:0')
        self.pred = graph.get_tensor_by_name('pred:0')
        
        # training and prediction tensors
        self.learn_rate = graph.get_tensor_by_name("learn_rate:0")
        self.keep_prob = graph.get_tensor_by_name("keep_prob:0")
        self.cross_entropy = graph.get_tensor_by_name('cross_entropy:0')
        self.train_step = graph.get_operation_by_name('train_step')
        self.pred = graph.get_tensor_by_name('pred:0')
        self.y_pred_prob = graph.get_tensor_by_name("y_pred_prob:0")
        self.y_pred_correct = graph.get_tensor_by_name('y_pred_correct:0')
        self.accuracy = graph.get_tensor_by_name('accuracy:0')
        
        # tensor of stored losses and accuricies during training
        self.train_loss = graph.get_tensor_by_name("train_loss:0")
        self.train_acc = graph.get_tensor_by_name("train_acc:0")
        self.valid_loss = graph.get_tensor_by_name("valid_loss:0")
        self.valid_acc = graph.get_tensor_by_name("valid_acc:0")
  
        return None
    
    def get_loss(self, session):
        train_loss = self.train_loss.eval(session = session)
        valid_loss = self.valid_loss.eval(session = session)
        return train_loss, valid_loss 
        
    def get_accuracy(self, session):
        train_acc = self.train_acc.eval(session = session)
        valid_acc = self.valid_acc.eval(session = session)
        return train_acc, valid_acc 
    
    def get_weights(self, session):
        conv1_W = self.conv1_W.eval(session = session)
        conv2_W = self.conv2_W.eval(session = session)
        conv3_W = self.conv3_W.eval(session = session)
        fullconn1_W = self.fullconn1_W.eval(session = session)
        fullconn2_W = self.fullconn2_W.eval(session = session)
        return conv1_W, conv2_W, conv3_W, fullconn1_W, fullconn2_W
    
    def get_biases(self, session):
        conv1_b = self.conv1_b.eval(session = session)
        conv2_b = self.conv2_b.eval(session = session)
        conv3_b = self.conv3_b.eval(session = session)
        fullconn1_b = self.fullconn1_b.eval(session = session)
        fullconn2_b = self.fullconn2_b.eval(session = session)
        return conv1_b, conv2_b, conv3_b, fullconn1_b, fullconn2_b
    
    def load_session_from_file(self, filename):
        tf.reset_default_graph()
        filepath = os.path.join(os.getcwd(), filename + '.meta')
        saver = tf.train.import_meta_graph(filepath)
        print(filepath)
        session = tf.Session()
        saver.restore(session, mn)
        graph = tf.get_default_graph()
        self.load_tensors(graph)
        return session
    
    def get_activations(self, session, x_data):
        feed_dict = {self.x_data: x_data, self.keep_prob: 1.0}
        conv1_h = self.conv1_h.eval(session = session, feed_dict = feed_dict)
        pool1_h = self.pool1_h.eval(session = session, feed_dict = feed_dict)
        conv2_h = self.conv2_h.eval(session = session, feed_dict = feed_dict)
        pool2_h = self.pool2_h.eval(session = session, feed_dict = feed_dict)
        conv3_h = self.conv3_h.eval(session = session, feed_dict = feed_dict)
        pool3_h = self.pool3_h.eval(session = session, feed_dict = feed_dict)
        fullconn1_h = self.fullconn1_h.eval(session = session, feed_dict = feed_dict)
        fullconn2_h = self.pred.eval(session = session, feed_dict = feed_dict)
        return conv1_h,pool1_h,conv2_h,pool2_h,conv3_h,pool3_h,fullconn1_h,fullconn2_h        
                    
names=['nn0']

cv_num=10
kfold=sklearn.model_selection.KFold(cv_num,shuffle=True,random_state=42)

for i,(train_index,valid_index) in enumerate(kfold.split(data_train)):
    start=datetime.datetime.now()
    
    x_train = data_train[train_index]
    y_train = data_train_labels_one_hot[train_index]
    x_valid = data_train[valid_index]
    y_valid = data_train_labels_one_hot[valid_index]
    
    args={
        'name': names[i],
        'conv1_filter': [36, 3],
        'conv2_filter': [36, 3],
        'conv3_filter': [36, 3],
        'fullconn1_filter_num': 576,
        'batch_size': 50,
        'keep_prob': 0.33,
        'learn_rate_array': [1e-3,7.5e-4,5e-4,2.5e-4,1e-4,1e-4,1e-4,7.5e-5,
                             0.5e-4,0.25e-4,1e-5,1e-5,7.5e-6,5e-6,2.5e-6,1e-6,
                             7.5e-7,5e-7,2.5e-7,1e-7],
        'learn_rate_change_every': 3,
        'log_every_progress': 0.2,
        'use_tb_summary': False,
        'use_tf_saver': False
    }
    
    nn_graph = cnn(args=args)
    nn_graph.create_graph()
    nn_graph.attach_saver()
    
    with tf.Session() as session:
        nn_graph.attach_summary(session)
        session.run(tf.global_variables_initializer())
        
        nn_graph.train_graph(session, x_train, y_train, x_valid, y_valid, n_epoch=1.0)
        nn_graph.train_graph(session, x_train, y_train, x_valid, y_valid, n_epoch=14.0, augmented=True)
        
        nn_graph.save_model(session)
    
    if True:
        break;

print('total running time for training: ', datetime.datetime.now() - start)

y_valid_pred={}
mn = names[0]
nn_graph = cnn(args=args)
session = nn_graph.load_session_from_file(mn)
y_valid_pred[mn] = nn_graph.predict(session, x_valid)
session.close()

cnf_matrix = sklearn.metrics.confusion_matrix(
    np.argmax(y_valid_pred[mn],1), np.argmax(y_valid,1)).astype(np.float32)

labels_array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fig, ax = plt.subplots(1,figsize=(10,10))
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xticklabels(labels_array)
ax.set_yticklabels(labels_array)
plt.title('Confusion matrix of validation set')
plt.ylabel('True digit')
plt.xlabel('Predicted digit')
plt.show();
train_loss={}
valid_loss={}
train_acc={}
valid_acc={}
mn = names[0]
nn_graph = cnn(args=args)
session = nn_graph.load_session_from_file(mn)
train_loss[mn], valid_loss[mn] = nn_graph.get_loss(session)
train_acc[mn], valid_acc[mn] = nn_graph.get_accuracy(session)
session.close()

print('final train/valid loss = %.4f/%.4f, train/valid accuracy = %.4f/%.4f'%(
    train_loss[mn][-1], valid_loss[mn][-1], train_acc[mn][-1], valid_acc[mn][-1]))

plt.figure(figsize=(10, 5));
plt.subplot(1,2,1);
plt.plot(np.arange(0,len(train_acc[mn])), train_acc[mn],'-b', label='Training')
plt.plot(np.arange(0,len(valid_acc[mn])), valid_acc[mn],'-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 1.1, ymin = 0.0)
plt.ylabel('accuracy')
plt.xlabel('log steps');

plt.subplot(1,2,2)
plt.plot(np.arange(0,len(train_loss[mn])), train_loss[mn],'-b', label='Training')
plt.plot(np.arange(0,len(valid_loss[mn])), valid_loss[mn],'-g', label='Validation')
plt.legend(loc='lower right', frameon=False)
plt.ylim(ymax = 3.0, ymin = 0.0)
plt.ylabel('loss')
plt.xlabel('log steps');
img_no = 5;
mn = names[0]
nn_graph = cnn(args=args)
session = nn_graph.load_session_from_file(mn)
(conv1_h, pool1_h, conv2_h, pool2_h,conv3_h, pool3_h, fullconn1_h,
 fullconn2_h) = nn_graph.get_activations(session, data_train[img_no:img_no+1])
session.close()
    
# original image
plt.figure(figsize=(15,9))
plt.subplot(2,4,1)
plt.imshow(data_train[img_no].reshape(28,28),cmap=cm.binary);

# 1. convolution
plt.subplot(2,4,2)
plt.title('conv1_h ' + str(conv1_h.shape))
conv1_h = np.reshape(conv1_h,(-1,28,28,6,6))
conv1_h = np.transpose(conv1_h,(0,3,1,4,2))
conv1_h = np.reshape(conv1_h,(-1,6*28,6*28))
plt.imshow(conv1_h[0], cmap=cm.binary);

# 1. max pooling
plt.subplot(2,4,3)
plt.title('pool1_h ' + str(pool1_h.shape))
pool1_h = np.reshape(pool1_h,(-1,14,14,6,6))
pool1_h = np.transpose(pool1_h,(0,3,1,4,2))
pool1_h = np.reshape(pool1_h,(-1,6*14,6*14))
plt.imshow(pool1_h[0], cmap=cm.binary);

# 2. convolution
plt.subplot(2,4,4)
plt.title('conv2_h ' + str(conv2_h.shape))
conv2_h = np.reshape(conv2_h,(-1,14,14,6,6))
conv2_h = np.transpose(conv2_h,(0,3,1,4,2))
conv2_h = np.reshape(conv2_h,(-1,6*14,6*14))
plt.imshow(conv2_h[0], cmap=cm.binary);

# 2. max pooling
plt.subplot(2,4,5)
plt.title('pool2_h ' + str(pool2_h.shape))
pool2_h = np.reshape(pool2_h,(-1,7,7,6,6))
pool2_h = np.transpose(pool2_h,(0,3,1,4,2))
pool2_h = np.reshape(pool2_h,(-1,6*7,6*7))
plt.imshow(pool2_h[0], cmap=cm.binary);

# 3. convolution
plt.subplot(2,4,6)
plt.title('conv3_h ' + str(conv3_h.shape))
conv3_h = np.reshape(conv3_h,(-1,7,7,6,6))
conv3_h = np.transpose(conv3_h,(0,3,1,4,2))
conv3_h = np.reshape(conv3_h,(-1,6*7,6*7))
plt.imshow(conv3_h[0], cmap=cm.binary);

# 3. max pooling
plt.subplot(2,4,7)
plt.title('pool3_h ' + str(pool3_h.shape))
pool3_h = np.reshape(pool3_h,(-1,4,4,6,6))
pool3_h = np.transpose(pool3_h,(0,3,1,4,2))
pool3_h = np.reshape(pool3_h,(-1,6*4,6*4))
plt.imshow(pool3_h[0], cmap=cm.binary);

# 4. FC layer
plt.subplot(2,4,8)
plt.title('fullconn1_h ' + str(fullconn1_h.shape))
fullconn1_h = np.reshape(fullconn1_h,(-1,24,24))
plt.imshow(fullconn1_h[0], cmap=cm.binary);

# 5. FC layer
np.set_printoptions(precision=2)
print('fullconn2_h = ', fullconn2_h)
mn = names[0]
nn_graph = cnn(args=args)
session = nn_graph.load_session_from_file(mn)
y_valid_pred[mn] = nn_graph.predict(session, x_valid)
session.close()

y_valid_pred_label = np.argmax(y_valid_pred[mn],1)
y_valid_label = np.argmax(y_valid,1)
y_val_false_index = []

for i in range(y_valid_label.shape[0]):
    if y_valid_pred_label[i] != y_valid_label[i]:
        y_val_false_index.append(i)

print('# false predictions: ', len(y_val_false_index),'out of', len(y_valid))

plt.figure(figsize=(10,15))
for j in range(0,5):
    for i in range(0,10):
        if j*10+i<len(y_val_false_index):
            plt.subplot(10,10,j*10+i+1)
            plt.title('%d/%d'%(y_valid_label[y_val_false_index[j*10+i]],
                               y_valid_pred_label[y_val_false_index[j*10+i]]))
            plt.imshow(x_valid[y_val_false_index[j*10+i]].reshape(28,28),cmap=cm.binary)  
data_test = test.iloc[:,0:].values.reshape(-1,width,height,1) # (28000,28,28,1) array
data_test = data_test.astype(np.float)
data_test = data_test/data_test.max()

y_test_pred={}
y_test_pred_labels={}

mn = names[0]
nn_graph = cnn(args=args)
session = nn_graph.load_session_from_file(mn)
y_test_pred[mn] = nn_graph.predict(session, data_test )
session.close()

y_test_pred_labels[mn] = np.argmax(y_test_pred[mn],1)

print(mn+': y_test_pred_labels[mn].shape = ', y_test_pred_labels[mn].shape)
unique, counts = np.unique(y_test_pred_labels[mn], return_counts=True)
print(dict(zip(unique, counts)))

# save predictions
np.savetxt('submission.csv', 
           np.c_[range(1,len(data_test)+1), y_test_pred_labels[mn]], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

print('submission.csv completed')
plt.figure(figsize=(10,15))
for j in range(0,5):
    for i in range(0,10):
        plt.subplot(10,10,j*10+i+1)
        plt.title('%d'%y_test_pred_labels[mn][j*10+i])
        plt.imshow(data_test[j*10+i].reshape(28,28), cmap=cm.binary)