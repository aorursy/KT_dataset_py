import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pdb
%matplotlib inline
train = pd.read_csv("../input/train.csv") # (7352, 563)
test = pd.read_csv("../input/test.csv") # (2947, 563)
train.head()
train = train.sample(frac=1)
test = test.sample(frac=1)
Y_train = train.Activity
train = train.drop(['Activity','subject'], axis=1) #(7352, 561)

Y_test = test.Activity
test = test.drop(['Activity','subject'], axis=1) #(2947, 561)
def generate_initial_centroid(samples, num_cluster):
  '''
  This is the first step of K-means. 
  samples: tensor (num_samples, num_features)
  num_cluster: K, number of clusters.
  
  '''
  num_samples = tf.shape(samples)[0]
  random_indices = tf.random_shuffle(tf.range(0, num_samples))
  centroid_indices = tf.slice(random_indices, [0], [num_cluster])
  init_centroid = tf.gather(samples, centroid_indices)
  
  return init_centroid
  
init_test = train.iloc[:1500].values

init_placeholder = tf.placeholder(tf.float32, [1500, 561])
init_res = generate_initial_centroid(init_placeholder, 6)
with tf.Session() as sess:
  init_centroid = sess.run(init_res, feed_dict={init_placeholder:init_test})
  
print("The expected shape is (6, 561) and the generated shape is {0}".format(init_centroid.shape))
# Check broadcasting rules. 

a = np.array([[[1,2,3],[4,5,6]]]) # (1,2,3) -- (1, num_samples, num_features)
b = np.array([[[1,1,1]],[[4,4,4]]]) # (2,1,3) -- (num_centroids, 1, num_features)
print("The result of a - b is \n{0}".format(a - b))
print("The shape of a - b is {0}".format((a-b).shape))
def assign_to_nearest(samples, centroids):
  """
  This function assign each sample to its nearest centroid. 
  samples: tensor, (num_samples, num_features)
  centroids: tensor, (num_centroids, num_features)
  """
  expend_samples = tf.expand_dims(samples, 0) # samples become (1, num_samples, num_features)
  expend_centroid = tf.expand_dims(centroids, 1) # centroid become (num_centroid, 1, num_features)
  
  ## each entry represents how far a sample to a centroid. 
  distances = tf.reduce_sum(tf.square(tf.subtract(expend_samples, expend_centroid)), 2) # distance: (num_centroid, num_samples)
  
  ## which centorid each sample is assigned to. 
  nearest_index = tf.argmin(distances, 0) # nearest_index:(num_samples)
  
  return nearest_index
assign_samples = tf.constant(np.array([[1,2,3],[4,5,6]]))
assign_centroid = tf.constant(np.array([[1,1,1],[4,4,4]]))
with tf.Session() as sess:
  assign_nearest_index = assign_to_nearest(assign_samples, assign_centroid)
  assign_res = sess.run(assign_nearest_index)

print("The expected output is (0,1), and the actual output is {0}".format(assign_res))
print("The first sample (1,2,3) should be assigned to centroid (1,1,1)")
print("The second sample (4,5,5) should be assigned to centroid (4,4,4)")
def update_centroid(samples, nearest_index, num_clusters):
  """
  samples: tensor, (num_samples, num_features)
  nearest_index: tensor, (num_samples)
  num_clusters: int
  """
  
  nearest_index = tf.to_int32(nearest_index)
  partitions = tf.dynamic_partition(samples, nearest_index, num_clusters)
  new_centroids = tf.concat([tf.reduce_mean(partition, 0, keep_dims=True) for partition in partitions], axis=0)
  
  return new_centroids, nearest_index
# Test the function: update_centroid.
with tf.Session() as sess:
  new_cent, _ = update_centroid(assign_samples, assign_res, 2)
  update_res = sess.run(new_cent)
  
print("The expected new centroids are (1,2,3), (4,5,6)")
print("The actual new centroids are \n{0}".format(update_res))
def update_centroid(samples, nearest_index, num_clusters):
  """
  samples: tensor, (num_samples, num_features)
  nearest_index: tensor, (num_samples)
  num_clusters: int
  """
  
  nearest_index = tf.to_int32(nearest_index)
  partitions = tf.dynamic_partition(samples, nearest_index, num_clusters)
  new_centroids = tf.concat([tf.reduce_mean(partition, 0, keep_dims=True) for partition in partitions], axis=0)
  
  return new_centroids, nearest_index
k_means_placeholder = tf.placeholder(tf.float32, shape=(7352, 561))
updated_centroids = tf.placeholder(tf.float32, shape=(6, 561))

init_centroids = generate_initial_centroid(k_means_placeholder, num_cluster=6)


nearest_index = assign_to_nearest(k_means_placeholder,updated_centroids)
updated_centroid = update_centroid(k_means_placeholder, nearest_index, 6)

with tf.Session() as sess:
  centroids = sess.run(init_centroids, feed_dict={k_means_placeholder:train})
  for i in range(0, 300):
    
    centroids,nearest_index = sess.run(updated_centroid, feed_dict={k_means_placeholder:train,
                                                         updated_centroids:centroids})
    
pd.crosstab(nearest_index, Y_train)
Y_train = pd.get_dummies(Y_train)
Y_test = pd.get_dummies(Y_test)

train = train.as_matrix()
test = test.as_matrix()

Y_train = Y_train.as_matrix()
Y_test = Y_test.as_matrix()
FEATURE_DIM = 561
LEARNING_RATE = 0.001
LABEL_DIM = 6
BATCH_SIZE = 64
NUM_EPOCH = 100
class Neural_Network():
  
  def __init__(self, feature_dim = FEATURE_DIM, label_dim = LABEL_DIM):
    self.feature_dim = feature_dim
    self.label_dim = label_dim
    
    
  def build_network(self, learning_rate=LEARNING_RATE):
    
    self.train_X = tf.placeholder(tf.float32, [None, self.feature_dim])
    self.train_Y = tf.placeholder(tf.float32, [None, self.label_dim])
    
    self.layer_1 = self.dense_layer(self.train_X, self.feature_dim, 
                                    1024, activation=tf.nn.relu, name='layer_1')
    self.layer_2 = self.dense_layer(self.layer_1, 1024, 512, 
                                   activation=tf.nn.relu, name='layer_2')
    self.layer_3 = self.dense_layer(self.layer_2, 512, 64, 
                                   activation=tf.nn.relu, name='layer_3')
    self.output = self.dense_layer(self.layer_3, 64, 6, name='output')
    
    self.loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels = self.train_Y))
    
    self.optimizer = tf.train.AdamOptimizer(learning_rate)
    
    self.train_step = self.optimizer.minimize(self.loss)
    
    self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.output,1), 
                                                    tf.argmax(self.train_Y, 1)),'float'))
    
  def dense_layer(self, inputs, input_size, output_size, name, activation=None):
    
    W = tf.get_variable(name=name+'_w',shape=(input_size, output_size), 
                        initializer=tf.contrib.layers.xavier_initializer())
    b = tf.get_variable(name=name+'_b', shape=(output_size))
    out = tf.matmul(inputs, W) + b
    
    if activation:
      return activation(out)
    else:
      return out
    
class Data():
  
  def __init__(self, train_X, train_Y, batch_size=BATCH_SIZE):
    
    self.train_X = train_X
    self.train_Y = train_Y
    self.batch_size = batch_size
    self.num_batch = self.train_X.shape[0]//batch_size
    
  def generate_batch(self):
    
    for i in range(self.num_batch):
      
      x = self.train_X[(i*self.batch_size):(i+1)*self.batch_size, :]
      y = self.train_Y[(i*self.batch_size):(i+1)*self.batch_size]
      
      yield x, y 
    
class Learn():
  
  def __init__(self, train_X, train_Y, test_X, test_Y, 
               batch_size=BATCH_SIZE, epoch = NUM_EPOCH):
    
    self.batch_size = batch_size
    self.epoch = epoch
    
    self.network = Neural_Network()
    self.network.build_network(learning_rate=0.001)
    self.data = Data(train_X, train_Y, self.batch_size)
    self.test_X = test_X
    self.test_Y = test_Y
  
  def run_training(self):
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
      
      sess.run(init)
      
      training_loss = []
      counter, tmp_loss = 0, 0
      
      for i in range(self.epoch):
        
        for x, y in self.data.generate_batch():
          
          feed_dict = {self.network.train_X:x, self.network.train_Y:y}
        
          _, loss = sess.run([self.network.train_step, self.network.loss], 
                             feed_dict=feed_dict)
          
          if counter % 100 == 0 and counter!=0:
            training_loss.append(tmp_loss/100)
            tmp_loss = 0
          else:
            tmp_loss += loss
            
          counter += 1
          
        print("Epoch {0}, loss is {1}".format(i, loss))
        
        
        
      self.training_loss = training_loss
      acc = sess.run([self.network.accuracy], feed_dict={self.network.train_X:self.test_X,
                                                        self.network.train_Y:self.test_Y})
      print("The testing accuracy is {0}".format(acc))
      
  def plot_training_loss(self):
    plt.plot(self.training_loss)
tf.reset_default_graph()

learner = Learn(train, Y_train, test, Y_test, epoch=NUM_EPOCH)
learner.run_training()
learner.plot_training_loss()
