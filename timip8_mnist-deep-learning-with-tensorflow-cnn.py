# These are all the modules we'll be using later. Make sure you can import them

# before proceeding further.

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

import matplotlib.cm as cm

import seaborn as sns



import tensorflow as tf



# Config the matlotlib backend as plotting inline in IPython

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

labels = train['label']

images = train.iloc[:,1:]
image_size = 28

images.head()
from sklearn.model_selection import train_test_split

train_dataset, valid_dataset, train_labels, valid_labels = train_test_split(images, labels, test_size=0.33, random_state=42)

print(len(train_dataset))

print(len(valid_dataset))

print(len(train_labels))

print(len(valid_labels))
labels.value_counts()
data = images

fig = plt.figure() # this creates a fig 

fig.set_figheight(10); fig.set_figwidth(10) 

random_25 = data.loc[np.random.choice(len(data), 25)] 



for k in range(1, 26):

    ax = fig.add_subplot(5, 5, k)

    ax = plt.imshow(random_25.iloc[k-1].values.reshape(image_size,image_size), cmap=cm.binary)
print(train_dataset.shape)

print(valid_dataset.shape)

print(test.shape)

#from sklearn.metrics import accuracy_score



##runs logistic regression for different training set sizes

#reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0)

#training_size = [50,100,1000,2000,3000,4000,5000,6000,7000,8000]

#X = train_dataset

#output_predictions = []

#for i in training_size:

#    reg.fit(X[:i],train_labels[:i]) #reg.fit trains model

#    pred = reg.predict(X[i+1:i*2]) #uses other part of training set to predict

#    output_predictions.append(accuracy_score(train_labels[i+1:i*2],pred))#



#print(output_predictions)
image_size = 28

num_labels = 10

num_channels = 1 # grayscale



import numpy as np



def reformat(dataset, labels):

  dataset = dataset.values.reshape(

    (-1, image_size, image_size, num_channels)).astype(np.float32)

  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)

  return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)

valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)

test_dataset, test_labels = reformat(test,train_labels[:len(test)])

print('Training set', train_dataset.shape, train_labels.shape)

print('Validation set', valid_dataset.shape, valid_labels.shape)

print('Test set', test_dataset.shape, test_labels.shape)
batch_size = 16

patch_size = 5

depth = 16

num_hidden = 64

image_size = 28

num_labels = 10

num_channels = 1



graph = tf.Graph()



with graph.as_default():



  # Input data.

  tf_train_dataset = tf.placeholder(

    tf.float32, shape=(batch_size, image_size, image_size, num_channels))

  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

  tf_valid_dataset = tf.constant(valid_dataset)

  tf_test_dataset = tf.constant(test_dataset)    

    

  # Variables.

  layer1_weights = tf.Variable(tf.truncated_normal(

      [patch_size, patch_size, num_channels, depth], stddev=0.1))

  layer1_biases = tf.Variable(tf.zeros([depth]))

  layer2_weights = tf.Variable(tf.truncated_normal(

      [patch_size, patch_size, depth, depth], stddev=0.1))

  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

  layer3_weights = tf.Variable(tf.truncated_normal(

      [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))

  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

  layer4_weights = tf.Variable(tf.truncated_normal(

      [num_hidden, num_labels], stddev=0.1))

  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

  

  # Model.

  def model(data):

    conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')

    hidden = tf.nn.relu(conv + layer1_biases)

    conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')

    hidden = tf.nn.relu(conv + layer2_biases)

    shape = hidden.get_shape().as_list()

    reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)

    return tf.matmul(hidden, layer4_weights) + layer4_biases

  

  # Training computation.

  logits = model(tf_train_dataset)

  loss = tf.reduce_mean(

    tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    

  # Optimizer.

  optimizer = tf.train.AdamOptimizer().minimize(loss)

  

  # Predictions for the training, validation, and test data.

  train_prediction = tf.nn.softmax(logits)

  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))

  test_prediction = tf.nn.softmax(model(tf_test_dataset))
def accuracy(predictions, labels):

  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))

          / predictions.shape[0])
num_steps = 10001



with tf.Session(graph=graph) as session:

  tf.global_variables_initializer().run()

  print('Initialized')

  for step in range(num_steps):

    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)

    batch_data = train_dataset[offset:(offset + batch_size), :, :, :]

    batch_labels = train_labels[offset:(offset + batch_size), :]

    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}

    _, l, predictions = session.run(

      [optimizer, loss, train_prediction], feed_dict=feed_dict)

    if (step % 1000 == 0):

      print('Minibatch loss at step %d: %f' % (step, l))

      print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

      print('Validation accuracy: %.1f%%' % accuracy(

        valid_prediction.eval(), valid_labels))

  sub = np.argmax(test_prediction.eval(), 1)
output = pd.DataFrame({'Label': sub })

output.index += 1 

output.index.name = 'ImageId'

output.head()

output.to_csv('sub.csv')