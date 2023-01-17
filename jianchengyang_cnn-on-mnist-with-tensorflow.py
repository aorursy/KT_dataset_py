__auther__ = 'Jiancheng'

# Basic
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Data Analysis Specific
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Marchine Learning Specific
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# IPython magic
% matplotlib inline
dtrain = pd.read_csv('../input/train.csv')
dtest = pd.read_csv('../input/test.csv')

Xfull, yfull = dtrain.drop('label',axis =1), dtrain['label']
Xtrain, Xvalid, ytrain, yvalid = train_test_split(Xfull, yfull, test_size=0.1)

rf = RandomForestClassifier(100)
rf.fit(Xtrain,ytrain)
print('Training score:',rf.score(Xtrain,ytrain))
print('Validation score:', rf.score(Xvalid,yvalid))

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset = None, label = None):
  pixel_depth = 255.0  
  if dataset is not None:
    dataset = (dataset.values.reshape((-1, image_size, image_size, num_channels)).astype(np.float32) - pixel_depth/2) / pixel_depth
  if label is not None:
    label = (np.arange(num_labels) == label.values[:,None]).astype(np.float32)
  return dataset, label
tensorXfull, tensoryfull = reformat(Xfull, yfull)
tensorXvalid, tensoryvalid = reformat(Xvalid, yvalid)
tensorXtrain, tensorytrain = reformat(Xtrain, ytrain)
tensordtest, _ = reformat(dtest)
print('Full', tensorXfull.shape, tensoryfull.shape)
print('Training', tensorXtrain.shape, tensorytrain.shape)
print('Validation', tensorXvalid.shape, tensoryvalid.shape)
print('Testing', tensordtest.shape)
def onehot2label(tensor_y):
    y = np.argmax(tensor_y, axis = 1)
    return y
def accuracy(predictions, labels):
  return ((100.0 * np.sum(onehot2label(predictions) == onehot2label(labels))))/ predictions.shape[0]
print('Test the two function:',accuracy(tensoryvalid, reformat(label = yvalid)[1]))
def display(img):
    image_width,image_height, _ = img.shape
    # (28, 28, 1) -> (28, 28)
    image = img.reshape(image_width,image_height) 
    plt.axis('off')
    plt.imshow(image)

# output image 
show = 888
display(tensorXfull[show])
print("A normalized image of", onehot2label(tensoryfull)[show])
import tensorflow as tf

batch_size = 16
patch_size = 5
conv1_depth = 16
conv2_depth = 2*16
full_num1 = 192
full_num2 = 64
patch_stride = 1
pool_size = 2
pool_stride = 2
keep_conv = 0.8
keep_hidden = 0.5

myConvnet = tf.Graph()

with myConvnet.as_default():

  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(tensorXvalid)
  tf_test_dataset0 = tf.constant(tensordtest[:10000])
  tf_test_dataset1 = tf.constant(tensordtest[10000:20000])
  tf_test_dataset2 = tf.constant(tensordtest[20000:])
  
  # Variables.
  ceil_divide = lambda x,y: x//y+(x%y!=0) 
  layer1_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, num_channels, conv1_depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([conv1_depth]))
  # size_ratio = (patch_stride) * (pool_stride)
  layer2_weights = tf.Variable(tf.truncated_normal(
      [patch_size, patch_size, conv1_depth, conv2_depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(0.0, shape=[conv2_depth]))
  size_ratio = (patch_stride**2) * (pool_stride**2)
  layer3_weights = tf.Variable(tf.truncated_normal(
      [ceil_divide(image_size, size_ratio) * ceil_divide(image_size, size_ratio) * conv2_depth, full_num1], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(0.0, shape=[full_num1]))
  layer4_weights = tf.Variable(tf.truncated_normal(
      [full_num1, full_num2], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(0.0, shape=[full_num2]))
  layer5_weights = tf.Variable(tf.truncated_normal(
      [full_num2, num_labels], stddev=0.1))
  layer5_biases = tf.Variable(tf.constant(0.0, shape=[num_labels]))
    
  # Model.
  def model(data, dropout = False):
    conv1 = tf.nn.conv2d(data, layer1_weights, strides=[1, patch_stride, patch_stride, 1], padding='SAME')
    pooling1 = tf.nn.max_pool(conv1,ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')
    hidden1 = tf.nn.relu6(pooling1 + layer1_biases)
    if dropout:
        hidden1 = tf.nn.dropout(hidden1, keep_prob = keep_conv)
    conv2 = tf.nn.conv2d(hidden1, layer2_weights, strides=[1, patch_stride, patch_stride, 1], padding='SAME')
    pooling2 = tf.nn.max_pool(conv2,ksize=[1, pool_size, pool_size, 1], strides=[1, pool_stride, pool_stride, 1], padding='SAME')
    hidden2 = tf.nn.relu6(pooling2 + layer2_biases)
    if dropout:
        hidden2 = tf.nn.dropout(hidden2, keep_prob = keep_conv)
    shape = hidden2.get_shape().as_list()
    reshape = tf.reshape(hidden2, [shape[0], shape[1] * shape[2] * shape[3]])
    full_hidden1 = tf.nn.relu6(tf.matmul(reshape, layer3_weights) + layer3_biases)
    if dropout:
        full_hidden1 = tf.nn.dropout(full_hidden1, keep_prob = keep_hidden)
    full_hidden2 = tf.nn.relu6(tf.matmul(full_hidden1, layer4_weights) + layer4_biases)
    if dropout:
        full_hidden2 = tf.nn.dropout(full_hidden2, keep_prob = keep_hidden) 
        
    return tf.matmul(full_hidden2, layer5_weights) + layer5_biases

  # Training computation.
  logits = model(tf_train_dataset, dropout = True)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    
  # Optimizer.
  optimizer = tf.train.AdagradOptimizer(0.033).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  test_prediction0 = tf.nn.softmax(model(tf_test_dataset0))
  test_prediction1 = tf.nn.softmax(model(tf_test_dataset1))
  test_prediction2 = tf.nn.softmax(model(tf_test_dataset2))

keep_step = 0
checkpoints = [0,3000]

init = False
if checkpoints[0] == 0:
    init = True
    checkpoints = checkpoints[1:]
    
for checkpoint in checkpoints:
    with tf.Session(graph=myConvnet) as session:
        saver = tf.train.Saver()
        if init:
            tf.initialize_all_variables().run()
            print('Initialized')
            init = False
        else:
            saver.restore(session, "myConvnet_%s"%(keep_step))
            print("myConvnet_%s restored"%(keep_step))
        for step in range(keep_step, checkpoint):
            offset = (step * batch_size) % (tensorytrain.shape[0] - batch_size)
            batch_data = tensorXtrain[offset:(offset + batch_size), :, :, :]
            batch_labels = tensorytrain[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 200 == 0 or step == checkpoint-1):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), tensoryvalid))#虽然eval会再跑一次图，但跑的时候并没有feed_dict
        keep_step = checkpoint
        test_submission0 = onehot2label(test_prediction0.eval())
        test_submission1 = onehot2label(test_prediction1.eval())
        test_submission2 = onehot2label(test_prediction2.eval())
        test_submission = np.concatenate([test_submission0,test_submission1,test_submission2])
        pd.DataFrame({'ImageId': range(1, len(test_submission)+1), 'Label':test_submission}).to_csv(
            'myConvnet_%s.csv' % keep_step,index = None)
        save_path = saver.save(session, "myConvnet_%s"%(keep_step))
        print("Model saved in file: %s\n" % save_path)

print('Finished, made checkpoints %s, keep_step=%s'%(checkpoints,keep_step))
keep_step = 3000
checkpoints = [5000,7000]

init = False
if checkpoints[0] == 0:
    init = True
    checkpoints = checkpoints[1:]
    
for checkpoint in checkpoints:
    with tf.Session(graph=myConvnet) as session:
        saver = tf.train.Saver()
        if init:
            tf.initialize_all_variables().run()
            print('Initialized')
            init = False
        else:
            saver.restore(session, "myConvnet_%s"%(keep_step))
            print("myConvnet_%s restored"%(keep_step))
        for step in range(keep_step, checkpoint):
            offset = (step * batch_size) % (tensorytrain.shape[0] - batch_size)
            batch_data = tensorXtrain[offset:(offset + batch_size), :, :, :]
            batch_labels = tensorytrain[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 200 == 0 or step == checkpoint-1):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), tensoryvalid))#虽然eval会再跑一次图，但跑的时候并没有feed_dict
        keep_step = checkpoint
        test_submission0 = onehot2label(test_prediction0.eval())
        test_submission1 = onehot2label(test_prediction1.eval())
        test_submission2 = onehot2label(test_prediction2.eval())
        test_submission = np.concatenate([test_submission0,test_submission1,test_submission2])
        pd.DataFrame({'ImageId': range(1, len(test_submission)+1), 'Label':test_submission}).to_csv(
            'myConvnet_%s.csv' % keep_step,index = None)
        save_path = saver.save(session, "myConvnet_%s"%(keep_step))
        print("Model saved in file: %s\n" % save_path)

print('Finished, made checkpoints %s, keep_step=%s'%(checkpoints,keep_step))
print(check_output(["ls", "."]).decode("utf8"))



print('That\'s it, thanks.')
