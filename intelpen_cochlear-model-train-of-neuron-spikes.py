# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.



import cochlea

from scipy.io import wavfile



import numpy as np

import scipy

import pandas as pd

#import os

#import multiprocessing



def accumulate(spike_trains, ignore=None, keep=None):

    """Concatenate spike trains with the same meta data. Trains will

    be sorted by the metadata.

    """



    assert None in (ignore, keep)



    keys = spike_trains.columns.tolist()



    if ignore is not None:

        for k in ignore:

            keys.remove(k)



    if keep is not None:

        keys = keep



    if 'duration' not in keys:

        keys.append('duration')



    if 'spikes' in keys:

        keys.remove('spikes')





    groups = spike_trains.groupby(keys, as_index=False)



    acc = []

    for name,group in groups:

        if not isinstance(name, tuple):

            name = (name,)

        spikes = np.concatenate(tuple(group['spikes']))

        acc.append(name + (spikes,))



    columns = list(keys)

    columns.append('spikes')



    acc = pd.DataFrame(acc, columns=columns)



    return acc







def extractSpikes(filename , channelsNo = 128 , colsNo = 512):

  fs, samples = wavfile.read(filename)

  norm = (500.0)/np.max(np.abs(samples))

  samples = norm*samples  

  samples = cochlea.set_dbspl(samples,50)

  fs100kHz = 100e3

  down = fs*100/fs100kHz 

  samples100kHz =scipy.signal.resample_poly(samples,up = 100, down = down)

  samples100kHz = cochlea.set_dbspl(samples100kHz,50)

  anf_trains = cochlea.run_zilany2014(sound= samples100kHz, fs = fs100kHz, anf_num = (0,0,50), cf= (125,10e3,channelsNo) ,seed = 0,

                                    powerlaw="approximate",  species = 'human' )

  anf_trains= accumulate(anf_trains)

  spikes = anf_trains["spikes"].as_matrix()

  spikesImage = np.zeros(shape = [channelsNo, colsNo])

  maxTime=np.concatenate(spikes).max()

  for index in range(spikes.shape[0]):

    spikesNorm = (colsNo-1)*spikes[index]/maxTime

    spikesInt = spikesNorm.astype(dtype = np.int)

    spikesImage[index, spikesInt] = 1 

  #plt.figure()

  #plt.imshow(spikesImage)

  #plt.show()

  return spikesImage

def extractSpikesParallel(arguments):

  try :

    filenameIn, filenameOut = arguments

    print(filenameIn)

    print(filenameOut)

    spikesTrain = extractSpikes(filenameIn, channelsNo = 128 , colsNo = 512)

    np.savetxt(fname =filenameOut, X =spikesTrain,fmt="%d", delimiter = ";")  

    print("finished"+filenameOut)

  except :

    print("There was an error here")

  return 1






from scipy import misc

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.python.framework import ops

from tf_utils import load_dataset, random_mini_batchesT, convert_to_one_hot

import sys

#np.random.seed(1)

import time



labelsList = ["_silence_","_unknown_","yes","no","up","down","left","right","on","off","stop","go"]

noClasses = len(labelsList)

#%%

class Params:

  imageWidth = 512

  imageHeight = 128

  def __init__(self, imageWidth = 512, imageHeight = 128):

    self.imageWidth = imageWidth

    self.imageHeight = imageHeight





def load_dataset(rootDir, labelsList):

  print(rootDir)

  X = []

  Y = []  

  for label in os.listdir(rootDir):

    

    dirLabel = rootDir + "//" + label

    print('Found directory: %s' % dirLabel)

    index = 0

    for filename in os.listdir(dirLabel):

       if index % 1 == 0 :

          #print('\t%s' % fname)

          filename = dirLabel+"//"+ filename

          image= np.zeros(shape = [128,512], dtype = np.uint8)

          img = pd.read_csv(filename, sep=";").as_matrix()

          image[0:127,:] = img

          

          X.append(image)

          yL = np.zeros(shape = [noClasses])

          yL[labelsList.index(label)]=1

          Y.append(yL)

          

          #if index > 500:

          #  break

       index = index +1

  X = np.array(X)

  Y= np.array(Y)

  

  print(X.shape)

  print(Y.shape)

  

  permutation = list(np.random.permutation(X.shape[0]))

  X = X[permutation,:]

  Y = Y[permutation,:]

  

  no_examples =X.shape[0] 

  X_Train= X[0:int(0.90*no_examples),:]

  Y_Train =Y[0:int(0.90*no_examples),:] 

  X_Test = X[int(0.90*no_examples):int(1*no_examples),:]

  Y_Test = Y[int(0.90*no_examples):int(1*no_examples),:]

  classes = labelsList

  return X_Train, Y_Train, X_Test, Y_Test, classes



def deepnn(x):

  noClasses=12

  """deepnn builds the graph for a deep net for classifying digits.  Args:

    x: an input tensor with the dimensions (N_examples, 3*92*46), where 784 is the

    number of pixels in a standard MNIST image.

  Returns:

    A tuple (y, keep_prob). y is a tensor of shape (N_examples, noClasses), with values equal to the logits of classifying the imagesinto one of 2 classes .

    keep_prob is a scalar placeholder for the probability of     dropout.

  """

  # Reshape to use within a convolutional neural net.   # Last dimension - it would be 3 for an RGB image, 4 for RGBA, etc.

  XFloat = tf.cast(x, dtype = tf.float32)

  #XNorm = tf.scalar_mul(1.0/255.0, XFloat)

  

  with tf.name_scope('reshape'):

    x_image = tf.reshape(XFloat, [-1, 128, 512, 1]) #was 28 28, 28, 1

  # First convolutional layer - maps one grayscale image to 32 feature maps.

  with tf.name_scope('conv1'):

    W_conv1 = weight_variable([5, 5, 1, 16]) #W_conv1 = weight_variable([5, 5, 1, 32]) try fatter features

    b_conv1 = bias_variable([16])

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)



  # Pooling layer - downsamples by 2X.

  with tf.name_scope('pool1'):

    #h_pool1= tf.sqrt(tf.nn.avg_pool(tf.square(h_conv1),ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'))

    h_pool1 = max_pool_2x2(h_conv1)



  # Second convolutional layer -- maps 32 feature maps to 64.

  with tf.name_scope('conv2'):

    W_conv2 = weight_variable([8, 8, 16, 32]) #W_conv2 = weight_variable([5, 5, 32, 64])

    b_conv2 = bias_variable([32])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)



  # Second pooling layer.

  with tf.name_scope('pool2'):

    #h_pool2= tf.sqrt(tf.nn.avg_pool(tf.square(h_conv2),ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'))

    h_pool2 = max_pool_2x2(h_conv2)





# third convolutional layer -- maps 32 feature maps to 64.

  with tf.name_scope('conv3'):

    W_conv3 = weight_variable([5, 5, 32, 64]) #W_conv2 = weight_variable([5, 5, 32, 64])

    b_conv3 = bias_variable([64])

    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)



  # Second pooling layer.

  with tf.name_scope('pool3'):

    #h_pool2= tf.sqrt(tf.nn.avg_pool(tf.square(h_conv2),ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME'))

    h_pool3 = max_pool_2x2(h_conv3)





  # Fully connected layer 1 -- after 2 round of downsampling, our 128x512 image

  # is down to 16x64x64 feature maps -- maps this to 1024 features.

  with tf.name_scope('fc1'):

    W_fc1 = weight_variable([16*64*64, 512])

    b_fc1 = bias_variable([512])



    h_pool2_flat = tf.reshape(h_pool3, [-1,16*64*64])

    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



  # Dropout - controls the complexity of the model, prevents co-adaptation of

  # features.

  with tf.name_scope('dropout'):

    keep_prob = tf.placeholder(tf.float32, name = "keep_prob")

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)



  # Map the 1024 features to 10 classes, one for each digit

  with tf.name_scope('fc2'):

    W_fc2 = weight_variable([512, noClasses])

    b_fc2 = bias_variable([noClasses])

    y_conv = tf.add(tf.matmul(h_fc1_drop, W_fc2),b_fc2, name = "Y_Conv")

  return y_conv, keep_prob

def conv2d(x, W):

  """conv2d returns a 2d convolution layer with full stride."""

  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):

  """max_pool_2x2 downsamples a feature map by 2X."""

  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],

                        strides=[1, 2, 2, 1], padding='SAME')

def max_pool_4x4(x):

  """max_pool_2x2 downsamples a feature map by 2X."""

  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],

                        strides=[1, 4, 4, 1], padding='SAME')

  

def weight_variable(shape):

  """weight_variable generates a weight variable of a given shape."""

  initial = tf.truncated_normal(shape, stddev=0.1)

  return tf.Variable(initial)

def bias_variable(shape):

  """bias_variable generates a bias variable of a given shape."""

  initial = tf.constant(0.1, shape=shape)

  return tf.Variable(initial)

#%%

  



def initParameters():

  global PARAMS

  PARAMS = Params()

  PARAMS.imageHeight = 128

  PARAMS.imageWidth = 512



def one_hot_matrix(labels, C):

  """ Creates a matrix where the i-th row corresponds to the ith class number and the jth column  corresponds to the jth training example. So if example j had a label i. Then entry (i,j)  will be 1. 

  Arguments:

  labels -- vector containing the labels 

  C -- number of classes, the depth of the one hot dimension

  Returns: 

  one_hot -- one hot matrix

  """

  C = tf.constant(C, name = "C")

  # Use tf.one_hot, be careful with the axis 

  one_hot_matrix = tf.one_hot( labels, depth = C, axis = 0 ) #you can receive variables without a placeholder ... probably if used once

  sess = tf.Session()

  one_hot = sess.run(one_hot_matrix )

  sess.close()

  return one_hot



def ones(shape):

  """

  Creates an array of ones of dimension shape

  Arguments:

  shape -- shape of the array you want to create

  Returns: 

  ones -- array containing only ones

  """

  ones = tf.ones(shape)

  sess = tf.Session()

  ones = sess.run(ones)

  sess.close()

  return ones



def create_placeholders( n_y):

  """  Creates the placeholders for the tensorflow session.

  Arguments:

  n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)

  n_y -- scalar, number of classes (from 0 to 5, so -> 6)

  Returns:

  X -- placeholder for the data input, of shape [n_x, None] and dtype "float"

  Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

  Tips:     - " None" let's us be flexible on the number of examples you will for the placeholders.  In fact, the number of examples during test/train is different.

  """

  X = tf.placeholder(dtype = tf.uint8, shape=(None,128,512), name = "X")

  Y = tf.placeholder(dtype = tf.float32, shape=(None,n_y), name = "Y")

  return X, Y



def compute_cost(Y_conv, Y):

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Y_conv, labels = Y))

    return cost



def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 1500, minibatch_size = 32, print_cost = True, tfModelSavePath =""):

    """     Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:

    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)

    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)

    X_test -- training set, of shape (input size = 12288, number of training examples = 120)

    Y_test -- test set, of shape (output size = 6, number of test examples = 120)

    learning_rate -- learning rate of the optimization

    num_epochs -- number of epochs of the optimization loop

    minibatch_size -- size of a minibatch

    print_cost -- True to print the cost every 100 epochs

    Returns:

    parameters -- parameters learnt by the model. They can then be used to predict.

    """

    tf.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

    tf.set_random_seed(1)                             # to keep consistent results

    seed = 3                                          # to keep consistent results

    m,n_y = Y_train.shape                            # n_y : output size

    costs = []                                        # To keep track of the cost

    train_accuracys = []                                        # To keep track of the cost

    test_accuracys = []                                        # To keep track of the cost

    # Create Placeholders of shape (n_x, n_y)    

    X, Y = create_placeholders(n_y)

    # Forward propagation: Build the forward propagation in the tensorflow graph

    y_conv, keep_prob = deepnn(X)

    # Cost function: Add cost function to tensorflow graph

    cost = compute_cost(y_conv,Y) 

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.

    #optimizer =  tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    optimizer =  tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    # Initialize all the variables

    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph

    correct_prediction = tf.equal(tf.argmax(y_conv, axis =1), tf.argmax(Y, axis = 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

        saver = tf.train.Saver() #SAVE

        # Run the initialization

        sess.run(init)

        curr_time = time.time()

        # Do the training loop

        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch

            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set

            seed = seed + 1

            minibatches = random_mini_batchesT(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:

                # Select a minibatch

                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch. Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).

                _ , minibatch_cost = sess.run([optimizer,cost], feed_dict = {X:minibatch_X, Y:minibatch_Y , keep_prob : 0.2 })

                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch

            if print_cost == True and epoch % 2 == 0:

                new_time = time.time()

                print("Time/epoch:"+str((new_time-curr_time)/2))

                curr_time=new_time

                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

            if print_cost == True and epoch % 5 == 0:

                costs.append(epoch_cost)

                (minibatch_X, minibatch_Y) = minibatches[0]

                train_accuracys.append(accuracy.eval({X: minibatch_X, Y: minibatch_Y, keep_prob : 1}))

                test_accuracys.append(accuracy.eval({X: X_test[0:128,:], Y: Y_test[0:128  ,:], keep_prob : 1}))

                plotAccuracys(train_accuracys,test_accuracys,learning_rate)

                plotCosts(costs,learning_rate)

        saver.save(sess, tfModelSavePath)

        tf.train.write_graph(sess.graph_def, tfModelSavePath , "ModelTFPerson.pb", as_text=True)

        #plot the cost

        plotCosts(costs,learning_rate)

        

        print ("Parameters have been trained!")

        #print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test, keep_prob : 1}))

        

def predictOnly(X_test ):

  

  tf.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables

  

  tf.set_random_seed(1)                             # to keep consistent results

  seed = 3                                          # to keep consistent results

  (m,n_x) = X_test.shape                          # (n_x: input size, m : number of examples in the train set)

  costs = []                                        # To keep track of the cost

  # Create Placeholders of shape (n_x, n_y)    

  X = tf.placeholder(dtype = tf.uint8, shape=(None, n_x))

  # Forward propagation: Build the forward propagation in the tensorflow graph

  y_conv, keep_prob = deepnn(X)

  init = tf.global_variables_initializer()

  # Start the session to compute the tensorflow graph

  with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

      sess.run(init)

      tfModelSavePath = "D:\\Dev\\ML\\PersonDetection\\NeuralNet3LTensorFlow\\TFSavedModels"

      saver = tf.train.Saver()

      saver = tf.train.import_meta_graph(tfModelSavePath +"\\.meta")   # Load :  Important

      saver.restore(sess, tfModelSavePath + "\\" ) # Load :  Important

      start_time = time.time()

      for i in range (100):

        yOut =  sess.run([y_conv], feed_dict = {X:X_test, keep_prob : 1 })

      end_time = time.time()

      print("Elapsed"+str(end_time-start_time))

  return yOut



        

def plotCosts(costs, learning_rate):

  plt.plot(np.squeeze(costs))

  plt.ylabel('Cost')

  plt.xlabel('iterations (per tens)')

  plt.title("Learning rate =" + str(learning_rate))

  plt.show()



def plotAccuracys(train_accuracys,test_accuracys,learning_rate):

  plt.plot(np.squeeze(train_accuracys))

  plt.plot(np.squeeze(test_accuracys))

  plt.ylabel('Accuracy')

  plt.xlabel('iterations (per tens)')

  plt.title("Learning rate =" + str(learning_rate))

  plt.show()



def printOutput(X, Y , Y_True): 

  

  for i in range(X.shape[0]):

    if Y[i]!=Y_True[i]:

      plt.imshow(np.reshape(X[i], newshape=[92,46,3]))

      plt.show()

      im=np.reshape(X[i], newshape=[92,46,3])

      print("Y_pred="+str(Y[i])+ "Y_true"+str(Y_True[i]))

      misc.imsave("D://data//ML//ImageSegmentation//out//True"+str(Y_True[i])+"//Pred"+str(Y[i])+"_" +str(i)+".jpg",im)

  

#%%

#%%

# Loading the dataset



rootDir = "E://data//kaggle//TFSpeechRC//train//spikes128x512"  

if not('X_train_orig' in locals()):

  X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(rootDir, labelsList)

# Change the index below and run the cell to visualize some examples in the dataset.



#%%





# Flatten the training and test images

X_train = X_train_orig #/255.

X_test = X_test_orig #/255.

Y_train = Y_train_orig

Y_test = Y_test_orig

print ("number of training examples = " + str(X_train.shape[0]))

print ("number of test examples = " + str(X_test.shape[0]))

print ("X_train shape: " + str(X_train.shape))

print ("Y_train shape: " + str(Y_train.shape))

print ("X_test shape: " + str(X_test.shape))

print ("Y_test shape: " + str(Y_test.shape))



#%%

# Example of a picture

index = 300

plt.imshow(X_train_orig[index])

plt.show()

print ("y = " + str(Y_train_orig[index])+labelsList[np.argmax(Y_train_orig[index])] )



index = 100

plt.imshow(X_train_orig[index])

plt.show()

print ("y = " + str(Y_train_orig[index])+labelsList[np.argmax(Y_train_orig[index])] )





index = 15

plt.imshow(X_train_orig[index])

plt.show()

print ("y = " + str(Y_train_orig[index])+labelsList[np.argmax(Y_train_orig[index])] )







index = 4

plt.imshow(X_test[index])

plt.show()

print ("y = " + str(Y_test[index])+labelsList[np.argmax(Y_test[index])] )



#%% def trainAndSave():

tfModelSavePath = "E://science//info//Competitions//Kaggle//TFSpeechRC//SavedModel//"

model(X_train, Y_train, X_test, Y_test,learning_rate = 1e-3,num_epochs = 50, minibatch_size = 64, print_cost = True,tfModelSavePath =tfModelSavePath )

labelsList = ["silence","unknown","yes","no","up","down","left","right","on","off","stop","go"]

#%%

def main():

  

  audioFilesDir = "E://data//kaggle//TFSpeechRC//test//audio//"

  spikesFilesDir = "E://data//kaggle//TFSpeechRC//test//spikes128x512//"

  #spikesFilesDir = "E://data//kaggle//TFSpeechRC//train//spikes128x512//off//"

  index = 0

  

  tf.reset_default_graph()   

  sess= tf.Session() 

  #model is restored here

  tfModelSavePath = "E://science//info//Competitions//Kaggle//TFSpeechRC//SavedModel"

  saver = tf.train.import_meta_graph("E://science//info//Competitions//Kaggle//TFSpeechRC//SavedModel//.meta")   # Load :  Important

  saver.restore(sess, tfModelSavePath + "\\" ) # Load :  Important

  graph = tf.get_default_graph() # Load :  Important

  x = graph.get_tensor_by_name("X:0") # Load :  Important

  keep = graph.get_tensor_by_name("dropout/keep_prob:0") # Load :  Important

  Y_conv = graph.get_tensor_by_name("fc2/Y_Conv:0") 

  

  

  outL = []

  fileList = os.listdir(spikesFilesDir)

  

  for file in fileList:

    index = index +1 

    if index % 1 == 0 :

      

      if (index%1000==0):

        print("Index" + str(index) +" : "+str(file))

      image= np.zeros(shape = [1,128,512], dtype = np.uint8)

      img = pd.read_csv(spikesFilesDir + file, sep=";").as_matrix()

      image[0,0:127,:] = img

      

      #plt.imshow(image[0])

      #plt.show()

      

     

      feed_dict ={x:image,keep:1.0 }  

      # Feed the audio data as input to the graph.    #   predictions  will contain a two-dimensional array, where one         #   dimension represents the input image count, and the other has     #   predictions per class

      YConv = sess.run(Y_conv, feed_dict)

      # Sort to show labels in order of confidence

      top_k = np.argmax(YConv)

      #print(YConv)

      human_string = labelsList[top_k]

      outL.append([file[0:-4],human_string])

      #print(file[0:-4] + human_string)

  df = pd.DataFrame(outL,index =None, columns = ["fname","label"])

  #print(df)

  df.to_csv("my_sample_submission3.csv",index = False)



      

        

if __name__ == '__main__':

  main()