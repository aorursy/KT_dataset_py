# # import the kaggle.json file to download the dataset from the API
# from google.colab import files
# files.upload()
#######to mount data from Google drive########
# from google.colab import drive
# drive.mount('/content/drive')
# pip install --upgrade pip
# to easy download your dataset without errors use kaggle==1.5.6 version
# pip install kaggle==1.5.6
#  ! mkdir ~/.kaggle
# ! cp kaggle.json ~/.kaggle/
#  ! chmod 600 ~/.kaggle/kaggle.json
# ! kaggle datasets list
# ! kaggle competitions download -c 'osic-pulmonary-fibrosis-progression'
# ! mkdir train
# ! unzip /content/osic-pulmonary-fibrosis-progression.zip -d train
# import pydicom to read dcm images 
# !pip install pydicom 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import os 
import pydicom
testing = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')
testing
testing.isnull().sum()
data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv',index_col=0)
labels_df.head()
testing.isnull().sum()
for patient in patients[:2]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices),label)
    print(slices[0])
plt.bar(labels_df['Sex'],labels_df['FVC'])
plt.bar(labels_df['SmokingStatus'],labels_df['FVC'],color='red')
plt.hist(labels_df['FVC'])
plt.plot(labels_df['FVC'])
for patient in patients[:10]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
      print(slices[0].pixel_array.shape, len(slices))
    except:
      print('')
len(patients)
import matplotlib.pyplot as plt

for patient in patients[:5]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    
    #          the first slice
    try:
      plt.imshow(slices[0].pixel_array)
      plt.show()
    except:
      print('None')
import cv2
import numpy as np

IMG_PX_SIZE = 50

for patient in patients[:1]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    fig = plt.figure()
    for num,each_slice in enumerate(slices[:12]):
        y = fig.add_subplot(3,4,num+1)
        new_img = cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE))
        y.imshow(new_img)
    plt.show()
import math
def chunks(l, n):
    # Credit: Ned Batchelder
    # Link: http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def mean(l):
    return sum(l) / len(l)

IMG_PX_SIZE = 50
hm_slices = 20

data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv',index_col=0)

for patient in patients[:10]:
    try:
        label = labels_df._get_value(patient, 'FVC')
        path = data_dir + patient
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        new_slices = []
        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
        chunk_sizes = math.ceil(len(slices) / hm_slices)
        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)

        print(len(slices), len(new_slices))
    except:
        # some patients don't have labels, so we'll just pass on this for now
        pass
len(patients)

for patient in patients[:10]:
    try:
        label = labels_df._get_value(patient, 'FVC')
        path = data_dir + patient
        slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
        slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
        new_slices = []

        slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]

        chunk_sizes = math.ceil(len(slices) / hm_slices)


        for slice_chunk in chunks(slices, chunk_sizes):
            slice_chunk = list(map(mean, zip(*slice_chunk)))
            new_slices.append(slice_chunk)

        if len(new_slices) == hm_slices-1:
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-2:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-3:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-4:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-5:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])            
            new_slices.append(new_slices[-1]) 

        if len(new_slices) == hm_slices-6:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-7:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])

        if len(new_slices) == hm_slices-8:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
        if len(new_slices) == hm_slices-9:
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1]) 
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])
            new_slices.append(new_slices[-1])                     
            new_slices.append(new_slices[-1])


        if len(new_slices) == hm_slices+2:
            new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
            del new_slices[hm_slices]
            new_slices[hm_slices-1] = new_val

        if len(new_slices) == hm_slices+1:
            new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
            del new_slices[hm_slices]
            new_slices[hm_slices-1] = new_val

        print(len(slices), len(new_slices))
    except Exception as e:
        # again, some patients are not labeled, but JIC we still want the error if something
        # else is wrong with our code
        print(str(e))
for patient in patients[:15]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    
    # a couple great 1-liners from: https://www.kaggle.com/gzuidhof/data-science-bowl-2017/full-preprocessing-tutorial
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    print(len(slices),label)
    # print(slices[0])
for patient in patients[:3]:
    label = labels_df._get_value(patient, 'FVC')
    path = data_dir + patient
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    new_slices = []

    slices = [cv2.resize(np.array(each_slice.pixel_array),(IMG_PX_SIZE,IMG_PX_SIZE)) for each_slice in slices]
    
    chunk_sizes = math.ceil(len(slices) / hm_slices)
    for slice_chunk in chunks(slices, chunk_sizes):
      slice_chunk = list(map(mean, zip(*slice_chunk)))
      new_slices.append(slice_chunk)

    if len(new_slices) == hm_slices-1:
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-2:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-3:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-4:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-5:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])            
      new_slices.append(new_slices[-1]) 
    if len(new_slices) == hm_slices-6:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-7:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-8:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices-9:
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1]) 
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])
      new_slices.append(new_slices[-1])                     
      new_slices.append(new_slices[-1])
    if len(new_slices) == hm_slices+2:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
        
    if len(new_slices) == hm_slices+1:
        new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
        del new_slices[hm_slices]
        new_slices[hm_slices-1] = new_val
    
    fig = plt.figure()
    for num,each_slice in enumerate(new_slices):
        y = fig.add_subplot(4,5,num+1)
        y.imshow(each_slice, cmap='gray')
    plt.show()
IMG_SIZE_PX=20
SLICE_COUNT=10


def chunks(l, n):
  for i in range(0, len(l), n):
    yield l[i:i + n]


def mean(l):
  return sum(l) / len(l)

def process_data(patient,labels_df,img_px_size=10, hm_slices=10, visualize=False):
  label = labels_df._get_value(patient, 'FVC')
  path = data_dir + patient
  slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
  try:
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
  except:
    print('')
  new_slices = []
  try:
    slices = [cv2.resize(np.array(each_slice.pixel_array),(img_px_size,img_px_size)) for each_slice in slices]
  except:
    return [0,0]
  chunk_sizes = math.ceil(len(slices) / hm_slices)
  for slice_chunk in chunks(slices, chunk_sizes):
    slice_chunk = list(map(mean, zip(*slice_chunk)))
    new_slices.append(slice_chunk)

  if len(new_slices) == hm_slices-1:
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-2:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-3:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-4:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-5:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])            
    new_slices.append(new_slices[-1]) 

  if len(new_slices) == hm_slices-6:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-7:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices-8:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
  if len(new_slices) == hm_slices-9:
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1]) 
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])
    new_slices.append(new_slices[-1])                     
    new_slices.append(new_slices[-1])

  if len(new_slices) == hm_slices+2:
    new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
    del new_slices[hm_slices]
    new_slices[hm_slices-1] = new_val
        
  if len(new_slices) == hm_slices+1:
    new_val = list(map(mean, zip(*[new_slices[hm_slices-1],new_slices[hm_slices],])))
    del new_slices[hm_slices]
    new_slices[hm_slices-1] = new_val

  if visualize:
    fig = plt.figure()
    for num,each_slice in enumerate(new_slices):
      y = fig.add_subplot(4,5,num+1)
      y.imshow(each_slice, cmap='gray')
      plt.show()

  if label.all == 1: label=np.array([0,1])
  elif label.all == 0: label=np.array([1,0])
        
  return np.array(new_slices),label

#                                               stage 1 for real.
data_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
patients = os.listdir(data_dir)
labels_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv',index_col=0)


much_data = []
for num,patient in enumerate(patients):
    if num % 100 == 0:
        print(num)
    try:
        img_data,label = process_data(patient,labels_df,img_px_size=IMG_SIZE_PX, hm_slices=SLICE_COUNT)
        #print(img_data.shape,label)
        much_data.append([img_data,label])
    except KeyError as e:
        print('This is unlabeled data!')

np.save('much_data-{}-{}-{}.npy'.format(IMG_SIZE_PX,IMG_SIZE_PX,SLICE_COUNT), much_data)
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np

IMG_PXL_SIZE = 10
SLICE_COUNT = 10

n_classes = 2
batch_size = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

keep_rate = 0.8
def conv3d(x, W):
  return tf.nn.conv3d(x, W, strides=[1,1,1,1,1], padding='SAME')

def maxpool3d(x):
  return tf.nn.max_pool3d(x, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME')
def convolutional_neural_network(x):
    #                # 5 x 5 x 5 patches, 1 channel, 32 features to compute.
    weights = {'W_conv1':tf.Variable(tf.random_normal([3,3,3,1,32])),
               #       5 x 5 x 5 patches, 32 channels, 64 features to compute.
               'W_conv2':tf.Variable(tf.random_normal([3,3,3,32,64])),
               #                                  64 features
               'W_fc':tf.Variable(tf.random_normal([13824  ,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    #                            image X      image Y        image Z
    x = tf.reshape(x, shape=[-1, 10, 10, 10, 1])

    conv1 = tf.nn.relu(conv3d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool3d(conv1)


    conv2 = tf.nn.relu(conv3d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool3d(conv2)

    fc = tf.reshape(conv2,[-1, 13824])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output = tf.matmul(fc, weights['out'])+biases['out']

    return output
much_data = np.load('./much_data-20-20-10.npy',allow_pickle=True)
# len(much_data)

much_data[:-2]
much_data = np.load('./much_data-20-20-10.npy',allow_pickle=True)
# # If you are working with the basic sample data, use maybe 2 instead of 100 here... you don't have enough data to really do this
train_data = much_data[:-2]
validation_data = much_data[-2:]

def train_neural_network(x):
  prediction = convolutional_neural_network(x)
  cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = prediction) )
  optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
  hm_epochs = 5
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    successful_runs = 0
    total_runs = 0
        
    for epoch in range(hm_epochs):
      epoch_loss = 0
      for data in train_data:
        total_runs += 1
        try:
          X = data[0]
          Y = data[1]
          _, c = sess.run([optimizer, cost], feed_dict={x: X, y: Y})
          epoch_loss += c
          successful_runs += 1
        except Exception as e:
          pass
          # print(str(e))
            
      print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)

      correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

      print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[0] for i in validation_data]}))      

    print('Done. Finishing accuracy:')
    print('Accuracy:',accuracy.eval({x:[i[0] for i in validation_data], y:[i[0] for i in validation_data]}))    
    print('fitment percent:',successful_runs/total_runs)

# Run this locally:
# train_neural_network(x)
