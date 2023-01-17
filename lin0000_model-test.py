## setting version

!pip install tensorflow==1.13.1
# %load ../input/109-sensor/ModelTest.py

import warnings

warnings.filterwarnings('ignore')

import skimage

from skimage import io,transform

import os

os.environ["CUDA_VISIBLE_DEVICES"]="-1"    

import tensorflow as tf

import numpy as np

from os import walk

from os.path import join



print(tf.__version__)

path = "../input/109-sensor/test-image/"

model_meta = '../input/model-result/model_result/model.ckpt.meta'

model_path = '../input/model-result/model_result/'

data_dict = {0:'ST',1:'ZN'}

w=100

h=100

c=3



global fullpath



def read_one_image(path):

    img = io.imread(path)

    img = transform.resize(img,(w,h))

    return np.asarray(img)



path_=[]	

for root, dirs, files in walk(path):

  for f in files:

    fullpath = join(root, f)

    print(fullpath)

    path_.append(fullpath)

    

with tf.Session() as sess:

    data = []

    for i in range(len(path_)):

        # print(path_[i])

        data1 = read_one_image(path_[i])

        data.append(data1)



    saver = tf.train.import_meta_graph(model_meta)

    saver.restore(sess,tf.train.latest_checkpoint(model_path))



    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")

    feed_dict = {x:data}



    logits = graph.get_tensor_by_name("logits_eval:0")



    classification_result = sess.run(logits,feed_dict)



    #print(classification_result)

	

    #print(tf.argmax(classification_result,1).eval())

    # print(classification_result)

    output = []

    output = tf.argmax(classification_result,1).eval()

    

    f = open('./answer.txt','w')

    for i in range(len(output)):

        print("這",i+1,"張圖片预测:"+data_dict[output[i]])

        f.write('{}'.format(i+1)+'\t'+data_dict[output[i]])

        f.write('\n')

    f.close()

anser = open('./answer.txt','r')

for i in anser:

    print(i)
