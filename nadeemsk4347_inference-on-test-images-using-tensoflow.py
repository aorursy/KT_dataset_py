# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

print(tf.__version__

     )
import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



BATCH_SIZE = 200



graph = tf.Graph()



testx = np.load('/kaggle/input/classifying-flowers-with-tensorflow/testx.npy')

testy = np.load('/kaggle/input/classifying-flowers-with-tensorflow/testy.npy')





with graph.as_default():

    with tf.Session(graph=graph) as sess:

        tf.saved_model.loader.load(sess=sess, tags=['tag'], export_dir='/kaggle/input/classifying-flowers-with-tensorflow/')



        x = graph.get_tensor_by_name('x:0')

        y_preds = graph.get_tensor_by_name('y_preds:0')

        

        y_true = []

        preds = []

        for batch in range(int(len(testx)/BATCH_SIZE)):

            batch_x = testx[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(testx))]

            batch_y = testy[batch*BATCH_SIZE:min((batch+1)*BATCH_SIZE,len(testy))]



            y_true.append(batch_y)

            preds.append(sess.run(y_preds, feed_dict={x: batch_x}))



        y_true = np.stack(np.array(y_true), axis=0)

        preds = np.stack(np.array(preds), axis=0)

# Calculte loss and accuracy

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.cast(preds, tf.float32), 

                                                              labels=tf.cast(y_true, tf.float32)))

correct = tf.equal(np.argmax(preds, axis=2), np.argmax(y_true, axis=2))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



with tf.Session() as sess:

    print('Loss :',loss.eval())

    print('Accuracy :', accuracy.eval())