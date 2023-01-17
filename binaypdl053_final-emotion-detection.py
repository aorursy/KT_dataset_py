import tensorflow as tf

from tensorflow.python.framework import tensor_shape, graph_util
from tensorflow.python.platform import gfile
import os
import os.path as path
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib


tf.test.gpu_device_name()
import numpy as np 
np.version.version
import csv
import pandas as pd 
tf.__version__
width = 48
height = 48
no_classes=7
bacth_size=16
flatten_size = width*height
MODEL_NAME = 'face expression'
NUM_STEPS = 30

tf.reset_default_graph()
import pandas as pd
data = pd.read_csv('../input/fer2013/fer2013.csv')
test_data = pd.read_csv('../input/testdata/test.csv')
df=pd.DataFrame(data)
test_df=pd.DataFrame(test_data)



        

def preprocessing():
    cnt_train=0
    cnt_test=0
    for i in range(0, len(df)):
        if (df.Usage[i]=="Training"):
            cnt_train += 1
            train_data= df
        else:
            test_data=df
            cnt_test +=1
            
    return cnt_train,cnt_test
train_size, test_size = preprocessing();
def preprocess_input(x):
    x =np.array(x, dtype=np.float32)
    x = x / 255.0
    return x
def next_batch(num):

    labels = []
    images = []
    tr_data = df.emotion
  
    idx = np.arange(0 , len(tr_data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = tr_data[idx]
    for i in idx:
        idxs = int(tr_data.iloc[i])
        label = [0, 0, 0, 0, 0, 0, 0]
        label[idxs] = 1
        labels.append(label)
        pixels = df.pixels[i]
        pixels = pixels.split(" ")
        image = np.array(pixels, dtype = np.float32)
        images.append(image)
    return  preprocess_input(images), labels
        
def next_batchs(num):

    labels = []
    images = []
    ts_data = test_df.emotion
  
    idx = np.arange(0 , len(ts_data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = ts_data[idx]
    for i in idx:
        idxs = int(ts_data.iloc[i])
        label = [0, 0, 0, 0, 0, 0, 0]
        label[idxs] = 1
        labels.append(label)
        pixels = test_df.pixels[i]
        pixels = pixels.split(" ")
        image = np.array(pixels, dtype = np.float32)
        images.append(image)
    return  preprocess_input(images), labels
        
def model_input(input_node_name):
    x = tf.placeholder(tf.float32, shape=[None, 48*48], name=input_node_name)
    
    y_ = tf.placeholder(tf.float32, shape=[None, no_classes])
    return x, y_
def build_model(x, y_, output_node_name):
    reshaped = tf.reshape(x, [-1, width, height, 1])
    weight_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32]))
    bias_conv1 = tf.Variable(tf.constant(0.1, shape = [32]))
    output_conv1 = tf.nn.relu(tf.nn.conv2d(reshaped, weight_conv1, strides=[1, 1, 1, 1], padding='SAME') + bias_conv1)
    output_maxpool1 = tf.nn.max_pool(output_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    weight_conv2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64]))
    bias_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
    output_conv2 = tf.nn.relu(tf.nn.conv2d(output_maxpool1, weight_conv2, strides = [1, 1, 1, 1], padding='SAME') + bias_conv2)
    output_maxpool2 = tf.nn.max_pool(output_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    flatten_conv = tf.reshape(output_maxpool2, [-1, 12 * 12 * 64])
    weight_fc1 = tf.Variable(tf.truncated_normal(shape=[12* 12 * 64, 1152], dtype=tf.float32))
    bias_fc1 = tf.Variable(tf.constant(0.1, shape=[1152], dtype=tf.float32))
    output_fc1 = tf.add(tf.matmul(flatten_conv, weight_fc1), bias_fc1)
    activated_output_fc1 = tf.nn.relu(output_fc1)
    weight_fc2 = tf.Variable(tf.truncated_normal(shape=[1152, 576], dtype=tf.float32))
    bias_fc2 = tf.Variable(tf.constant(0.1, shape=[576], dtype=tf.float32))
    output_fc2 = tf.add(tf.matmul(activated_output_fc1, weight_fc2), bias_fc2);
    activated_output_fc2 = tf.nn.relu(output_fc2)
    weight_output_layer = tf.Variable(tf.truncated_normal(shape=[576, no_classes], dtype=tf.float32))
    bias_output_layer = tf.Variable(tf.constant(0.1, shape=[no_classes], dtype=tf.float32))
    opt_layer = tf.add(tf.matmul(activated_output_fc2, weight_output_layer), bias_output_layer, name=output_node_name)
    # loss
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=opt_layer))

    # train step
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    # accuracy
    correct_prediction = tf.equal(tf.argmax(opt_layer, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy", accuracy)
    merged_summary_op = tf.summary.merge_all()

    return train_step, loss, accuracy, merged_summary_op
def train(x, y_, train_step, loss, accuracy,
        merged_summary_op, saver):
    print("training start...")

    

    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        tf.train.write_graph(sess.graph_def, 'out',
            MODEL_NAME + '.pbtxt', True)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter('logs/',
            graph=tf.get_default_graph())

        for step in range(500):
            for i in range(int(train_size/128)):
                x_train, y_train = next_batch(128)
            
                
                _, summary = sess.run([train_step, merged_summary_op],
                feed_dict={x: x_train, y_: y_train})
                summary_writer.add_summary(summary, step)
            if step % 1 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                x: x_train, y_: y_train})
                print('step %d, training accuracy %f' % (step, train_accuracy))
             
            
               

            saver.save(sess, 'out/' + MODEL_NAME + '.chkp')
            
            for i in range(int(test_size/128)):
                x_test, y_test = next_batchs(128) 
            
                test_accuracy = accuracy.eval(feed_dict={x: x_test,
                                    y_:y_test,
                                    })
            print('test accuracy %g' % test_accuracy)

    print("training finished!")
def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, input_node_names, [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")
def main():
    
    
    

    input_node_name = 'input'
   
    output_node_name = 'output'

    x, y_ = model_input(input_node_name)

    train_step, loss, accuracy, merged_summary_op = build_model(x,
        y_, output_node_name)
    saver = tf.train.Saver()

    train(x, y_, train_step, loss, accuracy,
        merged_summary_op, saver)

    export_model([input_node_name], output_node_name)
if __name__ == '__main__':
    main()




