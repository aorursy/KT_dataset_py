import tensorflow as tf
tf.test.gpu_device_name()
import numpy as np 

import csv
import pandas as pd 
width = 48
height = 48
no_classes=7
bacth_size=128
flatten_size = width*height

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
    return images, labels
        



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
    return images, labels
        

train_size
x= tf.placeholder(tf.float32, shape=[None, flatten_size], name='input_image')
y= tf.placeholder(tf.float32, shape=[None, no_classes], name='input_lables')
def neural_network_model(input_x):
    reshaped = tf.reshape(input_x, [-1, width, height, 1])
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
    opt_layer = tf.add(tf.matmul(activated_output_fc2, weight_output_layer), bias_output_layer)
    
    return opt_layer
        
    
    
    
    


  
def build_and_train_model(x):
    pred_res = neural_network_model(x)
   
    
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = pred_res, labels = y))
    optimizer = tf.train.AdamOptimizer(3e-4).minimize(cross_entropy)
    truth_table_bool = tf.equal(tf.argmax(pred_res, 1), tf.argmax(y, 1));
    truth_table_int = tf.cast(truth_table_bool, tf.int32)
    correct_preds = tf.reduce_sum(truth_table_int)
    epochs = 1
    saver = tf.train.Saver()
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        for ep in range(epochs):
            total_cost = 0
            print(ep)
            
            for i in range(int(train_size/128)):
                x_train, y_train = next_batch(128)
                cost= sess.run([optimizer, cross_entropy], feed_dict = {x : x_train, y : y_train})
                if(i%100==0):
                    x_test, y_test = next_batchs(128) 
                    acc=sess.run(correct_preds, feed_dict={x : x_test, y : y_test})
                    print("accuray",acc)
                    print("step",i)
                    
        print("pred_res",pred_res)
                    
                
                
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)      
               
           
            
                
                
                
        

build_and_train_model(x)
# def export_model(input_node_names, output_node_name):
#     freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
#         'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
#         "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

#     input_graph_def = tf.GraphDef()
#     with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
#         input_graph_def.ParseFromString(f.read())

#     output_graph_def = optimize_for_inference_lib.optimize_for_inference(
#             input_graph_def, input_node_names, [output_node_name],
#             tf.float32.as_datatype_enum)

#     with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
#         f.write(output_graph_def.SerializeToString())

#     print("graph saved!")

