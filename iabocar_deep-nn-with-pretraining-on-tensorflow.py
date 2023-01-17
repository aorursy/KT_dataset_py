%matplotlib inline

import tensorflow as tf

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



train_data = pd.read_csv('../input/train.csv').values

test_data = pd.read_csv('../input/test.csv').values

dataX = train_data[:,1:];

dataYdense = train_data[:,0:1];



def dense_to_one_hot(labels_dense, num_classes):

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot

dataY = dense_to_one_hot(dataYdense, 10 )

dataX= dataX/np.max(dataX)



test_data = test_data/np.max(dataX)
train_dataX = dataX[:-1000]

cross_test_dataX = dataX[-1000:]

train_dataY = dataY[:-1000]

cross_test_dataY = dataY[-1000:]

cross_test_classes = dataYdense[-1000:]
starter_learning_rate = 0.01

trainning_epochs=12

batch_size = 512

display_step = 10



TRAINSIZE = tf.constant( np.float32(train_dataX.shape[0]))

LAMBDA = tf.constant(0.00001)

N_HIDDEN_1 = 400

N_HIDDEN_2 = 300

N_HIDDEN_3= 100

N_INPUT = 784

N_OUTPUT = 10

KEEP_PROB = 0.35

def trainSimpleDecoder(data,n_input, n_hidden,keep_prob = 1.0):

    

    print("trainSimpleDecoder input %d, layer %d" %(n_input, n_hidden)) 

    x = tf.placeholder("float",[None,n_input])

    keep_prob_op = tf.placeholder(tf.float32)

    encoder_h1 =tf.Variable(tf.random_normal([n_input, n_hidden]))



    encoder_b1 = tf.Variable(tf.random_normal([n_hidden]))

    decoder_b1 = tf.Variable(tf.random_normal([n_input]))



    encoder_op = tf.nn.dropout(tf.nn.sigmoid(tf.add(tf.matmul(x, encoder_h1),encoder_b1)),keep_prob_op)

    y_pred = tf.nn.sigmoid(tf.add(tf.matmul(encoder_op,  tf.transpose(encoder_h1)),decoder_b1))

    y_true = x

    

    cost_J = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

    cost = cost_J

    

    global_step = tf.Variable(0, trainable=False)    

    learning_rate_op = tf.train.exponential_decay(starter_learning_rate, global_step,

                                           500, 0.96, staircase=True)

    

    optimizer = tf.train.RMSPropOptimizer(learning_rate_op).minimize(cost,global_step=global_step)

    

    init = tf.global_variables_initializer()      

   

    config = tf.ConfigProto()

    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:

        sess.run(init)

        total_batch = int(data.shape[0]/batch_size)

        for epoch in range(trainning_epochs):

            for i in range(total_batch):

                batch_xs= data[i*batch_size:(i+1)*batch_size]

                _,c,learning_rate = sess.run([optimizer, cost,learning_rate_op], feed_dict={x: batch_xs,keep_prob_op: keep_prob})



            if epoch % display_step == 0:

                print("Epoch:", '%04d' % (epoch+1),

                      "cost=", "{:.9f}".format(c), "learingrate=","{:.9f}".format(learning_rate))

        print("Optimization Finished!","cost=", "{:.9f}".format(c), "learingrate=","{:.9f}".format(learning_rate))  

        encoder_op_result = np.empty([0,n_hidden])

        for i in range(total_batch):

                batch_xs= data[i*batch_size:(i+1)*batch_size]

                encoder_op_result = np.concatenate([encoder_op_result ,sess.run([encoder_op], feed_dict={x: batch_xs,keep_prob_op: 1.0})[0]])



        encoder_h1_result,encoder_b1 = sess.run([encoder_h1,encoder_b1], feed_dict={})

        sess.close()

        return encoder_op_result, encoder_h1_result,encoder_b1



encoder_op_result, weights1_pretrain,biases1_pretrain = trainSimpleDecoder(train_dataX, N_INPUT, N_HIDDEN_1, KEEP_PROB)

encoder_op_result, weights2_pretrain,biases2_pretrain = trainSimpleDecoder(encoder_op_result, N_HIDDEN_1, N_HIDDEN_2,0.4)

encoder_op_result, weights3_pretrain,biases3_pretrain = trainSimpleDecoder(encoder_op_result, N_HIDDEN_2, N_HIDDEN_3,0.5)

encoder_op_result, weights4_pretrain,biases4_pretrain = trainSimpleDecoder(encoder_op_result, N_HIDDEN_3, N_OUTPUT,0.7)
def drawNeuronLayer(neuron_weight, p = plt):

    amountNeorons = neuron_weight.shape[0]

    neoronsOnSide = np.ceil(np.sqrt(neuron_weight.shape[0])).astype(int)

    imagesize = np.floor(np.sqrt(neuron_weight.shape[1])).astype(int)

    neuron_image = np.zeros([imagesize*neoronsOnSide,imagesize*neoronsOnSide])

    for i in range(0,neoronsOnSide):

        if(i*neoronsOnSide>=amountNeorons):

            break;

        for j in range(0,neoronsOnSide):

            if(i*neoronsOnSide + j>=amountNeorons):

                break;

            one_neoron = np.reshape(neuron_weight[i*neoronsOnSide +j][0:imagesize*imagesize],(imagesize,imagesize)) 

            for k in range(0,imagesize):

                for h in range(0,imagesize):

                    neuron_image[i*imagesize +k][j*imagesize+h] = one_neoron[k,h]

    p.imshow(neuron_image)

plt.set_cmap("plasma")

plt.figure(figsize = (10,10))   

drawNeuronLayer(weights1_pretrain.T)
starter_learning_rate = 0.01

trainning_epochs=15

batch_size = 256

display_step = 1

KEEP_PROB = 0.5



weights_placeholder = {

    "weights_1" : tf.placeholder_with_default(weights1_pretrain,[N_INPUT,N_HIDDEN_1]),

    "weights_2" : tf.placeholder_with_default(weights2_pretrain,[N_HIDDEN_1,N_HIDDEN_2]),

    "weights_3" : tf.placeholder_with_default(weights3_pretrain,[N_HIDDEN_2,N_HIDDEN_3]),  

    "weights_4" : tf.placeholder_with_default(weights4_pretrain,[N_HIDDEN_3,N_OUTPUT])  

}

biases_placeholder = {

    "en_biases_1" : tf.placeholder_with_default(biases1_pretrain,[N_HIDDEN_1]),

    "en_biases_2" : tf.placeholder_with_default(biases2_pretrain,[N_HIDDEN_2]),

    "en_biases_3" : tf.placeholder_with_default(biases3_pretrain,[N_HIDDEN_3]),

    "en_biases_4" : tf.placeholder_with_default(biases4_pretrain,[N_OUTPUT])

}

weights = {

}



biases ={

    "de_biases_1": tf.Variable(tf.random_normal([N_HIDDEN_3])),

    "de_biases_2": tf.Variable(tf.random_normal([N_HIDDEN_2])),

    "de_biases_3": tf.Variable(tf.random_normal([N_HIDDEN_1])),

    "de_biases_4": tf.Variable(tf.random_normal([N_INPUT]))

}



for item, index in enumerate(weights_placeholder):

    weights[index] = tf.Variable(weights_placeholder[index])

for item, index in enumerate(biases_placeholder):

    biases[index] = tf.Variable(biases_placeholder[index])



X = tf.placeholder("float",[None,N_INPUT])

keep_prob = tf.placeholder(tf.float32)



def encode(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_1']),biases['en_biases_1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_2']),biases['en_biases_2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['weights_3']),biases['en_biases_3']))

    encode = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['weights_4']),biases['en_biases_4']))

    return encode

def decode(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, tf.transpose(weights['weights_4'])),biases['de_biases_1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, tf.transpose(weights['weights_3'])),biases['de_biases_2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, tf.transpose(weights['weights_2'])),biases['de_biases_3']))

    decode = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, tf.transpose(weights['weights_1'])),biases['de_biases_4']))

    return decode





encode_op = encode(tf.nn.dropout(X,keep_prob))

decode_op = decode(encode_op)

      

y_true = X              

y_pred = decode_op

cost_J = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

cost = cost_J



global_step = tf.Variable(0, trainable=False)    

learning_rate_op = tf.train.exponential_decay(starter_learning_rate, global_step,

                                       500, 0.96, staircase=True)



optimizer = tf.train.RMSPropOptimizer(learning_rate_op).minimize(cost,global_step=global_step)

    

init = tf.global_variables_initializer()      



config = tf.ConfigProto()

config.gpu_options.allow_growth=True

with tf.Session(config=config) as sess:

    sess.run(init)

    total_batch = int(train_dataX.shape[0]/batch_size)

    for epoch in range(trainning_epochs):

        for i in range(total_batch):

            batch_xs= train_dataX[i*batch_size:(i+1)*batch_size]

            _,c = sess.run([optimizer, cost], feed_dict={X: batch_xs,keep_prob: KEEP_PROB})

        

        c_cross,learning_rate = sess.run([cost,learning_rate_op], feed_dict={X: cross_test_dataX,keep_prob: 1.0})

                                           

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1),

                  "cost=", "{:.9f}".format(c), "c_cross=", "{:.9f}".format(c_cross),"learingrate=","{:.9f}".format(learning_rate))

    print("Optimization Finished!","cost=", "{:.9f}".format(c),  "c_cross=", "{:.9f}".format(c_cross), "learingrate=","{:.9f}".format(learning_rate))  

    neuron_weights_after_deep_1,neuron_weights_after_deep_2,neuron_weights_after_deep_3,neuron_weights_after_deep_4 = sess.run([weights['weights_1'],weights['weights_2'],weights['weights_3'],weights['weights_4']])

    neuron_biases_after_deep_1,neuron_biases_after_deep_2,neuron_biases_after_deep_3,neuron_biases_after_deep_4 = sess.run([biases['en_biases_1'],biases['en_biases_2'],biases['en_biases_3'],biases['en_biases_4']])

 

    sess.close()
f,a = plt.subplots(1, 2, figsize=(12, 12))

a[0].set_title("Before")

drawNeuronLayer(weights1_pretrain.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_after_deep_1.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(weights2_pretrain.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_after_deep_2.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(weights3_pretrain.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_after_deep_3.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(weights4_pretrain.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_after_deep_4.T,a[1])
starter_learning_rate = 0.01

trainning_epochs=10

batch_size = 512

total_batch = int(train_dataX.shape[0]/batch_size)

display_step = 1

excample_to_show = 10

KEEP_PROB = 0.55

N_HIDDEN_LAYER = 3





weights_placeholder = {

    "weights_1" : tf.placeholder_with_default(neuron_weights_after_deep_1,[N_INPUT,N_HIDDEN_1]),

    "weights_2" : tf.placeholder_with_default(neuron_weights_after_deep_2,[N_HIDDEN_1,N_HIDDEN_2]),

    "weights_3" : tf.placeholder_with_default(neuron_weights_after_deep_3,[N_HIDDEN_2,N_HIDDEN_3]),  

    "weights_4" : tf.placeholder_with_default(neuron_weights_after_deep_4,[N_HIDDEN_3,N_OUTPUT])  

}

biases_placeholder = {

    "biases_1" : tf.placeholder_with_default(neuron_biases_after_deep_1,[N_HIDDEN_1]),

    "biases_2" : tf.placeholder_with_default(neuron_biases_after_deep_2,[N_HIDDEN_2]),

    "biases_3" : tf.placeholder_with_default(neuron_biases_after_deep_3,[N_HIDDEN_3]),

    "biases_4" : tf.placeholder_with_default(neuron_biases_after_deep_4,[N_OUTPUT])

}

weights = {

}



biases ={

}



for item, index in enumerate(weights_placeholder):

    weights[index] = tf.Variable(weights_placeholder[index])

for item, index in enumerate(biases_placeholder):

    biases[index] = tf.Variable(biases_placeholder[index])

    

X = tf.placeholder("float",[None,N_INPUT])

y_true = tf.placeholder("float",[None,N_OUTPUT])

keep_prob = tf.placeholder(tf.float32)



def classify(x):

    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['weights_1']),biases['biases_1']))

    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['weights_2']),biases['biases_2']))

    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['weights_3']),biases['biases_3']))

    layer_4 = tf.add(tf.matmul(layer_3, weights['weights_4']),biases['biases_4'])

    y_hat = tf.nn.sigmoid(layer_4);

    return y_hat,layer_4



yhat_op,yhat_op_sigmoidless = classify(tf.nn.dropout(X,keep_prob))



cost_J = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(yhat_op_sigmoidless, y_true))

cost_reg = tf.mul(LAMBDA , tf.add_n([tf.nn.l2_loss(weights['weights_1']),\

                                     tf.nn.l2_loss(weights['weights_2']),\

                                     tf.nn.l2_loss(weights['weights_4']),\

                                     tf.nn.l2_loss(weights['weights_3'])]))

ypredict = tf.argmax(yhat_op,1)



cost =cost_J

    





global_step = tf.Variable(0, trainable=False)

learning_rate_op = tf.train.exponential_decay(starter_learning_rate, global_step,

                                           500, 0.7, staircase=True)



optimizer = tf.train.RMSPropOptimizer(learning_rate_op).minimize(cost,global_step=global_step)





init = tf.global_variables_initializer()

    

config = tf.ConfigProto()

config.gpu_options.allow_growth=True

J = []

crossJ = []

with tf.Session(config=config) as sess:

   

    sess.run(init)   

    for epoch in range(trainning_epochs):

        for i in range(total_batch):

            batch_xs= train_dataX[i*batch_size:(i+1)*batch_size]

            batch_ys= train_dataY[i*batch_size:(i+1)*batch_size]    

            _,c = sess.run([optimizer, cost], feed_dict={X: batch_xs, y_true:batch_ys,keep_prob: KEEP_PROB})

            J.append(c)

           

        c_cross,learning_rate = sess.run([cost,learning_rate_op], feed_dict={X: cross_test_dataX,  y_true:cross_test_dataY,keep_prob: 1.0})

        crossJ.append(c_cross)

        if epoch % display_step == 0:

            print("Epoch:", '%04d' % (epoch+1),

                  "cost=", "{:.9f}".format(c),  "c_cross=", "{:.9f}".format(c_cross),"learingrate=","{:.9f}".format(learning_rate))

        





    print("Optimization Finished!") 

    neuron_weights_1,neuron_weights_2,neuron_weights_3,neuron_weights_4 = sess.run([weights['weights_1'],weights['weights_2'],weights['weights_3'],weights['weights_4']])

    ypredict_test = sess.run(ypredict, feed_dict={X: cross_test_dataX,keep_prob: 1.0})

    ypredict_final = sess.run(ypredict, feed_dict={X: test_data,keep_prob: 1.0}) 

 
f,a = plt.subplots(1, 10, figsize=(10, 1))



for i in range(excample_to_show): 

    for j in range(0,1):

        a[i].axis('off')

    a[i].imshow(np.reshape(test_data[i],(28,28)))

    a[i].set_title(ypredict_final[i])

plt.draw()

 
from sklearn import metrics

cross_test_classes = cross_test_classes.ravel()

accuracy = np.sum(cross_test_classes == ypredict_test)/cross_test_classes.shape[0]

print ("validation accuracy:", accuracy)

print ("Precision", metrics.precision_score(cross_test_classes, ypredict_test,average='macro'))

print ("Recall", metrics.recall_score(cross_test_classes, ypredict_test,average='macro'))

print ("f1_score", metrics.f1_score(cross_test_classes, ypredict_test,average='macro'))

print ("confusion_matrix")

print (metrics.confusion_matrix(cross_test_classes, ypredict_test))
f,a = plt.subplots(1, 2, figsize=(12, 12))

a[0].set_title("Before")

drawNeuronLayer(neuron_weights_after_deep_1.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_1.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(neuron_weights_after_deep_2.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_2.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(neuron_weights_after_deep_3.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_3.T,a[1])



f,a = plt.subplots(1, 2, figsize=(10, 10))

a[0].set_title("Before")

drawNeuronLayer(neuron_weights_after_deep_4.T,a[0])

a[1].set_title("After")

drawNeuronLayer(neuron_weights_4.T,a[1])
submission = pd.DataFrame({

        "ImageId": range(1,len(ypredict_final)+1),

        "Label": ypredict_final

    })



submission.to_csv("predict_numers.csv", index=False)