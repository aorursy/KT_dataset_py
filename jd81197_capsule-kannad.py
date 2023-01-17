# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from __future__ import division, print_function, unicode_literals

import numpy as np # linear algebra

import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow.compat.v1 as tf

from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split

tf.disable_v2_behavior()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
tf.reset_default_graph()
train_data=pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

test_data=pd.read_csv('/kaggle/input/Kannada-MNIST/test.csv')

sample_sub=pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')
train_data.shape
train_data.head(1)
samples=5

plt.figure(figsize=(samples*6,700))

for index in range(samples):

    plt.subplot(1, samples, index+1)

    image=np.array(train_data.iloc[index, 1:]).reshape(28,28)

    plt.imshow(image, cmap="binary")

    plt.axis("off")

plt.show()
train_data.label[:samples]
X_train=train_data.drop('label', axis=1)

Y_train=train_data.label

test_data=test_data.drop('id', axis=1)
X_train=X_train/255

test_data=test_data/255
X_train=X_train.astype('float32')

test_data=test_data.astype('float32')
X_train=X_train.values.reshape(-1,28,28,1)

test_data=test_data.values.reshape(-1,28,28,1)
X_train,X_test,y_train,y_test=train_test_split(X_train,Y_train,random_state=42,test_size=0.15)
abc=np.zeros(5000, dtype='int64')

abc.shape
batch_size=50

train_dataset_data = tf.data.Dataset.from_tensor_slices(X_train)

train_dataset_labels = tf.data.Dataset.from_tensor_slices(y_train)

train_dataset = tf.data.Dataset.zip((train_dataset_data, train_dataset_labels)).shuffle(500).repeat().batch(batch_size)
valid_dataset_data = tf.data.Dataset.from_tensor_slices(X_test)

valid_dataset_labels= tf.data.Dataset.from_tensor_slices(y_test)

valid_dataset = tf.data.Dataset.zip((valid_dataset_data, valid_dataset_labels)).shuffle(500).repeat().batch(batch_size)
test_dataset_data=tf.data.Dataset.from_tensor_slices(test_data)

test_data_labels=tf.data.Dataset.from_tensor_slices(np.zeros(5000, dtype='int64'))

test_dataset=tf.data.Dataset.zip((test_dataset_data, test_data_labels)).batch(50)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,

                                               train_dataset.output_shapes)

next_element = iterator.get_next()
training_init_op = iterator.make_initializer(train_dataset)

init_op = tf.global_variables_initializer()

validation_init_op = iterator.make_initializer(valid_dataset)

test_init_op=iterator.make_initializer(test_dataset)
with tf.Session() as sess:

    sess.run(init_op)

    sess.run(training_init_op)

    for i in range(1):

        data=sess.run(next_element)

        print(data[1])

        temp=tf.reshape(data[0], [-1, 28*28], name="temp")

        print(temp)
#X=tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
#X.dtype
conv_layer1_params={

    "filters" : 256,

    "kernel_size" : 9,

    "strides" : 1,

    "padding" : "valid",

    "activation" : tf.nn.relu,

}
caps1_maps=32

caps1_caps=caps=caps1_maps*6*6

caps1_dims=8
conv_layer2_params={

    "filters" : caps1_maps*caps1_dims,

    "kernel_size" : 9,

    "strides" : 2,

    "padding" : "valid",

    "activation" : tf.nn.relu

}
conv1=tf.layers.conv2d(next_element[0], name="conv1", **conv_layer1_params)

conv2=tf.layers.conv2d(conv1, name="conv2", **conv_layer2_params)
caps1_reshape=tf.reshape(conv2, [-1, caps1_caps, caps1_dims], name="caps1_reshape")
def squash(s, axis=-1, epsilon=1e-7, name=None):

    with tf.name_scope(name, default_name="squash"):

        squared_s=tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)

        mod_s=tf.sqrt(squared_s+epsilon)

        squash_factor=squared_s/(1.+squared_s)

        unit_vector=s/mod_s

        return squash_factor*unit_vector
caps1_output=squash(caps1_reshape, name="caps1_output")
caps1_output
caps2_caps=10

caps2_dims=16
sigma=0.1

W_init=tf.random_normal(

        shape=(1, caps1_caps, caps2_caps, caps2_dims, caps1_dims ),

        stddev=sigma, dtype=tf.float32, name="W_init"

)
W=tf.Variable(W_init, name="W")
W_tiled=tf.tile(W, [batch_size, 1,1,1,1], name="W_tiled")
caps1_output_extended=tf.expand_dims(caps1_output, -1, name="caps1_output_extended")

caps1_output_tile=tf.expand_dims(caps1_output_extended, 2, name="caps1_output_tile")

caps1_output_tiled=tf.tile(caps1_output_tile, [1,1,caps2_caps,1,1], name="caps1_output_tiled")
W_tiled
caps1_output_tiled
caps2_predicted=tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")


caps2_predicted
caps2_bij=tf.zeros([batch_size, caps1_caps, caps2_caps, 1, 1], dtype=np.float32, name="caps2_bij")
softmax_bij=tf.nn.softmax(caps2_bij, dim=2, name="softmax_bij")
weighted_pred=tf.multiply(softmax_bij, caps2_predicted, name="weighted_pred")
weighted_pred
weighted_sum=tf.reduce_sum(weighted_pred, axis=1, keep_dims=True, name="weighted_sum")
weighted_sum
caps2_output1=squash(weighted_sum, axis=-2, name="caps2_output1")
caps2_output1
caps2_output1_tiled=tf.tile(caps2_output1, [1, caps1_caps, 1,1,1], name="caps2_output1_tiled")
caps2_output1_tiled
agreement_prod=tf.matmul(caps2_predicted, caps2_output1_tiled, transpose_a=True, name="agreement_prod")
agreement_prod
caps2_bij=tf.add(caps2_bij, agreement_prod)
softmax_bij_2=tf.nn.softmax(caps2_bij, dim=2, name="softmax_bij_2")

weighted_pred_2=tf.multiply(softmax_bij_2, caps2_predicted, name="weighted_pred_2")

weighted_sum_2=tf.reduce_sum(weighted_pred_2, axis=1, keep_dims=True, name="weighted_sum_2")

caps2_output2=squash(weighted_sum_2, axis=-2, name="caps2_output2")

caps2_output2_tiled=tf.tile(caps2_output2, [1, caps1_caps, 1,1,1], name="caps2_output2_tiled")

agreement_prod=tf.matmul(caps2_predicted, caps2_output2_tiled, transpose_a=True, name="agreement_prod")

caps2_bij=tf.add(caps2_bij, agreement_prod)
caps2_output_final=caps2_output2
caps2_output_final
def find_class(output_tensor, axis=-1,  epsilon=1e-7, keep_dims=False, name=None):

    with tf.name_scope(name, default_name="find_class"):

        squared_prob=tf.reduce_sum(tf.square(output_tensor), axis=axis, keep_dims=keep_dims)

        return tf.sqrt(squared_prob+epsilon)
final_prob=find_class(caps2_output_final, axis=-2, name="final_prob")
final_prob
max_prob_digit=tf.argmax(final_prob, axis=2, name="final_prob" )

max_prob_digit
y_pred = tf.squeeze(max_prob_digit, axis=[1,2], name="y_pred")
y_pred.shape
#y=tf.placeholder(shape=[None], dtype=tf.int64, name="y")
m_plus=0.9

m_minus=0.1

lambdaa=0.5
T=tf.one_hot(next_element[1], caps2_caps, name="T")
caps2_output_T=find_class(caps2_output_final, axis=-2, keep_dims=True,

                              name="caps2_output_T")
True_class_margin_loss=tf.square(tf.maximum(0., m_plus-caps2_output_T), name="True_class_margin_loss")

True_class_margin_loss
True_class_loss_reshape=tf.reshape(True_class_margin_loss, shape=(-1, 10), name="True_class_loss_reshape")
True_class_loss_reshape
False_class_margin_loss = tf.square(tf.maximum(0., caps2_output_T - m_minus),

                             name="False_class_margin_loss")

False_class_loss_reshape = tf.reshape(False_class_margin_loss, shape=(-1, 10),

                          name="False_class_oss_reshape")
L=tf.add(T*True_class_loss_reshape, lambdaa*(1.0-T)*False_class_loss_reshape, name="L")
L
margin_loss=tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
mask_output_label=tf.placeholder_with_default(True, shape=(), name="mask_output_label")
mask_output_label
reconstruction_targets=tf.cond(mask_output_label, lambda:next_element[1], lambda:y_pred, name="reconstruction_targets")
reconstruction_targets
reconstruction_mask = tf.one_hot(reconstruction_targets,

                                 depth=caps2_caps,

                                 name="reconstruction_mask")
reconstruction_mask_reshape=tf.reshape(reconstruction_mask, [-1, 1, caps2_caps, 1, 1], name="reconstruction_mask_reshape")
caps2_output_mask=tf.multiply(caps2_output_final, reconstruction_mask_reshape, name="caps2_output_mask")
decoder_input=tf.reshape(caps2_output_mask, [-1, caps2_caps*caps2_dims], name="decoder_input")
decoder_input
hidden1_units=512

hidden2_units=1024

output_image=28*28
with tf.name_scope("decoder"):

    hidden1=tf.layers.dense(decoder_input, hidden1_units, activation=tf.nn.relu, name="hidden1")

    hidden2=tf.layers.dense(hidden1, hidden2_units, activation=tf.nn.relu, name="hidden2")

    decoder_output=tf.layers.dense(hidden2, output_image, activation=tf.nn.sigmoid, name="decoder_output")
Input_flat=tf.reshape(next_element[0], [-1, output_image], name="Input_flat")

squared_difference=tf.square(Input_flat-decoder_output, name="squared_difference")

reconstruction_loss=tf.reduce_mean(squared_difference, name="reconstruction_loss")
alpha=0.0005

loss=tf.add(margin_loss, alpha*reconstruction_loss, name="loss")
correct = tf.equal(next_element[1], y_pred, name="correct")

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
optimizer = tf.train.AdamOptimizer()

training_op = optimizer.minimize(loss, name="training_op")
init = tf.global_variables_initializer()

saver = tf.train.Saver()
epochs=10

restore_checkpoint=True

iterations_per_epoch_train=X_train.shape[0] // batch_size

iterations_per_epoch_test=X_test.shape[0] // batch_size

best_loss_value=np.infty

checkpoint_path= "../input/Kannada-MNIST/capsule_params"



with tf.Session() as sess:

    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):

        saver.restore(sess, checkpoint_path)

    else:

        init.run()

        

    sess.run(init_op)

    for epoch in range(epochs):

        

        sess.run(training_init_op)

        for iteration in range(1, iterations_per_epoch_train + 1):

            

            _,loss_train=sess.run([training_op, loss])

            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.5f}".format(

                      iteration, iterations_per_epoch_train,

                      iteration * 100 / iterations_per_epoch_train,

                      loss_train),

                  end="")

            

        

        

        sess.run(validation_init_op)

        loss_vals=[]

        acc_vals=[]

        for iteration in range(1, iterations_per_epoch_test + 1):

            

            loss_val,acc_val=sess.run([loss, accuracy], feed_dict={mask_output_label : False})

            loss_vals.append(loss_val)

            acc_vals.append(acc_val)

            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(

                      iteration, iterations_per_epoch_test,

                      iteration * 100 / iterations_per_epoch_test),

                  end=" " * 10)

                

        Mean_loss=np.mean(loss_vals)

        Mean_accuracy=np.mean(acc_vals)

        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(

            epoch + 1, Mean_accuracy * 100, Mean_loss,

            " (improved)" if Mean_loss < best_loss_value else ""))

        if Mean_loss < best_loss_value:

            #save_path=saver.save(sess, checkpoint_path)

            best_loss_value=Mean_loss

        
pred=[]

with tf.Session() as sess:

    sess.run(init)

    sess.run(init_op)

    sess.run(test_init_op)

    for i in range(100):

        caps2_output_value, y_pred_value=sess.run([caps2_output_final, y_pred])

        sample_sub['label'][50*i : 50*i + 50]= y_pred_value

    
y_pred_value
sample_sub.to_csv('submission.csv',index=False)