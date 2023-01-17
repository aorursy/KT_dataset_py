import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import tensorflow as tf

from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
# prefix = 'd:\\AI_datasets\\mnist\\'

prefix = '../input/'

train_data = pd.read_csv(prefix + 'train.csv')

test_data = pd.read_csv(prefix + 'test.csv')
x_train_data = train_data.loc[:, 'pixel0':].values

y_train_data = train_data.loc[:, 'label'].values.reshape(-1,1)

y_train_data = OneHotEncoder(categories='auto').fit_transform(y_train_data).toarray()

print('x_train_shape:',x_train_data.shape)

print('y_train_shape:',y_train_data.shape)



x_train, x_test, y_train, y_test = train_test_split(x_train_data, y_train_data, test_size=0.01)
# tensorboard 参数总结显示

def var_summary(var):

        mean = tf.reduce_mean(var)

        stddev = tf.reduce_mean(tf.square(var - mean))

        tf.summary.scalar('mean',mean) # 平均值

        tf.summary.scalar('stddev',stddev) #标准差

        tf.summary.scalar('max',tf.reduce_max(var)) # 最大值

        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值

        # tf.summary.histogram('histogram',var) # 直方图
# 宏观参数

n_input = 784

n_output = 10



n_conv1 = (3, 64)

n_conv2 = (3, 128)



n_fc1 = 1024



keep_prob = 0.6

learning_rate = 0.01

tf.reset_default_graph()



with tf.name_scope('parameters_of_C1'):

    # 标准差设置很重要，默认标准差为1，测试中生成的最大权重可能为2，经多层网络传播后，最后的输出层有1e19之大，直接导致loss值为nan！！！

    wc1 = tf.Variable(tf.truncated_normal([n_conv1[0], n_conv1[0], 1, n_conv1[1]],stddev=0.1), name='weights')

    bc1 = tf.Variable(tf.truncated_normal([n_conv1[1]]), name='biases')

    

    var_summary(wc1)

    var_summary(bc1)



    

with tf.name_scope('parameters_of_C2'):

    wc2 = tf.Variable(tf.truncated_normal([n_conv2[0], n_conv2[0], n_conv1[1], n_conv2[1]],stddev=0.01), name='weights')

    bc2 = tf.Variable(tf.truncated_normal([n_conv2[1]]), name='biases')

        

    var_summary(wc2)

    var_summary(bc2)



with tf.name_scope('parameters_of_FC1'):

    # 7 * 7 经过两个pooling层后图像由28*28->7*7

    # 128 上一层有128个特征图，需要转为一维向量

    wfc1 = tf.Variable(tf.truncated_normal([7 * 7 * n_conv2[1], n_fc1],stddev=0.01), name='weights')

    bfc1 = tf.Variable(tf.truncated_normal([n_fc1]), name='baises')

    

        

    var_summary(wfc1)

    var_summary(bfc1)

    

    

with tf.name_scope('parameters_of_FC2'):

    wfc2 = tf.Variable(tf.truncated_normal([n_fc1, n_output],stddev=0.01), name='weights')

    bfc2 = tf.Variable(tf.truncated_normal([n_output]), name='baises')

    var_summary(wfc2)



weights = {

    'wc1' : wc1,

    'wc2' : wc2,

    'wfc1': wfc1,

    'wfc2' : wfc2

}



baises = {

    'bc1' : bc1,

    'bc2' : bc2,

    'bfc1': bfc1,

    'bfc2': bfc2

}
def forward_propagation(_input,w ,b, keepratio):

    # preprocess intput to tf format

    #[n, h, w, c]

    input_r = tf.reshape(_input, shape=[-1, 28, 28, 1])

    

    #CONV LAYER 1

    with tf.name_scope('CONVOLUTION_LAYER_1'):

        conv1 = tf.nn.conv2d(input_r, w['wc1'], [1, 1, 1, 1], padding='SAME',name = 'conv1_convolution')

        # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=True,data_format='NHWC',dilations=[1, 1, 1, 1],name=None

        

        conv1 = tf.nn.relu(tf.add(conv1,b['bc1'], name='conv1_bais_add'), name='conv1_activation')

        

    # POOL LAYER 1

    with tf.name_scope('POOL_LAYER_1'):

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool1_pooling')

        pool1_dr = tf.nn.dropout(pool1, keepratio, name='pool1_dropout')

        

    #CONV LAYER 2

    with tf.name_scope('CONVOLUTION_LAYER_2'):

        conv2 = tf.nn.conv2d(pool1_dr, w['wc2'], [1, 1, 1, 1], padding='SAME',name = 'conv2_convolution')

        # tf.nn.conv2d(input,filter,strides,padding,use_cudnn_on_gpu=True,data_format='NHWC',dilations=[1, 1, 1, 1],name=None

        

        conv2 = tf.nn.relu(tf.add(conv2,b['bc2'], name='conv2_bais_add'), name='conv2_activation')

        

    # POOL LAYER 2

    with tf.name_scope('POOL_LAYER_2'):

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool2_pooling')

        pool2_dr = tf.nn.dropout(pool2, keepratio, name='pool1_dropout')  

        

        

    #FULLY CONNECTED LAYER 1

    with tf.name_scope('FULLY_CONNECTED_LAYER_1'):

         # change shape of result of pool layer 2 to satisify fully connected layer

        fc1_input = tf.reshape(pool2_dr, [-1, w['wfc1'].get_shape().as_list()[0]], name='fc1_reshape')

        

        fc1_z = tf.add(tf.matmul(fc1_input, w['wfc1']),b['bfc1'], name='fc1_full_connection')

        fc1_a = tf.nn.relu(fc1_z, name='fc1_activation')

        



    #FULLY CONNECTED LAYER 2

    with tf.name_scope('FULLY_CONNECTED_LAYER_2'):

        fc2_z = tf.add(tf.matmul(fc1_a, w['wfc2']),b['bfc2'], name='fc2_full_connection')

        

    out = {

        'conv1': conv1,

        'pool1': pool1,

        'conv2': conv2,

        'fc1_input': fc1_input,

        'fc1_a': fc1_a,

        'fc2_z': fc2_z

    }    

    return out
# INPUT

with tf.name_scope('INPUT'):

    x = tf.placeholder(tf.float32, [None, n_input], name='input_x')

    y = tf.placeholder(tf.float32, [None, n_output], name='input_y')



# FORWARD_PROPAGATION

with tf.name_scope('FORWARD_PROPAGATION'):

    pred = forward_propagation(x, weights, baises, keep_prob)['fc2_z']

    prediction = tf.argmax(pred, 1, output_type=tf.int32)



out = forward_propagation(x, weights, baises, keep_prob)    

# LOSS FUNCTION

with tf.name_scope('LOSS_FUNCTION'):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred), name='average_loss')

    tf.summary.scalar('loss', loss)

# OPTIMIZE

with tf.name_scope('OPTIMIZER'):

    optim = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)



# ACCURACY

with tf.name_scope('ACCURACY'):

    same_matrix = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))

    accuracy = tf.reduce_mean(tf.cast(same_matrix, tf.float32))

    sum_acc = tf.summary.scalar('accuracy', accuracy)





init = tf.global_variables_initializer()

merged = tf.summary.merge_all()

saver =tf.train.Saver()
n_epoch = 30



print_step = 2



batch_size = 100

n_batch = len(x_train) // batch_size





do_train = 1
%%time

if do_train:

    with tf.Session() as sess:

        sess.run(init)

        writer = tf.summary.FileWriter('.',sess.graph)



        for epoch in range(n_epoch):

            ave_loss         =      0

            ave_train_acc    =      0

            ave_test_acc     =      0

            for batch in range(n_batch):

                start = batch * batch_size

                batch_xs, batch_ys = x_train[start: start + batch_size], y_train[start: start + batch_size]

                summary, _, p_acc, p_loss = sess.run([merged, optim,accuracy,loss],feed_dict={x:batch_xs, y:batch_ys})

                

                ave_train_acc += (p_acc / n_batch)

                ave_loss += (p_loss / n_batch)

                

                if batch % (n_batch // 10) == 0:

                    print('#', end='')

            print('\n',end='')

            

            writer.add_summary(summary, epoch)

            if epoch % print_step == 0:

                ave_test_acc = sess.run(accuracy, feed_dict={x:x_test, y:y_test})

                print("Epoch: %d/%d, Loss: %.4f, train accuracy: %.4f, test accuracy: %.4f" % 

                      (epoch, n_epoch, ave_loss, ave_train_acc, ave_test_acc))

        saver.save(sess, 'cnn_mnist.ckpt')

        do_train = 0
if not do_train:

    n_batch = len(test_data) // batch_size

    res = np.array([])

    with tf.Session() as sess:

        sess.run(init)

        saver.restore(sess, 'cnn_mnist.ckpt')

        for batch in range(n_batch):

            start = batch * batch_size

            batch_xs = test_data[start: start + batch_size]

            res = np.hstack((res, sess.run(prediction,feed_dict={x:batch_xs.values})) ).astype(int) 

        sub = pd.DataFrame({

        'ImageId': list(range(1,len(test_data) + 1)),

        'Label':res

                 })

        sub.to_csv('submission.csv',index=0)
# init = tf.global_variables_initializer()

# with tf.Session() as sess:

#     sess.run(init)

#     print(sess.run(tf.reduce_max(wc1)))

#     print(sess.run(tf.reduce_min(wc1)))

#     print(sess.run(tf.reduce_max(wc2)))

#     print(sess.run(tf.reduce_min(wc2)))

#     loss = (sess.run((accuracy), feed_dict={x:x_train[:1000], y: y_train[:1000]}))
# plt.figure(dpi=300)

# num_total = 50

# for i in range(num_total):

#     plt.subplot(num_total // 5 ,5,i+1)

#     plt.axis('off')

#     plt.imshow( res['conv2'][2,:,:,i], cmap='bone')
sub = pd.read_csv('submission.csv')

sub.head(10)
sub.to_csv('submission.csv',index=0)