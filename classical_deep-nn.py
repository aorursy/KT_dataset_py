import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
data_dir = r'../input/train.csv'
# mnist = input_data.read_data_sets(data_dir, one_hot=True)
train_set = pd.read_csv(data_dir)
test_set = pd.read_csv('../input/test.csv')
print(train_set.head())
print(test_set.head())
y_train = pd.get_dummies((train_set['label']))
x_train = pd.get_dummies(train_set.drop(labels=['label'], axis=1))
# y_test = pd.get_dummies((test_set['label']))
test = pd.get_dummies(test_set)

print(x_train.head())
print(y_train.head()) 

in_x = 784
w1_num = 512
w2_num = 216
out = 10
learn_rate = 0.001
alpha = 0.001
pred = tf.placeholder(tf.float32, [None, 10])
x = tf.placeholder(tf.float32, [None, in_x])
# help(tf.initializers.random_normal)
weight = {
    'w1': tf.Variable(tf.random_normal([in_x, w1_num], stddev=tf.sqrt(1/w1_num))),
    'w2': tf.Variable(tf.random_normal([w1_num, w2_num], stddev=tf.sqrt(1/w2_num))),
    'out': tf.Variable(tf.random_normal([w2_num, out], stddev=tf.sqrt(1/out)))
}
bias = {
    'b1': tf.Variable(tf.random_normal([w1_num], stddev=tf.sqrt(1/w1_num))),
    'b2': tf.Variable(tf.random_normal([w2_num], stddev=tf.sqrt(1/w2_num))),
    'out': tf.Variable(tf.random_normal([out], stddev=tf.sqrt(1/out)))
}
o_1 = tf.nn.relu(tf.matmul(x, weight['w1']) + bias['b1'])
o_2 = tf.nn.relu(tf.matmul(o_1, weight['w2']) + bias['b2'])
y = tf.matmul(o_2, weight['out']) + bias['out']
# tf.summary.histogram('weight_out', weight['out'])
# tf.summary.histogram('bias_out', bias['out'])
l2 = alpha * tf.square(tf.reduce_sum(weight['w1'])+tf.reduce_sum(weight['w2'])+tf.reduce_sum(weight['out']))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=pred, logits=y) + l2)
# tf.summary.scalar('loss', loss)
train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)
init = tf.global_variables_initializer()
epoch = 100
batch_size = 128
enx = OneHotEncoder()
eny = OneHotEncoder()
eny.fit_transform(train_set[['label']])
with tf.Session() as sess:
    sess.run(init)
#     merged = tf.summary.merge_all()
#     writer = tf.summary.FileWriter(r'./logs/lastdemo', sess.graph)
    for i in range(epoch):
        batch_num = int(x_train.shape[0] / batch_size)
        for j in range(batch_num):
            batch = train_set.sample(n=batch_size)
            batch_ys = eny.transform(batch[['label']]).toarray()
            batch_xs = np.array(batch.drop(labels='label', axis=1))
            _, a = sess.run([train_step, loss], feed_dict={x: batch_xs, pred: batch_ys})
            if j % 50 == 0:
                print('训练第{0}次, loss:={1}'.format(j,a))
#                 writer.add_summary(merged_summary, i*batch_num+j)
        if (i) % 1 == 0:
            train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))
#             test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1)), tf.float32))
            print('{0}次\nloss:{1}\ntrain_acc:{2}\n:'.format(i+1, a, sess.run(train_accuracy, feed_dict={x: x_train, pred: y_train}))), 
#     writer.close()
                
    # predict:
    mytest_batch = train_set.sample(n=500)
    mytest_x, mytest_y= sess.run([x,y], feed_dict={x: np.array(mytest_batch.drop(labels='label', axis=1))})
    mytest_y = np.argmax(mytest_y, 1)
#     print(mytest_y)
    mytest_pred = np.array(mytest_batch['label'])
#     print(mytest_pred)

    print(mytest_x.shape)
    wrong_index = np.where(mytest_y != mytest_pred)[0]
    print(wrong_index)
    fig = plt.figure(figsize=(84/4, len(wrong_index)/4))
    for i in enumerate(wrong_index):
        ax = fig.add_subplot(3, 15, i[0]+1)
        ax.imshow(mytest_x[i[1]].reshape((28, 28)))
        ax.set_title('pred %d\nlabel%d' % (mytest_y[i[1]], mytest_pred[i[1]]))
    plt.show()
    
    
    
    pred_test = sess.run(y, feed_dict={x: test_set})
    pred_test = tf.argmax(pred_test, 1).eval()
    res = pd.Series(pred_test, name="Label")
    imgid = pd.Series(range(1, 28001), name="ImageId")
    submission = pd.concat([imgid, res], axis=1)
    print(submission[:10])
    submission.to_csv("mnist_sub.csv",index=False)
    
            

