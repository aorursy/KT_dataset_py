import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#載入數據集
mnist = input_data.read_data_sets('../input',one_hot = True)

#每個批次的大小
batch_size = 100
#計算一共有多少個批次
n_batch = mnist.train.num_examples // batch_size

#定義兩個placeholder
x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

#創建一個簡單的神經網路
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.add(tf.matmul(x,W),b))

#二次代價函數
loss = tf.reduce_mean(tf.square(prediction-y))
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用梯度下降法
#train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)#0.8371

#使用各種優化器
#train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)#0.8833
#train_step = tf.train.MomentumOptimizer(0.01,momentum=0.9).minimize(loss)#0.9062
#train_step = tf.train.AdamOptimizer(0.01).minimize(loss)#0.924
#train_step = tf.train.FtrlOptimizer(0.01).minimize(loss)#0.8834
train_step = tf.train.RMSPropOptimizer(0.01).minimize(loss)#0.9282<<=
#train_step = tf.train.ProximalGradientDescentOptimizer(0.01).minimize(loss)#0.8371
#train_step = tf.train.ProximalAdagradOptimizer(0.01).minimize(loss)#0.8828
#train_step = tf.train.AdadeltaOptimizer(0.01).minimize(loss)#0.787

#結果存放在一個布林型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#返回一維張量中最大的值所在的位置
#求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#初始化變量
init = tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+ str(epoch) + ",Testing Accuracy= "+ str(acc))