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
prediction = tf.matmul(x,W)+b
#******此部份因重複softmax，故將原本的prediction = tf.nn.softmax(tf.matmul(x,W)+b),
#改為prediction = tf.matmul(x,W)+b。******

#使用cross_entropy
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#使用各種優化器
#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss) #Accuracy 0.9249
#train_step = tf.train.AdadeltaOptimizer(0.2).minimize(loss) #Accuracy 0.9134
#train_step = tf.train.AdadeltaOptimizer(0.2).minimize(loss) #Accuracy 0.914
#train_step = tf.train.AdagradOptimizer(0.2).minimize(loss) #Accuracy 0.9258
#train_step = tf.train.AdagradDAOptimizer(0.2).minimize(loss)
train_step = tf.train.MomentumOptimizer(0.2,0.2).minimize(loss) #Accuracy 0.9267
#train_step = tf.train.AdamOptimizer(0.2).minimize(loss) #Accuracy 0.8804
#train_step = tf.train.FtrlOptimizer(0.2).minimize(loss) #Accuracy 0.9247
#train_step = tf.train.ProximalGradientDescentOptimizer(0.2).minimize(loss) #Accuracy 0.9253
#train_step = tf.train.ProximalAdagradOptimizer(0.2).minimize(loss) #Accuracy 0.9262
#train_step = tf.train.RMSPropOptimizer(0.2).minimize(loss) #Accuracy 0.8844

#******以上是我把影片內所列出的優化器個執行一次，每次執行迭代21次，並把學習率設為0.2，其中 AdagradDAOptimizer缺少一個變數故無法執行
#，剩餘優化器執行後，所得出 MomentumOptimizer 跑出的準確率最高。******


#初始化變量
init = tf.global_variables_initializer()

#結果存放在一個布林型列表中
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#返回一維張量中最大的值所在的位置
#求準確率
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter "+ str(epoch) + ",Testing Accuracy "+ str(acc))