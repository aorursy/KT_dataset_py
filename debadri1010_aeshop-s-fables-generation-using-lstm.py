import numpy as np
import collections
import tensorflow as tf

n_inputs = 3
n_hidden = 512
learning_rate = 0.00001
words = 'long ago , the mice had a general council to consider what measures they could take to outwit their common enemy , the cat . some said this , and some said that but at last a young mouse got up and said he had a proposal to make , which he thought would meet the case . you will all agree , said he , that our chief danger consists in the sly and treacherous manner in which the enemy approaches us . now , if we could receive some signal of her approach , we could easily escape from her . i venture , therefore , to propose that a small bell be procured , and attached by a ribbon round the neck of the cat . by this means we should always know when she was about , and could easily retire while she was in the neighbourhood . this proposal met with general applause , until an old mouse got up and said that is all very well , but who is to bell the cat ? the mice looked at one another and nobody spoke . then the old mouse said it is easy to propose impossible remedies .'.strip().split(' ')
count = collections.Counter(words).most_common()
dic = dict()
for word,_ in count:
    dic[word] = len(dic)

rev_dic = dict(zip(dic.values(),dic.keys()))

vocab_size = len(dic)
print(dic)
def RNN(x,weights,biases):
    x = tf.reshape(x,[-1,n_inputs])
    x = tf.split(x,n_inputs,1)
    #print('x = ',x)
    cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden,reuse=True)
    outputs,states = tf.nn.static_rnn(cell1,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, vocab_size]))
}
biases = {
    'out': tf.Variable(tf.random_normal([vocab_size]))
}
x = tf.placeholder(shape=(n_inputs,),dtype=tf.float32)
pred = RNN(x,weights,biases)
y = tf.placeholder(shape=(vocab_size,),dtype=tf.float32)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
loss = tf.reshape(loss,[1,1])
with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
import matplotlib.pyplot as plt
with tf.Session() as ss:
    ss.run(init)
    offset = 0
    losses = []
    for i in range(10000):
        
        three_words = np.array([dic[words[(offset+j)%len(words)]] for j in range(offset,offset+n_inputs)],dtype= np.float32)
        #print('Three = ',three_words)
        one_hot = np.zeros([vocab_size],dtype = np.float32)
        one_hot[dic[words[(offset+n_inputs)%len(words)]]] = 1.0
        _,cost, onehot_pred = ss.run([optimizer,loss, pred], feed_dict={x: three_words, y: one_hot })
        if(i%100==0):
            print('Epoch: ',i)
            losses.append(cost[0,0])
        offset = (offset+1)%len(words)
plt.plot(losses)
plt.show()
print('My story=======')
print('a general council',end=' ')
#a = np.array([dic[words[(0+j)%len(words)]] for j in range(0,0+n_inputs)],dtype= np.float32)
a = np.array([dic['a'],dic['general'],dic['council']],dtype=np.float32)
#print("shape = ",a.shape)
index = 3
ss1 = tf.Session()
ss1.run(init)
for k in range(len(words)-3):
    pred1 = RNN(a,weights,biases)
    pred1 = tf.nn.softmax(pred1)
    next_word = tf.argmax(pred1,1)
    next_ind = ss1.run(next_word)
    print(rev_dic[next_ind[0]],end=' ')
    a[0] = a[1]
    a[1] = a[2]
    a[2] = dic[words[index]]
    index+=1
