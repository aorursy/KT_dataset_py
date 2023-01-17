import tensorflow as tf

tf.disable_eager_execution()



import numpy as np
path_to_file = '../input/shakespeare/Shakespeare'

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')

# text = "hey i am here"

vocab = sorted(set(text))

vocabLen = len(vocab)
def oneHot(batch):

    oneHotVector = []

    for b in batch:

        v = np.zeros((sequenceSize, vocabLen))

        for itr, d, in enumerate(b):

            v[itr][d]=1

        oneHotVector.append(v)

    return oneHotVector
def Embedding(input_x, vocabLen, emb_size):

    with tf.variable_scope("emb", tf.AUTO_REUSE):

        emb_var = tf.get_variable(shape= [vocabLen, emb_size], dtype= tf.float32, name= "Emb_Var")

    emb_out = tf.nn.embedding_lookup(params= emb_var, ids= input_x)

    return emb_out


def model(emb_ouputs):

    

    print(emb_ouputs)

    emb_ouputs = tf.squeeze(emb_ouputs, axis= 2)

    print(emb_ouputs)

    cell = tf.nn.rnn_cell.GRUCell(124, kernel_initializer=tf.initializers.glorot_normal()) 

    print(cell)

    outputs, state = tf.nn.dynamic_rnn(cell, emb_ouputs, dtype = tf.float32)

    print(outputs)

    out = tf.layers.dense(outputs,  vocabLen)

    print(out)

    return out
char2idx = {u:i for i, u in enumerate(vocab)}

idx2char = np.array(vocab)



text_as_int = np.array([char2idx[c] for c in text])
sequenceSize = 100

dataLen = len(text_as_int) - len(text_as_int) % sequenceSize + 1

dataX = text_as_int[:dataLen-1].reshape((-1, sequenceSize, 1))

dataY = text_as_int[1:dataLen].reshape((-1, sequenceSize, 1))
tf.reset_default_graph()



X = tf.placeholder(tf.int32, [None, sequenceSize, 1])

Y = tf.placeholder(tf.float32, [None, sequenceSize, vocabLen])

lr = tf.placeholder(tf.float32, None)



emb_ouputs = Embedding(input_x= X, vocabLen = vocabLen, emb_size= 300)

pred = model(emb_ouputs)

computeCost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=pred))

trainOpt = tf.train.AdamOptimizer(lr).minimize(computeCost)
sess = tf.Session() 

tf.global_variables_initializer().run(session = sess)
dataX.shape
t = sess.run(pred, feed_dict = { X : dataX[:1] })

print(t.shape, dataY[:1].shape)

batchSize  = 128

subE = len(dataX)//batchSize

print(subE)

for i in range(3):

    for b in range( subE ):

        bX = dataX[ b*batchSize : (b+1)*batchSize ]

        bY = dataY[ b*batchSize : (b+1)*batchSize ]

#         print(idx2char[np.ravel(bX)])

#         print(idx2char[np.ravel(bY)])

        

        _, cost =  sess.run( [trainOpt, computeCost], feed_dict = { X : bX, Y :  oneHot(bY), lr : 0.001 } )

        if b==subE-1:

            print(i, cost)
d[:1].shape
#1

d = dataX[:1]

t = sess.run(pred, feed_dict = { X : d })

t = np.argmax(t,2)



print(''.join(np.ravel(idx2char[d])))

print("=="*20)

print(''.join(np.ravel(idx2char[t])))