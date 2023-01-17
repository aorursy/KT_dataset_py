from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()
import numpy as np
def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim
tokens1 = ["I", "am", "a", "bad","boy"]
tokens2 = ["I", "am", "a", "good","girl"]
vectors1 = elmo.embed_sentence(tokens1)
vectors2 = elmo.embed_sentence(tokens2)
print(vectors1.shape)
print(vectors2.shape)
cos_sim(vectors1[0][3], vectors2[0][3]) 
import numpy as np
import tensorflow as tf
input_x = tf.placeholder(tf.float32,shape=[None,30],name='inputs')
labels = tf.placeholder(tf.int32,shape=[None,],name='labels')
with tf.variable_scope('weight',reuse=tf.AUTO_REUSE):
    W1 = tf.get_variable('w1',shape=[30,128])
    b1 = tf.get_variable('b1',shape=[128])
    W2 = tf.get_variable('w2',shape=[128,4])
    b2 = tf.get_variable('b2',shape=[4])

#inference part
logits = tf.nn.relu(tf.matmul(input_x,W1)+b1)
logits = tf.matmul(logits,W2)+b2
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#显示可训练的变量
variables_names = [v.name for v in tf.trainable_variables()]
values = sess.run(variables_names)
for k, v in zip(variables_names, values):
    print ("Variable: ", k)
    print ("Shape: ", v.shape)
    #print (v)
#将不同的学习率应用到不同的层上
var1 = tf.trainable_variables()[:2]
var2 = tf.trainable_variables()[2:]
train_op1 = tf.train.GradientDescentOptimizer(0.00001).minimize(loss, var_list=var1) 
train_op2 = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, var_list=var2)
train_op = tf.group(train_op1, train_op2)
