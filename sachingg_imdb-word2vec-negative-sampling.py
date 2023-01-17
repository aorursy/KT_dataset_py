# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
review = pd.read_csv("/kaggle/input/IMDB Dataset.csv")

review.tail()
import tensorflow as tf
raw_comments = review['review'].tolist()

import re

STOPWORDS = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'

cleanwords = re.compile(STOPWORDS)
vocab_size = 1000

all_comments = []

all_comments_set = set()

for comments in raw_comments:

    #Limit the vocab size to vovab_size

    if(len(all_comments_set) > vocab_size):

        break

    comments = re.sub(cleanwords,' ',comments)

    temp = comments.split()

    all_comments.append(temp)

    all_comments_set.update(temp) #for adding a list to set
vocab_size = len(all_comments_set)

print(len(all_comments_set))

word_to_int_dict = {}

int_to_word_dict = {}



for i,word in enumerate(all_comments_set):

    word_to_int_dict[word] = i

    int_to_word_dict[i] = word
print(word_to_int_dict['One'])

print(int_to_word_dict[word_to_int_dict['One']])

vocab_size = len(int_to_word_dict)

vocab_size
#The below function will create a oneHot Representation of all words in dictonary

len_dict = len(int_to_word_dict)

oneHot = np.zeros((len_dict,len_dict))

for key in int_to_word_dict:

    oneHot[key][key] = 1

oneHot[2]
#let's use a window size of last 3 words to create the context/target pairs

context_target_pair = []

for raw_comments in all_comments:

    n = len(raw_comments)

    for i in range(0,n-2):

        print("adding pair :",raw_comments[i],raw_comments[i+1])

        pair=(raw_comments[i],raw_comments[i+1])

        context_target_pair.append(pair)
pair_list = []



for i in range(0,len(context_target_pair)):

    word1 = context_target_pair[i][0]

    value1 = word_to_int_dict[word1]

    word2 = context_target_pair[i][1]

    value2 = word_to_int_dict[word2]

    #print(word1,word2)

    pair_list.append((value1,value2,1))



print(len(pair_list))

print(pair_list[0:2])

#let's use Negative Sampling approach to create word embeddings. This means that for every X,Y pair we need to add K invalid X,Y pairs

K=4

import random



datalen = len(pair_list)

for i in range(0,datalen):

    for j in range(0,K):

        rand_int = random.randint(0,vocab_size)

        pair_list.append((value1,value2,0))



print(len(pair_list))

m = len(pair_list)
random.shuffle(pair_list)

pair_list[0:20]
X_temp = []

Y_temp = []

X_target_temp = []

for i in range(0,m):

    X_temp.append(oneHot[pair_list[i][0]])

    Y_temp.append(pair_list[i][2])

    X_target_temp.append(oneHot[pair_list[i][1]])



#X = pd.DataFrame(X_train,columns=['x1','x2','valid'],dtype='float32')

X_temp[1:5]
X_train = pd.DataFrame(X_temp,dtype='float32')

X_target_train = pd.DataFrame(X_target_temp,dtype='float32')



Y_train = pd.DataFrame(Y_temp,columns=['target'],dtype='float32')
def create_placeholders(n_x):

    X1 = tf.placeholder(tf.float32,name='X',shape=(None,n_x))

    Y1 = tf.placeholder(tf.float32,name='Y',shape=(None))

    X1_target = tf.placeholder(tf.float32,name='X_target',shape=(None,n_x))



    return X1,Y1,X1_target



EMBEDDING_DIM = 300 



def get_minibatch(X,Y,X_target,batch_size):

    m,n = X.shape

    total_batch = (np.int32)(m/batch_size)

    left_over = (np.int32)(m%batch_size)



    minibatches = []

    start=0

    for i in range(0,total_batch):

        end=start+batch_size

        X_temp = X[start:end]

        X_target_temp = X_target[start:end]

        Y_temp = Y[start:end]

        minibatches.append((X_temp,X_target_temp,Y_temp))

        start = end

    

    X_temp = X[start:m]

    X_target_temp = X_target[start:m]

    Y_temp = Y[start:m]

    minibatches.append((X_temp,X_target_temp,Y_temp))

    

    return minibatches



def init_params(n_x):



    Xinitializer = tf.contrib.layers.xavier_initializer(dtype=tf.dtypes.float32)

    #xavier_initializer_conv2d is designed to keep the scale of the gradients roughly the same in all layers

    W1 = tf.Variable(Xinitializer(shape=(EMBEDDING_DIM,vocab_size)))

    W2 = tf.Variable(Xinitializer(shape=(vocab_size,EMBEDDING_DIM)))

    W3 = tf.Variable(Xinitializer(shape=(vocab_size,1)))



    b1 = tf.Variable(Xinitializer(shape=(EMBEDDING_DIM,1)))

    b2 = tf.Variable(Xinitializer(shape=(vocab_size,1)))

    b3 = tf.Variable(Xinitializer(shape=(1,1)))



    parameters = {

        'W1' : W1,

        'W2' : W2,

        'W3' : W3,

        'b1' : b1,

        'b2' : b2,

        'b3' : b3

    }

    

    return parameters



def fwd_move(X,Y,X_target,parameters):

    

    W1 = parameters['W1']

    W2 = parameters['W2']

    b1 = parameters['b1']

    b2 = parameters['b2']

    W3 = parameters['W3']

    b3 = parameters['b3']





    print("W1 shape is :",W1.shape)

    print("X shape is :",X.shape)

    z1 = tf.matmul(W1,tf.transpose(X)) + b1

    print("z1 shape is :",z1.shape)



    #We also need to make sure that input at is normalized. For Now we are leaving it

    a1 = tf.nn.relu(z1)

    print("a1 shape is :",a1.shape)

    print("W2 shape is :",W2.shape)

    print("b2 shape is :",b2.shape)



    z2 = tf.matmul(W2,a1) + b2

    print("z2 shape is :",z2.shape)

    a2 = tf.nn.relu(z2)



    #We have a2 here which represents a vector of vocab size binary clasification problems

    z3 = tf.matmul(X_target,W3) + b3

    print("z3 shape is :",z3.shape)



    return z3,parameters



def sigmoid_cost(z,Y):



    logit = z

    label=Y_train

    print("Logit Shape :",logit.shape)

    print("Label Shape :",label.shape)

    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=label))

    print("Inside Cost function Cost is: ",cost)

    return cost
def sigmoid(z):

    return (1/(1+np.exp(-z)))
print(X_target_train.shape)
sess = tf.Session()

def model(X_train,X_tatget_train,Y_train,num_epochs=1500,minibatch_size=32,learning_rate=0.001):

    m,n_x = X_train.shape

    parameters = init_params(n_x)

    X1,Y1,X_target1 = create_placeholders(n_x)

    z,params = fwd_move(X_train,Y_train,X_tatget_train,parameters)

    cost = sigmoid_cost(z,Y_train)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    total_batch = (np.int16)(m/minibatch_size)

    init_op = tf.initialize_all_variables()

    init = tf.global_variables_initializer()

    #Graph for tensorflow

    writer = tf.summary.FileWriter('C:/Users/sacgupt5/Documents/AI/graphs', sess.graph)

    sess.run(init_op)

    sess.run(init)

    sess.run(parameters)

    #sess.run(print("Shape of W1 is :",W1.shape))

    minibatches = get_minibatch(X_train,Y_train,X_target_train,minibatch_size)

    for epoch in range (0,num_epochs):

        #print("Epoch is :", epoch)

        minibatch_cost = 0

        epoch_minibatch_cost = 0



        for i in range (0,len(minibatches)):

            X_mini,X_target_mini,Y_mini = minibatches[i]

            #print("Minibatch Shape is ",X1.shape,Y1.shape)

            #z1,params = sess.run(fwd_move(X1,X_target1,Y1,parameters,feed_dict={X: X_mini, Y: Y_mini,X_target: X_target_mini,parameters:parameters}))

            _ , epoch_minibatch_cost = sess.run([optimizer, cost], feed_dict={X1: X_mini, Y1: Y_mini,X_target1: X_target_mini})

            #_ , epoch_minibatch_cost = sess.run([optimizer, cost], feed_dict={z:z1,Y: Y_mini})

            minibatch_cost = epoch_minibatch_cost + minibatch_cost

        epoch_cost = minibatch_cost/total_batch



        if(epoch % 100 == 0):

            print("Epoch Error is ",epoch_cost)



    print("Final Error is :",epoch_cost)



    #z_test,params = sess.run(fwd_move(X_test,Y_test,parameters))

    #predicted_cost = (sigmoid(z_test))

    #accuracy = tf.metrics.accuracy((predicted_cost),Y_test)

    #print("Accuracy is : ", accuracy)

    return parameters
  
parameters = model(X_train,X_target_train,Y_train,num_epochs=400,minibatch_size=32,learning_rate=0.001)
W1 = parameters['W1']

b1 = parameters['b1']

print(sess.run(W1))

print(sess.run(b1))

vector = sess.run(W1+b1)



temp = np.array(vector)

Embed = np.transpose(temp)
Embed.shape
def cosine_similarity(u, v):

    """

    Cosine similarity reflects the degree of similariy between u and v

        

    Arguments:

        u -- a word vector of shape (n,)          

        v -- a word vector of shape (n,)



    Returns:

        cosine_similarity -- the cosine similarity between u and v defined by the formula above.

    """



    distance = 0.0

    

    dot = np.dot(u,v)

    norm_u = np.linalg.norm(u)

    

    norm_v = np.linalg.norm(v)

    cosine_similarity = ((dot)/(norm_u * norm_v))



    return cosine_similarity
out = Embed[word_to_int_dict['out']]

Jake  = Embed[word_to_int_dict['Jake']]

wonderful = Embed[word_to_int_dict['wonderful']]

little = Embed[word_to_int_dict['little']]

print("cosine_similarity(out,Jake) = ", cosine_similarity(out, Jake))

print("cosine_similarity(wonderful,little) = ", cosine_similarity(wonderful, little))
watching = Embed[word_to_int_dict['watching']]

print("cosine_similarity(wonderful,with) = ", cosine_similarity(watching,wonderful))

def euclidean_dist(vec1, vec2):

    return np.sqrt(np.sum((vec1-vec2)**2))



def find_closest(word_index, vectors):

    min_dist = 10000 # to act like positive infinity

    min_index = -1

    query_vector = vectors[word_index]

    for index, vector in enumerate(vectors):

        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):

            min_dist = euclidean_dist(vector, query_vector)

            min_index = index

    return min_index
print(int_to_word_dict[find_closest(word_to_int_dict['with'],Embed)])

print(int_to_word_dict[find_closest(word_to_int_dict['out'],Embed)])
#print(int_to_word_dict['out'])

print(int_to_word_dict[find_closest(word_to_int_dict['wonderful'],Embed)])