import numpy as np

import pandas as pd

import nltk

import string

import matplotlib.pyplot as plt

from collections import Counter



from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.svm import SVC, LinearSVC, NuSVC

from sklearn.model_selection import train_test_split

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier



import tensorflow as tf

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



%matplotlib inline
df_train = pd.read_csv('../input/Reviews.csv')

df_train.head()
df_train = df_train[df_train.Score != 3]

df_train['Target'] = 'Pos'

df_train['Target'][df_train.Score < 3] = 'Neg'
df_train = df_train.drop(['Id', 'ProductId', 'UserId', 'ProfileName', 'HelpfulnessNumerator', 'HelpfulnessDenominator',

                         'Score', 'Time'], 1)
df_train['Finaltext'] = df_train.Summary.str.cat(df_train.Text, sep = ' . ')

df_train.Finaltext = df_train.Finaltext.astype(str)
df_train = df_train[:150000]
df_pos = df_train[df_train.Target == 'Pos']

df_neg = df_train[df_train.Target == 'Neg']
print('Positive entries:', len(df_pos), 'Negative entries:', len(df_neg))
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

exclude = set(string.punctuation)



def create_lexicon(data):

    lexicon = []

    for lines in data:

        if type(lines) is str:

            words = word_tokenize(lines.lower())

            lexicon += [w for w in words if w not in [stop_words,exclude]]

    lexicon = [lemmatizer.lemmatize(i) for i in lexicon]

    word_count = Counter(lexicon)

    final_lexicon = []

    for word in word_count:

        if (len(data) / 2) > word_count[word] > (len(data)/150):

            final_lexicon.append(word)

    return final_lexicon, word_count
lexicon_amazon, word_count_amazon = create_lexicon(df_train.Finaltext)
print(len(lexicon_amazon))



print(lexicon_amazon[:20])
def create_dataset(pos,neg,lexicon):

    dataset = []

    for lp in pos:

        words_pos = word_tokenize(lp)

        if type(words_pos) is str:

            words_pos = words_pos.lower()

        words_pos = [lemmatizer.lemmatize(i) for i in words_pos]

        features_pos = np.zeros(len(lexicon) + 2)

        features_pos[-1] = 1

        for word in words_pos:

            if word in lexicon:

                index = lexicon.index(word)

                features_pos[index] += 1

        dataset.append(features_pos)

    

    for ln in neg:

        words_neg = word_tokenize(ln)

        if type(words_neg) is str:

            words_neg = words_neg.lower()

        words_neg = [lemmatizer.lemmatize(i) for i in words_neg]

        features_neg = np.zeros(len(lexicon) + 2)

        features_neg[-1] = 0

        for word in words_neg:

            if word in lexicon:

                index = lexicon.index(word)

                features_neg[index] += 1

        dataset.append(features_neg)

    dataset = np.array(dataset)

    np.random.shuffle(dataset)

    return dataset
data = create_dataset(df_pos.Finaltext, df_neg.Finaltext, lexicon_amazon)
X = data[:, :-1]

y = data[:,-1]



def y2indicator(y):

    N = len(y)

    y = y.astype(np.int32)

    ind = np.zeros((N, 2))

    for i in range(N):

        ind[i, y[i]] = 1

    return ind



T = y2indicator(y)
X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(X, y, T, test_size=0.2, random_state=42)
#First, we create this function to calculate the accuracy. 



def accuracy(p, t):

    accuracy = np.mean(p == t)

    return accuracy



#Now let's define some parameters of our NN. You can defintely play with these to improve the speed and accuracy. 



max_iter = 7

print_period = 50

lr = 0.00005

reg = 0.001



N, D = X_train.shape

batch_sz = 1500

n_batches = int(N / batch_sz)



#I will be using one NN with 3 layers of 500 hidden nodes each. 



M1 = 500

M2 = 500

M3 = 500

K = 2



#These are the values to initialize the weights and biases. 



W1_init = np.random.randn(D, M1) / np.sqrt(N)

b1_init = np.zeros(M1)

W2_init = np.random.randn(M1, M2) / np.sqrt(M1)

b2_init = np.zeros(M2)

W3_init = np.random.randn(M2, M3) / np.sqrt(M2)

b3_init = np.zeros(M3)

W4_init = np.random.randn(M3, K) / np.sqrt(M3)

b4_init = np.zeros(K)



#And now these are the tf variables. 



X = tf.placeholder(tf.float32, shape=(None, D), name='X')

T = tf.placeholder(tf.float32, shape=(None, K), name='T')

W1 = tf.Variable(W1_init.astype(np.float32))

b1 = tf.Variable(b1_init.astype(np.float32))

W2 = tf.Variable(W2_init.astype(np.float32))

b2 = tf.Variable(b2_init.astype(np.float32))

W3 = tf.Variable(W3_init.astype(np.float32))

b3 = tf.Variable(b3_init.astype(np.float32))

W4 = tf.Variable(W4_init.astype(np.float32))

b4 = tf.Variable(b4_init.astype(np.float32))



#These are the activation values. I am using relu but this can be changed as well. 



Z1 = tf.nn.relu( tf.matmul(X, W1) + b1 )

Z2 = tf.nn.relu( tf.matmul(Z1, W2) + b2 )

Z3 = tf.nn.relu( tf.matmul(Z2, W3) + b3 )

Yish = tf.matmul(Z3, W4) + b4 





#This is our cost function, with will use Softmax. 



cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=Yish, labels=T))



#This line is for the optimizer. I am using RMSProp as it allows for momentum, but feel free to change it as well. 



train_op = tf.train.RMSPropOptimizer(lr, decay=0.99, momentum=0.9).minimize(cost)



#This is our prediction line



predict_op = tf.argmax(Yish, 1)



#And now we can start!



costs = []

init = tf.global_variables_initializer()

with tf.Session() as session:

    session.run(init)



    for i in range(0,max_iter):

        for j in range(0,n_batches):

            Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),:]

            Ybatch = T_train[j*batch_sz:(j*batch_sz + batch_sz),:]



            session.run(train_op, feed_dict={X: Xbatch, T: Ybatch})

            if j % print_period == 0:

                test_cost = session.run(cost, feed_dict={X: X_test, T: T_test})

                prediction = session.run(predict_op, feed_dict={X: X_test})

                acc = accuracy(prediction, y_test)

                print ("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, test_cost, acc))

                costs.append(test_cost)
print(classification_report(prediction, y_test))
