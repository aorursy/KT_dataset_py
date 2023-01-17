# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/spam.csv',encoding='latin1')

data.head()
del data['Unnamed: 2']

del data['Unnamed: 3']

del data['Unnamed: 4']
data.head()
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

data.head()
y = data['v1'].as_matrix()

X_text = data['v2'].as_matrix() 

print(X_text.shape)

print(y.shape)
from sklearn.feature_extraction.text import CountVectorizer

from nltk.corpus import stopwords

sw = stopwords.words("english")

cv = CountVectorizer(stop_words =sw)

tcv = cv.fit_transform(X_text).toarray()

#print(cv.vocabulary_)

print(len(tcv[0,:]))
print(tcv.shape)

print(y.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(stop_words=sw,lowercase=True)

X = vectorizer.fit_transform(X_text).toarray()

print(X.shape)

print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.202, random_state=42)

print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

y_test.shape
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

clf = LogisticRegression()

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy_score(y_test,pred)
from sklearn.naive_bayes import GaussianNB

clf = GaussianNB()

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy_score(y_test,pred)
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(hidden_layer_sizes=(500,500))

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy_score(y_test,pred)
from sklearn import tree

clf = tree.DecisionTreeClassifier()

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy_score(y_test,pred)
from sklearn.svm import SVC

clf = SVC(gamma=0.1,C=1,kernel='rbf')

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

accuracy_score(y_test,pred)
n_classes = 2

y_n_train = np.zeros((y_train.size,n_classes)).astype(int)

print(y_n_train.shape)

k = 0

for i in y_train:

    y_n_train[k,i] = 1

    k+=1

print(y_n_train)
import tensorflow as tf

from tensorflow.contrib import rnn

epochs = 25

n_classes = 2

batch_size = 78

chunk_size = 97

n_chunks = 88

rnn_size = 78

x = tf.placeholder('float',[None,n_chunks,chunk_size])

y = tf.placeholder('float')

#from tensorflow.python.ops import rnn, rnn_cell

def recurrent_neural_network(x):

    layer = {'weights':tf.Variable(tf.random_normal([rnn_size,n_classes])),

             'biases':tf.Variable(tf.random_normal([n_classes]))}



    x = tf.unstack(x, n_chunks, 1)

    #print x.shape

    lstm_cell = rnn.BasicLSTMCell(rnn_size, forget_bias=1.0)



    # Get lstm cell output

    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)



    output = tf.matmul(outputs[-1],layer['weights']) + layer['biases']



    return output

def train_neural_network(x):

    prediction = recurrent_neural_network(x)

    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y) )

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    

    

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())



        for epoch in range(epochs):

            epoch_loss = 0

            i = 0

            while i<len(X_train):

                start = i

                end = i + batch_size

                #print(start,end)

                epoch_x, epoch_y = X_train[start:end],y_n_train[start:end]

                epoch_x = epoch_x.reshape((batch_size,n_chunks,chunk_size))

                #print epoch_x.shape

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

                i += batch_size



            print('Epoch', epoch, 'completed out of',epochs,'loss:',epoch_loss)



        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))



        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        print('Accuracy:',accuracy.eval({x:X_train.reshape((-1, n_chunks, chunk_size)), y:y_n_train}))

        #pred = sess.run(prediction,feed_dict={x:X_test.reshape((-1, n_chunks, chunk_size))})

        #corr = tf.argmax(pred,1)

        #corr = sess.run(corr)

        #print(corr)

        #k = [i+1 for i in range(len(corr))]

        #yg = pd.DataFrame({'ImageId':pd.Series(k),'Label':pd.Series(corr)})

        #yg.to_csv('ans.csv',index=False)

train_neural_network(x)