# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df.shape
df.info()
df.describe()
df.head(3)
df.label.unique()
def image_show(x,y):

    if y=='Org':

        plt.imshow(df.iloc[x,1:].values.reshape(28,28), cmap='binary')

    else:

        plt.imshow(df.iloc[x,:].values.reshape(28,28), cmap='binary')
image_show(2,'Org')
y = df.label
df.drop('label', axis=1, inplace=True)
df = df/255.0
df.describe()
image_show(2,"")
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import train_test_split
ml = LabelBinarizer()
y_b = ml.fit_transform(y.values)
y_b.shape
X_train, X_test, y_train, y_test = train_test_split(df.as_matrix(), y_b, test_size=0.20, random_state=101)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train,y_train)
pred = rf.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
print (classification_report(y_test, pred))
print(confusion_matrix(ml.inverse_transform(y_test), ml.inverse_transform(pred)))
plt.imshow(X_test[30,:].reshape(28,28), cmap='binary')
ml.inverse_transform(y_test)[30]
ml.inverse_transform(pred)[30]
accuracy_score(y_test,pred)
df_test = pd.read_csv('../input/test.csv')
df_test.head(2)
df_test = df_test/255.0
pred_test = ml.inverse_transform(rf.predict(df_test.as_matrix()))
inpu = tf.placeholder(tf.float32, shape=[None, 784] )

outpu = tf.placeholder(tf.float32, shape=[None, 10] )

W1 = tf.Variable(tf.truncated_normal([784,20], stddev=0.1))

b1 = tf.Variable(tf.constant(0.1), [1,20])

W2 = tf.Variable(tf.truncated_normal([20,20], stddev=0.1))

b2 = tf.Variable(tf.constant(0.1), [1,20])

W3 = tf.Variable(tf.truncated_normal([20,20], stddev=0.1))

b3 = tf.Variable(tf.constant(0.1), [1,20])

W4 = tf.Variable(tf.truncated_normal([20,10], stddev=0.1))

b4 = tf.Variable(tf.constant(0.1), [1,10])

keep_prob = tf.placeholder(tf.float32)
Z1 = tf.add(tf.matmul(inpu,W1), b1)

A1 = tf.nn.relu(Z1)

Z2 = tf.add(tf.matmul(A1,W2), b2)

A2 = tf.nn.relu(Z2)

Z3 = tf.add(tf.matmul(A2,W3), b3)

A3 = tf.nn.relu(Z3)

Z4   = tf.add(tf.matmul(A3,W4), b4)

#Y_Hat = tf.nn.softmax(Z4)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=outpu, logits=Z4))
Opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
batch = 100

with tf.Session() as sess:

    init = tf.global_variables_initializer()

    sess.run(init)

    for i in range(5000):

        for j in range(0, X_train.shape[0], batch):

            _, c = sess.run([Opt,cross_entropy], feed_dict={inpu:X_train[j: j+batch], outpu:y_train[j: j+batch]})

        if i % 100 ==0:

            print(c)

    print(j)

    Y_Pred = sess.run(Z4, feed_dict={inpu:X_test})
Y_f=np.argmax(Y_Pred, axis=1)
y_t = y_test.argmax(axis=1)
y_t
Y_f
print(accuracy_score(y_t,Y_f))
'''df_s = pd.DataFrame(pred_test)

df_s.index.name='ImageId'

df_s.index+=1

df_s.columns=['Label']

df_s.to_csv('results.csv', header=True)'''