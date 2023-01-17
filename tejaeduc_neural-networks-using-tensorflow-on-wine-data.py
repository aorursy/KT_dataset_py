import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
%matplotlib inline
colnames = ['Class','Alcohol','Malic acid','Ash','Alcalinity of ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid phenols','Proanthocyanins','Color intensity','Hue','dilute','Proline']
df = pd.read_csv('../input/wine.data.txt',names = colnames,index_col = False)
df.head()
df.isnull().sum()
df.info()
df.Class.value_counts()
df.corr()
df = pd.get_dummies(df, columns=['Class'])
labels = df.loc[:,['Class_1','Class_2','Class_3']]
labels = labels.values
features = df.drop(['Class_1','Class_2','Class_3','Ash'],axis = 1)
features = features.values
print(type(labels))
print(type(features))
print(labels.shape)
print(features.shape)
train_x,test_x,train_y,test_y = train_test_split(features,labels)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
scale = MinMaxScaler(feature_range = (0,1))
train_x = scale.fit_transform(train_x)
test_x = scale.fit_transform(test_x)
print(train_x[0])
print(train_y[0])
X = tf.placeholder(tf.float32,[None,12]) # Since we have 12 features as input
y = tf.placeholder(tf.float32,[None,3])  # Since we have 3 outut labels
weights1 = tf.get_variable("weights1",shape=[12,80],initializer = tf.contrib.layers.xavier_initializer())
biases1 = tf.get_variable("biases1",shape = [80],initializer = tf.zeros_initializer)
layer1out = tf.nn.relu(tf.matmul(X,weights1)+biases1)

weights2 = tf.get_variable("weights2",shape=[80,50],initializer = tf.contrib.layers.xavier_initializer())
biases2 = tf.get_variable("biases2",shape = [50],initializer = tf.zeros_initializer)
layer2out = tf.nn.relu(tf.matmul(layer1out,weights2)+biases2)

weights3 = tf.get_variable("weights3",shape=[50,3],initializer = tf.contrib.layers.xavier_initializer())
biases3 = tf.get_variable("biases3",shape = [3],initializer = tf.zeros_initializer)
prediction =tf.matmul(layer2out,weights3)+biases3
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
acc = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(201):
        opt,costval = sess.run([optimizer,cost],feed_dict = {X:train_x,y:train_y})
        matches = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(matches, 'float'))
        acc.append(accuracy.eval({X:test_x,y:test_y}))
        if(epoch % 100 == 0):
            print("Epoch", epoch, "--" , "Cost",costval)
            print("Accuracy on the test set ->",accuracy.eval({X:test_x,y:test_y}))
    print("FINISHED !!!")

plt.plot(acc)
plt.ylabel("Accuracy")
plt.xlabel("Epochs")