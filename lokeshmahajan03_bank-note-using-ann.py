import numpy as np
import pandas as pd
import os
print(os.listdir("../input"))
from sklearn import preprocessing
df = pd.read_csv("../input/data_banknote_authentication.csv")
df.head()
from sklearn.utils import shuffle
x = shuffle(df , random_state=20)
#df = pd.DataFrame(preprocessing.normalize(df))
features = x[['3.6216','8.6661','-2.8073','-0.44699']]
features.shape
features.head()
features = pd.DataFrame(preprocessing.normalize(features))
target = pd.get_dummies(df['0'])
target.shape
target.head(10)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)
target = np.array(target)
target = target.reshape([-1,2])
target.shape
import tensorflow as tf
tf.reset_default_graph()
_X = tf.placeholder(tf.float32,shape = [None,4])
_Y = tf.placeholder('float',shape = [None,2])

input_size = 4
hidden1 = 3
hidden2 = 3
hidden3 = 3
output = 2
# weight = {
#     'layer1': tf.Variable(tf.random_normal([input_size,hidden1])),
#     'layer2': tf.Variable(tf.random_normal([hidden1,hidden2])),
#     'layer3': tf.Variable(tf.random_normal([hidden2,hidden3])),
#     'layer4': tf.Variable(tf.random_normal([hidden3,output]))
# }
# bias = {
#     'layer1': tf.Variable(tf.zeros([hidden1])),
#     'layer2': tf.Variable(tf.zeros([hidden2])),
#     'layer3': tf.Variable(tf.zeros([hidden3])),
#     'layer4': tf.Variable(tf.zeros([output]))
# }
def model(X):
    layer1 = tf.layers.dropout(tf.layers.dense(X,units=2,use_bias=True),rate=0.1)
    layer1 = tf.nn.relu(layer1)
    layer2 = tf.layers.dropout(tf.layers.dense(layer1,units=2,use_bias=True),rate=0.1)
    layer2 = tf.nn.relu(layer2)
    layer3 = tf.layers.dropout(tf.layers.dense(layer2,units=2,use_bias=True),rate=0.1)
    layer3 = tf.nn.relu(layer3)
    layer4 = tf.layers.dropout(tf.layers.dense(layer3,units=2,use_bias=True),rate=0.1)
    layer4 = tf.nn.relu(layer4)
    output  = tf.nn.sigmoid(layer4)
    return output
pred = model(_X)
pred
# loss = tf.reduce_mean(tf.square(_Y-pred))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=_Y))
get_output = tf.nn.softmax(pred)

train1 = tf.train.GradientDescentOptimizer(learning_rate=0.03).minimize(loss)
from sklearn.metrics import accuracy_score
# y_pred = dnn_clf.predict(X_test)
_acc = tf.equal(tf.argmax(_X,1),tf.argmax(pred,1))
accuracy = tf.reduce_mean(tf.cast(_acc, tf.float32))
training_steps = 600
init = tf.global_variables_initializer()
# with tf.Session() as sess:
sess = tf.Session()
sess.run(init) 
for itr in range(training_steps):
    features,target = shuffle(features,target,random_state=42)
    _,_loss,_pred,_accu = sess.run([train1,loss,pred,accuracy],feed_dict={_X:X_train,_Y:y_train})
    _loss2,_pred2,_accu2 = sess.run([loss,pred,accuracy],feed_dict={_X:X_test,_Y:y_test})
    print("Train Epochs:",itr,"Loss:",_loss,'acc:',_accu)
    print("Test Epochs:",itr,"Loss:",_loss2,'acc:',_accu2)


