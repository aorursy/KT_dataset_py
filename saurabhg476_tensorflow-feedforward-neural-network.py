%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
X_train = train.loc[:,train.columns != 'label'].T 
y_train = train.loc[:,['label']].T
X_train = X_train / 255
X_train = X_train.astype(np.float32)

# X_cross = train.loc[30001:,train.columns != 'label'].T 
# y_cross = train.loc[30001:,['label']].T
# X_cross = X_cross / 255
# X_cross = X_cross.astype(np.float32)

X_test = test.T

#i0nitializing tensorflow variables
tf.reset_default_graph()

X = tf.placeholder(tf.float32,shape=(784,None))
Y = tf.placeholder(tf.float32,shape=(10,None))

def get_parameters():
    W1 = tf.get_variable("W1",(500,784),initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1",(500,1),initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",(100,500),initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",(100,1),initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",(10,100),initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",(10,1),initializer = tf.zeros_initializer())

    parameters = {
        "W1":W1,
        "b1":b1,
        "W2":W2,
        "b2":b2,
        "W3":W3,
        "b3":b3
    }
    
    return parameters
def forward_pass(X,parameters):
    W1,b1,W2,b2,W3,b3 = parameters["W1"],parameters["b1"],parameters["W2"],parameters["b2"],parameters["W3"],parameters["b3"]
    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    return Z3
parameters = get_parameters()
Z3 = forward_pass(X,parameters)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=tf.transpose(Z3),labels=tf.transpose(Y)))
def get_one_hot_encoding(X):
    sess = tf.Session()
    return sess.run(tf.one_hot(X,10,axis=0))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
costs=[]
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
        _,reduced_cost = sess.run([optimizer,cost],feed_dict={X:X_train,Y:get_one_hot_encoding(np.squeeze(y_train))})
        print(reduced_cost)
        costs.append(reduced_cost)

    plt.plot(costs)
    plt.show()

    ##saving the parameters
    optimized_parameters = sess.run(parameters)
    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Train Accuracy:", accuracy.eval({X:X_train,Y:get_one_hot_encoding(np.squeeze(y_train))}))
#     print ("Test Accuracy:", accuracy.eval({X:X_cross, Y: get_one_hot_encoding(np.squeeze(y_cross))}))
    
    predictions = sess.run(tf.argmax(Z3),feed_dict={X:X_test})

submission = {
    "ImageId": range(1,X_test.shape[1]+1), 
    "Label": predictions
}

pd.DataFrame.from_dict(submission).to_csv("submission.csv",index=False) 

