import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import random as ran
import pandas as pd
matplotlib.rc('figure', figsize=(15, 7))
# Intégration
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
#TRAIN
test.shape
#TEST
train.shape
X = train.drop(['label'],axis=1).values
Y = train.label.values
#Transformation de Y :
print("Y before : \n",Y)
print("Size of Y : ",len(Y))
mat = np.zeros((len(Y), 10))
for i in range (0,len(Y)):
    mat[i,Y[i]]=1
Y_=mat
print("----------------------")
print("Y after :\n",Y_)
print("Size of Y : ",Y_.shape)
def display_digit(num,x,y):
    print(y[num])
    label = y[num].argmax(axis=0)
    image = x[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
display_digit(150,X,Y_)
X = X.astype('float32')
Y_ = Y_.astype('float32')
sess = tf.Session()
#place holder, variable used to feed data into
#A placeholder exists solely to serve as the target of feeds. 
#It is not initialized and contains no data.

ph = tf.placeholder(tf.float32, shape = [None, X.shape[1]])
phy = tf.placeholder(tf.float32, shape=[None, 10])
ph
phy
#Weights and bias

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#Classifier function also known as multinomial logistic regression

y = tf.nn.softmax(tf.matmul(ph,W) + b)
y
#Construction de la fonction de cout, elle calcule la précision 
#en comparant les vrais valeurs de y_train au résultat de y par exemple

cross_entropy = tf.reduce_mean(-tf.reduce_sum(phy * tf.log(y), reduction_indices=[1]))
# Train
x_train = X[:5000,:]
y_train = Y_[:5000,:]
x_test = X[5000:10000,:]
y_test = Y_[5000:10000,:]
x_test.shape
rate_learning = 0.1
train_steps = 250
init = tf.global_variables_initializer()
sess.run(init)
#train using Gradient Descent
#perform the gradient with a chosen learning rate to try to minimize our cross_entropy

training = tf.train.GradientDescentOptimizer(rate_learning).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(phy,1)) # oui si c'est la bonne 0 sinon
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #score du modèle
for i in range(train_steps+1):
    sess.run(training, feed_dict={ph: x_train, phy: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(y, feed_dict={ph: x_test, phy: y_test})) +
              str(sess.run(accuracy, feed_dict={ph: x_test, phy: y_test})) + 
              '  Loss = ' + str(sess.run(cross_entropy, {ph: x_train, phy: y_train})))
for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
plt.colorbar()
plt.show()
x_train = mnist.train.images[:15,:]
y_train = mnist.train.labels[:15,:]
display_digit(0)
answer = sess.run(y, feed_dict={ph: x_train})
print(answer.argmax())
x_train = mnist.train.images[1560,:].reshape(1,784)
y_train = mnist.train.labels[1560,:]
# THIS GETS OUR LABEL AS A INTEGER
label = y_train.argmax()
# THIS GETS OUR PREDICTION AS A INTEGER
prediction = sess.run(y, feed_dict={ph: x_train}).argmax()
plt.title('Prediction: %d Label: %d' % (prediction, label))
plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
plt.show()
# phy = placeholder for y and y = classifier function x*W +b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(phy * tf.log(y), reduction_indices=[1]))
x_train = mnist.train.images[:5000,:]
y_train = mnist.train.labels[:5000,:]
x_test = mnist.test.images[:10000,:]
y_test = mnist.test.labels[:10000,:]
sess.run(tf.global_variables_initializer())

display_digit(3846)
x_train
y_train
x_test
y_test
training = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(phy,1)) # oui si c'est la bonne 0 sinon
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #score du modèle
for i in range(3000):
    #train the tensorflow for each steps
    sess.run(training, feed_dict={ph: x_train, phy: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + 
              str(sess.run(accuracy, feed_dict={ph: x_test, phy: y_test})) +
              '  Loss = ' + str(sess.run(cross_entropy, {ph: x_train, phy: y_train})))
def one_hot_encode(y):
    df = pd.DataFrame(data = np.zeros((y.shape[0],10)))
    for i in range(y.shape[0]):
        df.at[i,y[i]]=1  
    return(df)
import pandas as pd
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
datatrain = train.drop(['label'],axis=1).values[:train.shape[0]-10000,:]
datavalid = train.drop(['label'],axis=1).values[train.shape[0]-10000:train.shape[0],:]
image_data = datatrain[:,1:]

print(datatrain.shape)
print(datavalid.shape)
print(image_data.shape)
label = train.label[6951]
image = datatrain[6951].reshape([28,28])
plt.title('Example: %d  Label: %d' % (0, label))
plt.imshow(image, cmap=plt.get_cmap('gray_r'))
plt.show()
target = one_hot_encode(train.label).values[:train.shape[0]-10000,:]
print(train.label[30:35].values)
print(target[30:35,:])
ph = tf.placeholder(tf.float32, shape = [None, datatrain.shape[1]])
phy = tf.placeholder(tf.float32, shape=[None,10])
ph
phy
#Weights and bias

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#Classifier function also known as multinomial logistic regression

y = tf.nn.softmax(tf.matmul(ph,W) + b)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=phy, logits=y))

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(phy * tf.log(y_), reduction_indices=[1]))
#SET UP COMPLETE
train_step = tf.train.GradientDescentOptimizer(1.0e-4).minimize(cross_entropy)

sess = tf.InteractiveSession() # launch interactive session

tf.global_variables_initializer().run(session = sess) #initialize variables
steps = 2500
N = 1000
init=0
final = N

for _ in range(steps):
    setx = datatrain[init:final,]
    sety = target[init:final,:]
    sess.run(train_step, feed_dict={ph:setx,phy:sety})
    init += N
    final += N
    if final >= datavalid.shape[0]:
        ind = np.random.shuffle(np.arange(datavalid.shape[0]))
        init = 0
        final = N
    print("Progress {:2.1%}".format(_ / 2500), end="\r")
prediction = tf.equal(tf.argmax(y,1),tf.argmax(phy,1))  #Get True/False for predictions
targetvalid = one_hot_encode(train.label).values[train.shape[0]-10000:train.shape[0],:]
accuracy = tf.reduce_mean(tf.cast(prediction,tf.float32)) #Get mean accuracy
print(sess.run(accuracy,feed_dict={ph:datavalid, phy:targetvalid}))
