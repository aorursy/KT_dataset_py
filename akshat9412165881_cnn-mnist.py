from tensorflow.contrib.layers import fully_connected

from tensorflow.keras.layers import Conv2D,MaxPool2D,Dense,BatchNormalization,Dropout

from tensorflow.keras.preprocessing.image import ImageDataGenerator as IDG

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import seaborn as sns

import tensorflow as tf

import pandas as pd

import numpy as np

import cv2
def obtain_params(layer_no,csv):

    activ ={'relu':tf.nn.relu,'tanh':tf.tanh,'sigm':tf.sigmoid,'None':None}

    bool = {'TRUE':True,'FALSE':False,'None':None,'Auto':tf.AUTO_REUSE}

    type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout = csv.iloc[layer_no]

    if filter!='None':filter=int(filter)

    if kernel!='None':kernel=int(kernel)

    if stride!='None':stride=int(stride)

    if dropout!='None':dropout=float(dropout)

    activation = activ[activation]

    reuse=bool[reuse];is_train = bool[is_train]

    return(type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout)
def update_weights(parameters):

    a=[variable.name for variable in tf.trainable_variables()]

    a=[a[i][:len(a[i])-2] for i in range(len(a))]

    for i in range(len(a)):

        ind = a[i].find('/')

        with tf.variable_scope(a[i][:ind],reuse=True):

            var = tf.get_variable(a[i][ind+1:])

        forge = tf.assign(var,best_weights)

    return forge.eval(feed_dict={best_weights:parameters[i]})



def check_accuracy(new_val,prev_val,counter,lr,lr_dec_limit,best_param):

    if new_val>prev_val:

        prev_val = new_val

        best_param = sess.run(params)

        counter = 0

    else:

        counter+=1

    if counter>lr_dec_limit:

        lr=lr/10

        counter=0

    return (prev_val,best_param,counter,lr)



def create_fig(output):

    """Create the figures of convolved feature maps from the output array"""

    shape = output.shape[3]

    factors = np.array(range(1,shape+1))

    factors = factors[shape%factors==0]

    n = len(factors)

    if n%2!=0:

        up=factors[np.int(n/2)]

        lo=factors[np.int(n/2)]

    else:

        up=factors[int(n/2)]

        lo=factors[int(n/2)-1]

    for i in range(1,shape+1):

        plt.subplot(up,lo,i)

        plt.imshow(output[0,:,:,i-1],cmap="Greys")

    plt.show()

    return()
df_train = pd.read_csv("../input/digit-recognizer/train.csv").iloc[:]

y_train = df_train.label

X_train = df_train.drop(["label"],axis=1)

df_test = pd.read_csv("../input/digit-recognizer/test.csv").iloc[:]

class fetch_mnist:

    def __init__(self,X=X_train,y=y_train,Xt=df_test):

        self.X_train = X/255

        self.y_train = y

        self.X_test = Xt/255

    def fetch_train_batch(self,n,cnn=False):

        rand_indx = np.random.permutation(len(self.X_train))

        self.X_train = self.X_train.iloc[rand_indx]

        self.y_train = self.y_train.iloc[rand_indx]

        if cnn==False:

            return(self.X_train[:n],self.y_train[:n])

        if cnn==True:

            shape = np.r_[n,28,28,1]

            data=self.X_train[:n]

            X_out = np.zeros(shape)

            X_out[:,:,:,0] = [np.array(data.iloc[i]).reshape(28,28) for i in range(data.shape[0])]

            return(X_out,self.y_train[:n])

    def fetch_test_batch(self,n,cnn=False):

        if cnn==False:

            return(self.X_test[:n])

        if cnn==True:

            shape = np.r_[n,28,28,1]

            data=self.X_test[:n]

            X_out = np.zeros(shape)

            X_out[:,:,:,0] = [np.array(data.iloc[i]).reshape(28,28) for i in range(data.shape[0])]

            return(X_out)

    def n_samples(self,tt):

        if tt == "train":return(len(self.X_train))

        if tt == "test":return(len(self.X_test))

batch_size = 200

lr = 0.01

n_iteration = 300

a=fetch_mnist()

x,y = a.fetch_train_batch(batch_size,cnn=True)

batch_norm_momentum=0.75

csv = pd.read_csv("../input/lenet5-parameters/lenet5.csv")
tf.reset_default_graph()

X=tf.placeholder(dtype=tf.float32,shape=(None,28,28,1),name="Features")

Y = tf.placeholder(dtype=tf.int32,shape=(None),name="Labels")

padded_input = tf.pad(X, [[0, 0], [2, 2], [2, 2], [0, 0]], "CONSTANT")

global_step = tf.Variable(0,dtype=tf.int32,trainable=False)

learning_rate = tf.train.exponential_decay(lr,global_step,100000,0.95)

data_gen = IDG(rotation_range=15)
################Convolution Layer 1

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(0,csv)

conv1 = Conv2D(filters=filter,kernel_size=[kernel,kernel],strides=[stride,stride],padding=padding,activation=activation)

out=conv1(padded_input)

#################Pooling Layer 1

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(1,csv)

pool1 = MaxPool2D(pool_size=(kernel,kernel),strides=stride)

out=pool1(out)

#################Convolution Layer 2

batchnorm1=BatchNormalization(momentum=batch_norm_momentum)

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(2,csv)

conv2 = Conv2D(filters=filter,kernel_size=[kernel,kernel],strides=[stride,stride],padding=padding,activation=activation)

out=conv2(batchnorm1(out))

#################Pooling Layer 1

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(3,csv)

pool2 = MaxPool2D(pool_size=(kernel,kernel),strides=stride)

out=pool2(out)

#################Convolution Layer 3

batchnorm2=BatchNormalization(momentum=batch_norm_momentum)

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(4,csv)

conv3 = Conv2D(filters=filter,kernel_size=[kernel,kernel],strides=[stride,stride],padding=padding,activation=activation)

out=conv3(batchnorm2(out))

################Fully Connected Layer 1

out = tf.reshape(out,(tf.shape(out)[0],out.get_shape()[3]))

batchnorm3=BatchNormalization(momentum=batch_norm_momentum)

drop1=Dropout(0.1)

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(5,csv)

fc1 = Dense(units=kernel,activation=activation)

out = fc1(drop1(batchnorm3(out)))

################Fully Connected Layer 2

batchnorm4=BatchNormalization(momentum=batch_norm_momentum)

drop2=Dropout(0.2)

type,filter,kernel,stride,padding,activation,scope,reuse,is_train,dropout=obtain_params(6,csv)

fc2 = Dense(units=kernel,activation=None)

out = fc2(drop2(batchnorm4(out)))
xentry=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y,logits=out)

loss = tf.reduce_sum(xentry,axis=0)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

training_op = optimizer.minimize(loss,global_step=global_step)
correct=tf.nn.in_top_k(out,Y,1)

al_in = tf.reduce_mean(tf.cast(correct,tf.float32))

params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='')

saver = tf.train.Saver()

init = tf.global_variables_initializer()
with tf.Session() as sess:

    acc_val=np.empty((1))

    sess.run(init)

    X_data,y_data = a.fetch_train_batch(a.n_samples('train'),cnn=True)

    for X_batch,y_batch in data_gen.flow(X_data,y_data,batch_size=batch_size):

        sess.run(training_op,feed_dict={X:X_batch,Y:y_batch})

        output=al_in.eval(feed_dict={X:X_batch,Y:y_batch})

        acc_val=np.append(acc_val,output)

        if global_step.eval()%50==0:print("Iteration:",global_step.eval(),"\taccuracy",output)

        if global_step.eval()==2000:break

# Obtaining prediction for all the samples

    X_batch,y_val = a.fetch_train_batch(a.n_samples("train"),cnn=True)

    y_pred = np.argmax(out.eval(feed_dict={X:X_batch}),axis=1)

    saver.save(sess,"../working/cnn_mnist_2.ckpt")

    sess.close()
acc_val = np.delete(acc_val,0)

plt.figure(figsize=(20,10))

plt.title("Accuracy v/s Iterations",fontdict={'fontsize':20})

plt.grid(True)

plt.xlabel("Iterations")

plt.ylabel("Accuracy")

plt.plot(acc_val)

plt.show()

#Evaluating the confusion matrix

cm = confusion_matrix(y_val, y_pred) 



# Transform to df for easier plotting

cm_df = pd.DataFrame(cm,

                     index = [str(i) for i in range(10)], 

                     columns = [str(i) for i in range(10)])



plt.figure(figsize=(13,10))

sns.heatmap(cm_df, annot=True)

plt.title("Confusion Matrix",fontdict={'fontsize':20})

plt.xlabel("Predictions")

plt.ylabel("Actual Values")

plt.show()
out_1 = conv1(padded_input)

with tf.Session() as sess:

    saver.restore(sess,"../working/cnn_mnist_2.ckpt")

    X_test=a.fetch_test_batch(1,cnn=True)

    output = out_1.eval(feed_dict={X:X_test})

    sess.close()

plt.imshow(X_test[0,:,:,0],cmap="Greys")

plt.title("Original Figure",fontdict={'fontsize':20})
fig = plt.figure(figsize=(10,10))

plt.suptitle("Filter's effect from Convolution layer 1",fontsize=20)

create_fig(output)
out_2 = conv3(conv2(pool1(conv1(padded_input))))

with tf.Session() as sess:

    saver.restore(sess,"../working/cnn_mnist_2.ckpt")

    X_test=a.fetch_test_batch(1,cnn=True)

    output = out_2.eval(feed_dict={X:X_test})

    sess.close()

fig = plt.figure(figsize=(10,10))

plt.suptitle("Filter's effect from Last Convolution layer",fontsize=20)

create_fig(output)
with tf.Session() as sess:

    saver.restore(sess,"../working/cnn_mnist_2.ckpt")

    X_test= a.fetch_test_batch(28000,cnn=True)

    acc_test = out.eval(feed_dict={X:X_test})

    out_put=np.argmax(acc_test,axis=1)

    sess.close()

data = pd.DataFrame({'ImageId':range(1,len(X_test)+1),'Label':out_put})

data.to_csv("../working/cnn_mnist_1.csv",index=False)

print(data.head())