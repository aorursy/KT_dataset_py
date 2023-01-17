# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
def get_pixels(examples):
    out=[]
    for example in examples:
        out.append(example.split())
    return np.array(out,dtype=np.float32)


def convpool(x,W1,W2,W3,W4,b1,b2,b3,b4,pool_size):
    conv_out1=tf.nn.conv2d(x,W1,strides=[1,1,1,1],padding='SAME')
    conv_out1=tf.nn.bias_add(conv_out1,b1)
    conv_out1=tf.nn.relu(conv_out1)
    
    conv_out2=tf.nn.conv2d(conv_out1,W2,strides=[1,1,1,1],padding='SAME')
    conv_out2=tf.nn.bias_add(conv_out2,b2)  
    conv_out2=tf.nn.relu(conv_out2)

    conv_out2=tf.nn.max_pool(conv_out2,ksize=[1,pool_size[0],pool_size[1],1],strides=[1,2,2,1],padding='SAME')
    
    conv_out3=tf.nn.conv2d(conv_out2,W3,strides=[1,1,1,1],padding='SAME')
    conv_out3=tf.nn.bias_add(conv_out3,b3) 
    conv_out3=tf.nn.relu(conv_out3)
    
    
    conv_out4=tf.nn.conv2d(conv_out3,W4,strides=[1,1,1,1],padding='SAME')
    conv_out4=tf.nn.bias_add(conv_out4,b4)
    conv_out4=tf.nn.relu(conv_out4)
    
    conv_out4=tf.nn.max_pool(conv_out4,ksize=[1,pool_size[0],pool_size[1],1],strides=[1,2,2,1],padding='SAME')
   
    return conv_out4

def init_filters(shape):
    return np.random.randn(*shape)*np.sqrt(2/np.prod(shape[:-1]))
def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()
df=pd.read_csv('../input/fer2013/fer2013.csv')
pixels=get_pixels(df.iloc[:,1])
df=pd.concat([df,pd.DataFrame(pixels)],axis=1)
df.drop('pixels',inplace=True,axis=1)
del pixels

train=df[df['Usage']=='Training'].drop('Usage',axis=1)
test=df[df['Usage']=='PrivateTest'].drop('Usage',axis=1)

xtrain=train.iloc[:,1:].values
ytrain=train.iloc[:,0].values

xtest=test.iloc[:,1:].values
ytest=test.iloc[:,0].values

sns.heatmap(xtrain[3,:].reshape((48,48)))

np.unique(ytrain,return_counts=True)
smote=SMOTE(random_state=42)
xtrain,ytrain=smote.fit_sample(xtrain,ytrain)


xtrain=xtrain.reshape((50505,48,48,1))/255
xtest=xtest.reshape((3589,48,48,1))/255

hot=OneHotEncoder()
hot.fit(
        ytrain.reshape((ytrain.shape[0],1))
    )
        
ytrain=hot.transform(
        ytrain.reshape((ytrain.shape[0],1))
        )
ytest=hot.transform(
        ytest.reshape((ytest.shape[0],1))
        )


ytrain=ytrain.astype(np.int32)
ytest=ytest.astype(np.int32)

print(xtrain.dtype)
print(xtest.dtype)
M1=2000
M2=1000
K=7

batch_sz=1000
N=ytrain.shape[0]
n_batches=N//batch_sz
epochs=100

#weights and biases init
W1_shape=init_filters((5,5,1,16)).astype(np.float32)
W1=tf.Variable(W1_shape)
b1=tf.Variable(np.zeros(16).astype(np.float32))

W2_shape=init_filters((5,5,16,32)).astype(np.float32)
W2=tf.Variable(W2_shape)
b2=tf.Variable(np.zeros(32).astype(np.float32))

W3_shape=init_filters((5,5,32,64)).astype(np.float32)
W3=tf.Variable(W3_shape)
b3=tf.Variable(np.zeros(64).astype(np.float32))


W4_shape=init_filters((5,5,64,128)).astype(np.float32)
W4=tf.Variable(W4_shape)
b4=tf.Variable(np.zeros(128).astype(np.float32))

W5=tf.Variable((np.random.randn(12*12*128,M1)*np.sqrt(2.0/(12*12*128))).astype(np.float32))
b5=tf.Variable(np.zeros([M1]).astype(np.float32))

W6=tf.Variable((np.random.randn(M1,M2)*np.sqrt(2/M1)).astype(np.float32))
b6=tf.Variable(np.zeros([M2]).astype(np.float32))

W7=tf.Variable((np.random.randn(M2,K)*np.sqrt(2/M2)).astype(np.float32))
b7=tf.Variable(np.zeros([K]).astype(np.float32))


#placehoders
X=tf.placeholder(dtype=np.float32,shape=(batch_sz,48,48,1))
Y=tf.placeholder(dtype=np.int32,shape=(batch_sz,7))

print(W1)
print(W2)
print(W3)
print(W4)
print(W5)
print(W6)
print(W7)

print('biase')
print(b1)
print(b2)
print(b3)
print(b4)
print(b5)
print(b6)
print(b7)
print('placeholders')
print(X)
print(Y)


#logits
out=convpool(X,W1,W2,W3,W4,b1,b2,b3,b4,(2,2))


out_shape=out.get_shape().as_list()

out=tf.reshape(out,[out_shape[0],np.prod(out_shape[1:])])


out=tf.nn.relu(tf.matmul(out,W5)+b5)
out=tf.nn.relu(tf.matmul(out,W6)+b6)
out=tf.matmul(out,W7)+b7

predict=tf.argmax(out,axis=1)


cost=tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=out,
                    labels=Y
                )
        )
        
opt=tf.train.AdamOptimizer(0.001).minimize(cost)
print(out)
print(opt)


init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    cost=[]
    for i in range(epochs):
        for j in range(n_batches):
            xb,yb=shuffle(xtrain,ytrain)
            x=xb[j*batch_sz:(j*batch_sz)+batch_sz]
            
            y=yb[j*batch_sz:(j*batch_sz)+batch_sz]

            y=y.toarray()
            sess.run(opt,feed_dict={X:x,Y:y})
            
            N_test=xtest.shape[0]
            test_n_batches=N_test//batch_sz
            if j%20==0:
                for k in range(test_n_batches):
                    xtb=xtest[k*batch_sz:(k*batch_sz)+batch_sz]
                    ytb=ytest[k*batch_sz:(k*batch_sz)+batch_sz]
                    ytb=ytb.toarray()
                    ypred=sess.run(predict,feed_dict={X:xtb,Y:ytb})
                    print(np.mean(ypred==np.argmax(ytb,axis=1)))
                    print(sigmoid_cost(np.argmax(ytb,axis=1),ypred))


























