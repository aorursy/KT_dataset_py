# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
X=np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')

Y=np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')





# In[5]:





print("Input shape: [%s,%s,%s]" %X.shape)

print("Output shape: [%s,%s]" %Y.shape)





# 1. So there are 2062 images with 64*64 pixel dimensions. Pictures are black and white.

# 2. Total labels for 2062 images are given but there are 10 classes. That means we have 10 digits to be identified



# #### For simplicity sake, we will use sign 0 and 1 only



# In[6]:





zero=Y[204:408]

one=Y[822:1027]





# In[7]:





y_zero=np.zeros(len(zero))

y_one=np.ones(len(one))





# In[8]:





x_zero=X[204:408]

x_one=X[822:1027]



b=np.concatenate((y_zero,y_one)).reshape(1,409)

a=np.concatenate((x_zero,x_one)).reshape(-1,409)





# In[9]:





print("Each example has  %s features/pixels" %a.shape[0])

print("There are %s examples" %a.shape[1])

print("There are %s targets" %b.shape[1])

print("Each examples is a %s by %s grid of pixels" %(np.sqrt(a.shape[0]),np.sqrt(a.shape[0])))





# In[10]:





plt.imshow(x_zero[2])

plt.show()
plt.imshow(x_one[3])





# ##### Split dataset in Train, Dev and Test sets



# #### Implementing 2-layer NN from Scratch

# 1. Initialize Parameters

# 2. Compute Forward Propagation 

# 3. Compute Cost

# 4. Compute Backward propagation

# 5. Update Parameters



# In[12]:





from sklearn.model_selection import train_test_split





# In[77]:





X_train, X_test, y_train, y_test = train_test_split(a.T, b.T, test_size=0.10, random_state=42)





# In[78]:





X_train=X_train.T

X_test=X_test.T

y_train=y_train.T

y_test=y_test.T





# In[79]:





# n_h = total features

# m = no of examples



# X_shape is (n_h,m)

# Y_shape is (m,10)





# In[80]:





# 1.Initialize parameters



def fn_initialize_params(layer_dims):

    """

    Initialize parameters W and b

    """

    

    W1=np.random.rand(layer_dims[1],layer_dims[0])*.01

    b1=np.zeros((layer_dims[1],1))

    W2=np.random.rand(layer_dims[2],layer_dims[1])*.01

    b2=np.zeros((layer_dims[2],1))

    W3=np.random.rand(layer_dims[3],layer_dims[2])*.01

    b3=np.zeros((layer_dims[3],1))

    

    params_cache=(W1,b1,W2,b2,W3,b3)

    

    return params_cache



def fn_relu_activation(X):

    """

    Relu Activation Function

    """    

    return X * (X > 0)

    



def fn_sigmoid_activation(X):

    """

    Sigmoid Activation Function

    """

    

    return (1/(1+np.exp(-X)))



# 2. Compute Forward Propagation



def fn_forward_prop(X,params_cache):

    """

    Compute forward propagation

    """

    (W1,b1,W2,b2,W3,b3) = params_cache

    

    Z1=np.dot(W1,X)+b1

    A1=fn_relu_activation(Z1)

    

    Z2=np.dot(W2,A1)+b2

    A2=fn_relu_activation(Z2)

    

    Z3=np.dot(W3,A2)+b3

    A3=fn_sigmoid_activation(Z3)

    

    

    forward_cache = (Z1,A1,Z2,A2,Z3,A3)



    return forward_cache

    



# 3. Compute Cost

def fn_cost(m,y,forward_cache):

    """

    Compute Cost

    """

    (Z1,A1,Z2,A2,Z3,A3)=forward_cache

    J=(-1/m)*(np.sum(np.multiply(y,np.log(A3)) + np.multiply((1-y),np.log(1-A3))))

    

    return J         







    

def fn_backward_prop(x,Y,m,forward_cache,params_cache):

    """

    Compute backward propagation

    """

    (Z1,A1,Z2,A2,Z3,A3)=forward_cache

    (W1,b1,W2,b2,W3,b3)=params_cache

    

    

    dZ3=Y-A3 

    dW3=1/m*(np.dot(dZ3,A2.T))

    db3=1/m*(np.sum(dZ3,axis=1,keepdims=True))

    dA2=np.dot(W3.T,dZ3)

    

    dZ2=dA2*np.int64(A2>0)  ##for Relu use this np.int64(A2>0) ## For sig use this (1-np.power(A2, 2))

    dW2=1/m*(np.dot(dZ2,A1.T))

    db2=1/m*(np.sum(dZ2,axis=1,keepdims=True))

    dA1=np.dot(W2.T,dZ2)

    

    dZ1=dA1*np.int64(A1>0)   ##for Relu use this np.int64(A1>0) ## For sig use this (1-np.power(A1, 2))

    dW1=1/m*(np.dot(dZ1,x.T))

    db1=1/m*(np.sum(dZ1,axis=1,keepdims=True))

    

    grads_cache=(dZ3,dW3,db3,dA2,dZ2,dW2,db2,dA1,dZ1,dW1,db1)

    

    return grads_cache



def fn_update_params(params_cache,grads_cache,learning_rate):

    

    (W1,b1,W2,b2,W3,b3)=params_cache

    (dZ3,dW3,db3,dA2,dZ2,dW2,db2,dA1,dZ1,dW1,db1)=grads_cache

    

    W1=W1-learning_rate*dW1

    b1=b1-learning_rate*db1

    W2=W2-learning_rate*dW2

    b2=b2-learning_rate*db2

    W3=W3-learning_rate*dW3

    b3=b3-learning_rate*db3

    

    params_cache=(W1,b1,W2,b2,W3,b3)

    

    return params_cache





def fn_predict(A3):

    Y_pred=np.zeros((1,A3.shape[1]))

    for i in range(A3.shape[1]):

        if A3[0,i]<0.5:

            Y_pred[0,i]=0

        else:

            Y_pred[0,i]=1

            

    return Y_pred





# In[81]:





cost_df=pd.DataFrame()

cost_df['index']=0

cost_df['cost']=0





# In[88]:





def model(X_train,y_train,learning_rate,layer_dims,iterations):

    

    params_cache=fn_initialize_params(layer_dims)

    

    for i in range(iterations):

        

        forward_cache=fn_forward_prop(X_train,params_cache)

        if (i%100==0):

            print("Cost after iteration %s is %s" %(i,fn_cost(X_train.shape[1],y_train,forward_cache)))



        grads_cache=fn_backward_prop(X_train,y_train,X_train.shape[1],forward_cache,params_cache)

        params_cache=fn_update_params(params_cache,grads_cache,learning_rate)

        

        cost_df.loc[i,'index']=i

        cost_df.loc[i,'cost']=fn_cost(X_train.shape[1],y_train,forward_cache)

        

    

    Y_pred=fn_predict(forward_cache[5])

    accuracy=np.mean(np.abs(y_train-Y_pred))*100

    print('\n')

    print("Accuracy is %s"  %accuracy)

    

    plt.plot(cost_df['index'],cost_df['cost'])        





# In[89]:





layer_dims=[X_train.shape[0],10,5,1]

learning_rate=.1





# In[90]:





model(X_train,y_train,learning_rate,layer_dims,1000)

from tensorflow.python.framework import ops

ops.reset_default_graph()
# ### Model through Tensorflow - For 2 classes



# In[61]:





X_train.shape

y_train.shape

X_test.shape

y_test.shape





import tensorflow as tf2





import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



# In[85]:





#tf.reset_default_graph()





# In[86]:





x=tf.placeholder(tf.float32,shape=(4096,None),name='x')

y=tf.placeholder(tf.float32,shape=(1,None),name='y')





# In[87]:





## initialize params

W1=tf.get_variable('W1',shape=[10,4096],dtype=tf.float32,initializer=tf2.initializers.GlorotUniform(seed=1))

b1=tf.get_variable('b1',shape=[10,1],dtype=tf.float32,initializer=tf.zeros_initializer())

W2=tf.get_variable('W2',shape=[5,10],dtype=tf.float32,initializer=tf2.initializers.GlorotUniform(seed=1))

b2=tf.get_variable('b2',shape=[5,1],dtype=tf.float32,initializer=tf.zeros_initializer())

W3=tf.get_variable('W3',shape=[1,5],dtype=tf.float32,initializer=tf2.initializers.GlorotUniform(seed=1))

b3=tf.get_variable('b3',shape=[1,1],dtype=tf.float32,initializer=tf.zeros_initializer())





# In[88]:





## Forward prop

Z1=tf.add(tf.matmul(W1,x),b1)

A1=tf.nn.relu(Z1)



Z2=tf.add(tf.matmul(W2,A1),b2)

A2=tf.nn.relu(Z2)



Z3=tf.add(tf.matmul(W3,A2),b3)





# In[89]:





## Compute Cost

cost=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(Z3),labels=tf.transpose(y)))





# In[90]:





## Backward Prop

back_prop=tf.train.GradientDescentOptimizer(learning_rate=.01).minimize(cost)





# In[91]:





with tf.Session() as sess:

    init = tf.global_variables_initializer()

    sess.run(init)

    cost_main=[]

    for i in range(2000):

        _,cost_iter=sess.run([back_prop,cost],feed_dict={x:X_train,y:y_train})

        

        if (i%100==0):

            print('Cost after Iteration %s is %s' %(i,cost_iter))

        cost_main.append(cost_iter)

        

    plt.plot(cost_main)



    correct_prediction=tf.equal(tf.argmax(Z3),tf.argmax(y))

    

    accuracy=tf.reduce_mean(tf.cast(correct_prediction,'float'))



    print('\n')

    print('Train Accuracy is %s' %accuracy.eval({x:X_train,y:y_train}))

    print('Test Accuracy is %s' %accuracy.eval({x:X_test,y:y_test}))