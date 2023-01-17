import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from keras.layers import Dense, Dropout

from keras.preprocessing.image import load_img, img_to_array

import matplotlib.pyplot as plt



import os

list_img = []

for dirname, _, filenames in os.walk('/kaggle/input/stanford-dogs-dataset/images/Images'):

    

    for filename in filenames:

        list_img.append(os.path.join(dirname, filename))

        

true_dog=pd.Series(list_img)

print(true_dog.shape)

# Any results you write to the current directory are saved as output.
d=27#image size

def load_preprocess(path, dim=(d, d)):

    """load and preprocess an image"""

    img = load_img(path, target_size=dim)  # Charger l'image 

    img = img_to_array(img)  # Convertir en tableau numpy

    #print(img.shape)

    #img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))  # Créer la collection d'images (un seul échantillon)

    #img = preprocess_input(img)

    return img*1./255

    #return minmax_scale(img)
test_img = true_dog.sample(3).apply(load_preprocess)

imtest = test_img.values

plt.figure(figsize=(17,4))

for i in range(3):

    plt.subplot(1,3,i+1)

    plt.imshow(imtest[i])

plt.show()

"""True image for discriminator"""

img_true = tf.placeholder(tf.float32, shape=(None, d, d, 3))



"""Generator input"""

dr=9

x_rand =tf.placeholder(tf.float32, shape=(None, dr))



"""other parameter"""

is_test = tf.placeholder(tf.bool)

rate = tf.cond(is_test, true_fn=lambda :0.0, false_fn=lambda:0.30)
def batchnorm(Ylogits, Offset, Scale,conv=False):

    

    if conv:

        mean, variance = tf.nn.moments(Ylogits, [0,1,2])

    else:

        mean, variance = tf.nn.moments(Ylogits, [0])

    

    Ybn = tf.nn.batch_normalization(Ylogits,mean,variance, Offset, Scale, variance_epsilon=1e-5)

    return Ybn
def cross_entropy(y_true, y_pred):

    return -tf.reduce_sum((y_true*tf.log(y_pred)+(1-y_true)*tf.log(1-y_pred)))
#Discriminant

#K=22*22*64

K=5*5*124

M=1000

n=1









with tf.name_scope('discrim'):

    """Varible_D"""

    Wconv = tf.Variable(tf.truncated_normal([3,3,3,32],mean=0, stddev=0.1))

    bconv=tf.Variable(tf.zeros([32]))

    Wconv2 = tf.Variable(tf.truncated_normal([3,3,32,64],mean=0, stddev=0.1))

    bconv2 = tf.Variable(tf.zeros([64]))

    Wconv3 = tf.Variable(tf.truncated_normal([3,3,64,64],mean=0, stddev=0.1))

    bconv3 = tf.Variable(tf.zeros([64]))

    Wconv4 = tf.Variable(tf.truncated_normal([3,3,64,124],mean=0, stddev=0.1))

    bconv4 = tf.Variable(tf.zeros([124]))#124



    """Variables FullyConnected"""

    W1=tf.Variable(tf.truncated_normal([K,M], mean=0, stddev=0.1))

    b1=tf.Variable(tf.zeros([M]))



    W2=tf.Variable(tf.truncated_normal([M,M], mean=0, stddev=0.1))

    b2=tf.Variable(tf.zeros([M]))



    W3=tf.Variable(tf.truncated_normal([M,M], mean=0, stddev=0.1))

    b3=tf.Variable(tf.zeros([M]))



    W4=tf.Variable(tf.truncated_normal([M,n],mean=0, stddev=0.1))

    b4=tf.Variable(tf.truncated_normal([n],mean=0, stddev=0.1), name="b4")

    #print(X.shape[0], b2, Wconv, Wconv3)
def discriminant(Img):

    """convolutions :"""

    Xconv_lin=tf.nn.conv2d(Img, filter=Wconv, strides = [1,1],padding='VALID')

    Xconv_lin = batchnorm(Xconv_lin, bconv, 1, conv=True)

    Xconv = tf.nn.relu(Xconv_lin)

    #Xconv = tf.nn.dropout(Xconv, rate=rate)



    Xconv_lin2=tf.nn.conv2d(Xconv, filter=Wconv2, strides = [1,1],padding='SAME')

    Xconv_lin2 = batchnorm(Xconv_lin2, bconv2, 1, conv=True)

    Xconv2 = tf.nn.relu(Xconv_lin2)

    #Xconv2 = tf.nn.dropout(Xconv2, rate=rate)



    Xpool=tf.nn.pool(Xconv2, [5,5], 'MAX', strides=[5,5], padding='SAME')



    Xconv_lin3=tf.nn.conv2d(Xconv2, filter=Wconv3, strides = [1,1],padding='SAME')

    Xconv_lin3 = batchnorm(Xconv_lin3, bconv3, 1, conv=True)

    Xconv3 = tf.nn.relu(Xconv_lin3)

    #Xconv3 = tf.nn.dropout(Xconv3, rate=rate)



    Xconv_lin4=tf.nn.conv2d(Xconv3, filter=Wconv4, strides =1 ,padding='SAME')

    Xconv_lin4 = batchnorm(Xconv_lin4, bconv4, 1, conv=True)

    Xconv4 = tf.nn.relu(Xconv_lin4)

    #Xconv4 = tf.nn.dropout(Xconv4, rate=rate)



    Xconv=tf.nn.pool(Xconv4, [5,5], 'MAX', strides=[5,5], padding='VALID')

    #Xconv=tf.nn.dropout(Xconv, rate=rate)

    print(Xpool, Xconv)

    



    """Fully Connected"""



    X1_lin=tf.matmul(tf.reshape(Xconv, (-1, Xconv.shape[1]*Xconv.shape[2]*Xconv.shape[3])), W1)

    print(X1_lin)

    X1_lin = batchnorm(X1_lin, Offset=b1, Scale=1, conv=False)



    X1=tf.nn.relu(X1_lin)

    X1 = tf.nn.dropout(X1, rate=rate)

    #X1=tf.nn.elu(tf.matmul(tf.reshape(Xconv, (-1, Xconv.shape[1]*Xconv.shape[2]*Xconv.shape[3])), W1)+b1)

    #Y = tf.nn.softmax(tf.matmul(X1, W2)+b2, axis=2)





    X2_lin=tf.matmul(X1, W2)

    X2_lin = batchnorm(X2_lin, Offset=b2, Scale=1, conv=False)

    X2 = tf.nn.relu(X2_lin)

    X2 = tf.nn.dropout(X2, rate=rate)



    X3_lin=tf.matmul(X2, W3)

    X3_lin= batchnorm(X3_lin, Offset=b2, Scale=1, conv=False)

    X3=tf.nn.relu(X3_lin)

    X3 = tf.nn.dropout(X3, rate=rate)





    Y = tf.nn.sigmoid(tf.matmul(X1, W4)+b4)

    return Y


with tf.name_scope("gen"): 

    gen1 = tf.Variable(tf.truncated_normal([dr,9*dr**2], mean=0, stddev=0.1))

    genb1=tf.Variable(tf.zeros([9*dr**2]))

    print(gen1)



    g_conv1 = tf.Variable(tf.truncated_normal([3,3,3, 64], mean=0, stddev=0.1))

    gc1_b = tf.Variable(tf.zeros([64]))

    print(g_conv1)



    g_conv2=tf.Variable(tf.truncated_normal([3,3,64,64], mean=0, stddev=0.1))

    gc2_b = tf.Variable(tf.zeros([64]))

    print(g_conv2)



    g_conv3=tf.Variable(tf.truncated_normal([dr,dr,32,64], mean=0, stddev=0.1))

    gc3_b = tf.Variable(tf.zeros([32]))



    g_conv4=tf.Variable(tf.truncated_normal([3,3,3,32], mean=0, stddev=0.1))

    gc4_b = tf.Variable(tf.zeros([3]))



    g_conv5=tf.Variable(tf.truncated_normal([3,3,3,3], mean=0, stddev=0.1))

    gc5_b = tf.Variable(tf.zeros([3]))

    gc5_a = tf.Variable(tf.truncated_normal([3], mean=1, stddev=0.1))

    
def generator(x_rand=x_rand):

    lin1=tf.matmul(x_rand, gen1)+genb1

    lin1=batchnorm(lin1, Offset=genb1, Scale=1)

    dense1 = tf.nn.relu(lin1)

    

    gen_conv1=tf.nn.conv2d(tf.reshape(dense1, (-1,dr,3*dr,3)),g_conv1, strides=3, padding='SAME')

    gen_conv1=batchnorm(gen_conv1, Offset=gc1_b, Scale=1, conv=True)

    gen_conv1=tf.nn.leaky_relu(gen_conv1)

    #print(gen_conv1)

    bs=tf.shape(gen_conv1)[0]

    #in_ = tf.constant(0.1, shape=[2,9,9,64])

    #in_conv = tf.nn.conv2d(in_, g_conv2, strides=[3,1], padding='SAME')

    #print(in_conv)

    out2=tf.stack([bs, 9,9,64])

    gen_conv2_t=tf.nn.conv2d_transpose(gen_conv1, g_conv2, output_shape=out2, strides=[3,1], padding='SAME')

    gen_conv2=batchnorm(gen_conv2_t, Offset=gc2_b, Scale=1, conv=True)

    gen_conv2=tf.nn.leaky_relu(gen_conv2)

    #print(gen_conv2)



    out3=tf.stack([bs, 27,27,32])

    gen_conv3=tf.nn.conv2d_transpose(gen_conv2, g_conv3, output_shape=out3, strides=3, padding='SAME')#[1,3,3,2]

    gen_conv3=batchnorm(gen_conv3, Offset=gc3_b, Scale=1, conv=True)

    gen_conv3=tf.nn.leaky_relu(gen_conv3)

    #print(gen_conv3)

    out4=tf.stack([bs, 27,27,3])

    gen_conv4=tf.nn.conv2d_transpose(gen_conv3, g_conv4, output_shape=out4, strides=1, padding='SAME')

    gen_conv4=batchnorm(gen_conv4, Offset=gc4_b, Scale=1, conv=True)

    gen_conv4=tf.nn.leaky_relu(gen_conv4)

    

    gen_conv5=tf.nn.conv2d_transpose(gen_conv4, g_conv5, output_shape=(bs,d,d,3), strides=1, padding='SAME')

    gen_conv5=batchnorm(gen_conv5, Offset=gc5_b, Scale=gc5_a, conv=True)

    gen_conv5=tf.nn.sigmoid(gen_conv5)

    

    return gen_conv5
optimizer_gen = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.9, beta2=0.999)

img_fake=generator(x_rand)

Y_fake_pred = discriminant(img_fake)

loss_gen =cross_entropy(tf.ones_like(Y_fake_pred), Y_fake_pred)

train_gen = optimizer_gen.minimize(loss_gen, var_list=tf.trainable_variables(scope='gen'))#



"""Our discriminator target"""

Y_fake = tf.zeros((tf.shape(img_fake)[0], 1))

Y_true = tf.ones((tf.shape(img_true)[0], 1))

"""steps"""

Y_real_pred = discriminant(img_true)

loss_discr = cross_entropy(Y_true, Y_real_pred)+cross_entropy(Y_fake, Y_fake_pred)

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999)

train_step=optimizer.minimize(loss_discr, var_list=tf.trainable_variables(scope='discrim'))





init = tf.global_variables_initializer()
nb_true = 30#per batch

nb_fake = 30



Y_ref = np.vstack((np.ones((nb_true,1)), np.zeros((nb_fake,1))))                

loop=60



init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    #saver.restore(sess, './model_gan2.h5')

    for e in range(loop):

        loss_generator =[]

        loss_discreminant=[]

        for i in range(50):

            true = true_dog.sample(nb_true).apply(load_preprocess)

            X_rand = np.random.random([nb_fake,9])

            #print(X_rand.shape)

            #gen_feed = {x_rand:X_rand, is_test:False}

            #X_fake = sess.run(img_fake, feed_dict=gen_feed)

            #plt.figure()

            #plt.imshow(X_fake[0])

            #plt.show()

            X_true = np.array(list(true))

            feed_dict = {x_rand:X_rand, img_true:X_true, is_test:False}# Y_:Y_ref

            sess.run(train_step, feed_dict=feed_dict)

            loss= sess.run([loss_discr], feed_dict=feed_dict)#gen_conv2_tcross_entropy

            sess.run(train_gen, feed_dict=feed_dict)

            loss_g = sess.run([loss_gen], feed_dict=feed_dict)

            loss_generator.append(loss_g)

            loss_discreminant.append(loss)     

        print('loss_G :',np.mean(loss_generator))

        print('loss_D: ',np.mean(loss_discreminant))

        gen_feed = {x_rand:X_rand, is_test:True}

        X_fake = sess.run(img_fake, feed_dict=gen_feed)

        plt.figure(figsize=(9,9))

        plt.imshow(X_fake[0])

        plt.axis('off')

        plt.show()

        if np.isnan(loss):

            break