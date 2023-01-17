import numpy as np

import pandas as pd

import tensorflow as tf



from tensorflow.keras import datasets, layers, models

import matplotlib.pyplot as plt

(train_imagesRaw, train_labels), (test_imagesRaw, test_labels) = datasets.cifar10.load_data()



train_images, test_images = train_imagesRaw / 255.0, test_imagesRaw / 255.0
from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)

onehot_encoded = onehot_encoder.fit_transform(train_labels)
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()



x=tf.placeholder(tf.float32,[None,32,32,3])

y=tf.placeholder(tf.int32,[None,10])



inputImg=tf.reshape(x,(-1,32,32,3))

prob = tf.placeholder_with_default(1.0, shape=())

conv1 = tf.layers.conv2d(inputs=inputImg,filters=32,kernel_size=[3, 3],

                         padding="same",activation=tf.nn.relu)



conv2 = tf.layers.conv2d(inputs=conv1,filters=64,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)



## 16x16*64

pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)



conv3 = tf.layers.conv2d(inputs=pool2,filters=64,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)



## 8*8*64

pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)





conv4 = tf.layers.conv2d(inputs=pool3,filters=128,kernel_size=[3, 3],

      padding="same",activation=tf.nn.relu)



## 4*4*128

pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)



pool2_flat = tf.reshape(pool4, [-1, 4 * 4 * 128])

dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)



dropout = tf.layers.dropout(dense, prob)

logits = tf.layers.dense(inputs=dropout, units=10)

pred=tf.nn.softmax(logits)

loss = tf.losses.softmax_cross_entropy(y, logits)

gradientsAttack=tf.gradients(loss,inputImg)



optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

train_op = optimizer.minimize(loss=loss)



init=tf.global_variables_initializer()

sess=tf.Session()



sess.run(init)

batchInt=0

for fulliters in range(20):

    batchInt+=1

    totLoss=0

    for batch in range(500):

        x_train=train_images[batch*100:batch*100+100]

        y_train=onehot_encoded[batch*100:batch*100+100]

        _,currLoss=sess.run([train_op,loss],feed_dict={x:x_train,y:y_train,prob:0.4})

        totLoss+=currLoss

    print('Batch: '+str(batchInt) + ' Loss is: '+ str(totLoss/200))
labelLst=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

for fulliters in range(1):



    x_test=test_images[6]

    classPre=sess.run(pred,feed_dict={x:x_test.reshape(1,32,32,3)})

print(labelLst[list(classPre[0]).index(max(list(classPre[0])))])
for fulliters in range(1):



    x_test=test_images[6]

    y_change=np.array([[0,0,1,0,0,0,0,0,0,0]])

    signGrd=sess.run(gradientsAttack,feed_dict={x:x_test.reshape(1,32,32,3),y:y_change})

signAttack=np.array(signGrd[0])
xNew=test_imagesRaw[6]/255.0+np.sign(signAttack)*0.2
plt.figure(figsize=(1,1))

plt.imshow(xNew[0])


for fulliters in range(1):

    classPre1=sess.run(pred,feed_dict={x:(xNew.reshape(1,32,32,3))/255.0})

print(labelLst[list(classPre1[0]).index(max(list(classPre1[0])))])
from keras.models import Model

from keras import applications

from keras.layers import Input,Flatten,Dense

from keras import backend as K



vgg_model = applications.VGG16(weights='imagenet', include_top=False)



input_tensor = Input(shape=(32, 32, 3))

vgg_model = applications.VGG16(weights='imagenet',

                               include_top=False,

                               input_tensor=input_tensor)



output_vgg16_conv = vgg_model(input_tensor)

x = Flatten(name='flatten')(output_vgg16_conv)

x = Dense(4096, activation='relu', name='fc1')(x)

outputX = Dense(10, activation='softmax', name='predictions')(x)



vgg_model = Model(inputs=input_tensor, outputs=outputX)

vgg_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

vgg_model.fit(train_images, onehot_encoded, epochs=14, batch_size=128,verbose=1)
y_pred=vgg_model.predict(xNew)

y_classes = y_pred.argmax(axis=-1)

labelLst[y_classes[0]]
y_pred=vgg_model.predict(test_images[6].reshape(1,32,32,3))

y_classes = y_pred.argmax(axis=-1)

labelLst[y_classes[0]]
plt.imshow(test_images[11])
gradients = vgg_model.optimizer.get_gradients(vgg_model.total_loss, input_tensor)