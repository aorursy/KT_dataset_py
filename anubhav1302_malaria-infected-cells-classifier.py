#Malaria Cells Classifier

#Custom Model

#No data augmentation done here except image reshaping since images were of varying shapes

#Work in Progress
import os

import cv2

import pandas as pd

import seaborn as sns

from glob import glob

from tqdm import tqdm

import numpy as np

from skimage import io

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf

from sklearn.metrics import confusion_matrix
#Path

img_path='../input/cell_images/cell_images'
target=[]

images=[]

for class_ in os.listdir(img_path):

    files=glob(os.path.join(img_path,class_,'*.png'))

    for file in tqdm(files):

        img=cv2.imread(file)

        img=cv2.normalize(img,None,0,1,cv2.NORM_MINMAX,dtype=cv2.CV_32F)

        img=cv2.resize(img,(150,150),cv2.INTER_AREA)

        images.append(img)

        if class_=='Uninfected':

            target.append(0)

        else:

            target.append(1)

target=np.reshape(target,(27558,1))
sns.distplot(target)
train_images,val_images,train_labels,val_labels=train_test_split(images,target,test_size=0.2)
print('Size of Train Data: {}'.format(len(train_images)))

print('Size of Val Data: {}'.format(len(val_images)))
#Placeholders

inp=tf.placeholder(tf.float32,[None,150,150,3],name='Input_image')

label=tf.placeholder(tf.float32,[None,1],'Input_label')
#Layer Functions

def Conv2D(input_,filters,kernel_size,strides,kernel_initializer='he_normal',padding='same'):

    layer=tf.layers.conv2d(input_,filters=filters,kernel_size=kernel_size,strides=strides,kernel_initializer=kernel_initializer,

                          padding=padding)

    return layer

def LeakyReLU(x,t=0.2):

    return tf.maximum(x,x*t)

def BatchNormalization(input_):

    return tf.layers.batch_normalization(input_)

def MaxPooling2D(input_,pool_size,strides):

    return tf.layers.max_pooling2d(input_,pool_size=pool_size,strides=strides)
#Branch-1

with tf.name_scope('Branch-1'):

    b1=Conv2D(inp,16,kernel_size=4,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b1=LeakyReLU(b1)

    b1=Conv2D(b1,16,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b1=Conv2D(b1,16,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b1=BatchNormalization(b1)

    b1_mp=MaxPooling2D(b1,pool_size=4,strides=4)



#Branch-2

with tf.name_scope('Branch-2'):

    b2=Conv2D(inp,32,kernel_size=4,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b2=LeakyReLU(b2)

    b2=Conv2D(b2,32,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b2=Conv2D(b2,32,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b2_mp=MaxPooling2D(b2,pool_size=4,strides=4)

    b2_mp=BatchNormalization(b2_mp)



#Branch-3

with tf.name_scope('Branch-3'):

    b3=Conv2D(inp,64,kernel_size=4,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b3=LeakyReLU(b3)

    b3=Conv2D(b3,64,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b3=Conv2D(b3,64,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Valid')

    b3_mp=MaxPooling2D(b3,pool_size=4,strides=4)

    b3_mp=BatchNormalization(b3_mp)





#Branch_connections

with tf.name_scope('Branch-Conv'):

    b1_con=Conv2D(b1_mp,32,kernel_size=3,strides=1,kernel_initializer='he_uniform',padding='Same')

    b3=LeakyReLU(b3)

with tf.name_scope('Branch-Connect'):

    con_1=tf.keras.layers.concatenate([b1_con,b2_mp])

    con_2=tf.keras.layers.concatenate([con_1,b3_mp])



with tf.name_scope('Final_Block'):

    layer_2=Conv2D(con_2,128,kernel_initializer='he_uniform',strides=1,padding='valid',kernel_size=1)

    layer_2=LeakyReLU(layer_2)

    layer_2_mp=MaxPooling2D(layer_2,pool_size=4,strides=4)



    layer_3=Conv2D(layer_2,256,kernel_initializer='he_uniform',strides=1,padding='valid',kernel_size=3)

    layer_3=LeakyReLU(layer_3)

    layer_3_mp=MaxPooling2D(layer_3,pool_size=4,strides=4)

    layer_3_mp=BatchNormalization(layer_3_mp)





    layer_4=Conv2D(layer_3,512,kernel_initializer='he_uniform',strides=1,padding='valid',kernel_size=3)

    layer_4=LeakyReLU(layer_4)

    layer_4_mp=MaxPooling2D(layer_4,pool_size=4,strides=4)

    layer_4_mp=BatchNormalization(layer_4_mp)



    layer_5=Conv2D(layer_4,1024,kernel_initializer='he_uniform',strides=1,padding='valid',kernel_size=3)

    layer_5=LeakyReLU(layer_5)

    layer_5_mp=MaxPooling2D(layer_5,pool_size=6,strides=6)

    layer_5_mp=BatchNormalization(layer_5_mp)



with tf.name_scope('Flatten'):

    flat=tf.contrib.layers.flatten(layer_5_mp)

    dense_1=tf.layers.dense(flat,2048,activation='relu',kernel_initializer='he_normal')

with tf.name_scope('Output_Units'):

    out=tf.layers.dense(dense_1,1,activation=None)

    prob=tf.nn.sigmoid(out)
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label,logits=out))

train_opt=tf.train.AdamOptimizer(0.0001).minimize(loss)

acc = tf.metrics.auc(labels=label,predictions=prob)
batch_size=25

train_batches=len(train_images)//batch_size

val_batches=len(val_images)//batch_size

epochs=30

train_loss,val_loss=[],[]

train_acc,val_acc=[],[]

print('Train Batches {}...Val Batches {}'.format(train_batches+1,val_batches+1))

saver=tf.train.Saver()
with tf.Session() as sess:

    writer = tf.summary.FileWriter("model_tensorboard", sess.graph)

    sess.run(tf.global_variables_initializer())

    sess.run(tf.local_variables_initializer())

    for epoch in range(epochs):

        #Training

        print('Epoch {} of {}'.format(epoch+1,epochs))

        for batch in tqdm(range(train_batches+1)):

            if batch==train_batches:

                t_img=train_images[(batch)*batch_size:]

                t_img=np.reshape(t_img,(len(t_img),150,150,3))

                t_label=train_labels[(batch)*batch_size:]

            else:

                t_img=train_images[batch*batch_size:(batch+1)*batch_size]

                t_img=np.reshape(t_img,(len(t_img),150,150,3))

                t_label=train_labels[batch*batch_size:(batch+1)*batch_size]

            t_loss,t_acc,_=sess.run([loss,acc,train_opt],feed_dict={inp:t_img,label:t_label})

        train_loss.append(t_loss)

        train_acc.append(t_acc)

        #Validation

        for batch in tqdm(range(val_batches+1)):

            if batch==val_batches:

                v_img=val_images[(batch)*batch_size:]

                v_img=np.reshape(v_img,(len(v_img),150,150,3))

                v_label=val_labels[(batch)*batch_size:]

            else:

                v_img=val_images[batch*batch_size:(batch+1)*batch_size]

                v_img=np.reshape(v_img,(len(v_img),150,150,3))

                v_label=val_labels[batch*batch_size:(batch+1)*batch_size]

            v_loss,v_acc=sess.run([loss,acc],feed_dict={inp:v_img,label:v_label})

        val_loss.append(v_loss)

        val_acc.append(v_acc)

        print('Training Loss: {}...Training Acc: {} '.format(t_loss,t_acc[0]))

        print('Validation Loss: {}...Validation Acc: {}'.format(v_loss,v_acc[0]))

        print('****************************************')    

    writer.close()

    saver.save(sess,'model_classifier.ckpt')

    print('****************END********************')  
#Restore model and make predictions of validation set

preds=[]

with tf.Session() as sess:

    saver.restore(sess,'model_classifier.ckpt')

    print('Classifier Restored')

    for batch in tqdm(range(val_batches+1)):

        if batch==val_batches:

            v_img=val_images[(batch)*batch_size:]

            v_img=np.reshape(v_img,(len(v_img),150,150,3))

        else:

            v_img=val_images[batch*batch_size:(batch+1)*batch_size]

            v_img=np.reshape(v_img,(len(v_img),150,150,3))

        p=sess.run(prob,feed_dict={inp:v_img})

        preds.extend(p)

    preds=['Parasitized' if i>=0.5 else 'Uninfected' for i in preds]

    l=['Parasitized' if i>=0.5 else 'Uninfected' for i in val_labels]

    #Plot predictions

    random_images=np.reshape(val_images[20:24],(4,150,150,3))

    random_preds=preds[20:24]

    random_labels=l[20:24]

    w,h=150,150

    fig=plt.figure(figsize=(8,8))

    columns=2

    rows=2

    for i in range(1,columns*rows+1):

        fig.add_subplot(rows,columns,i)

        img=io.imshow(random_images[i-1])

        plt.title(['True:{}|| Predicted:{}'.format(random_labels[i-1],random_preds[i-1])])

    plt.show()
#Confusion Matrix

cm=confusion_matrix(y_true=l,y_pred=preds)

con_mat=pd.DataFrame(cm,index=['Uninfected','Parasitized'],columns=['Uninfected','Parasitized'])

sns.heatmap(con_mat,annot=True)
con_mat
loss=train_loss

val_loss=val_loss

t_auc=[i[0] for i in train_acc]

v_auc=[i[0] for i in val_acc]

epochs=range(1,len(loss)+1)

plt.plot(epochs,loss,'b',color='red',label='Training Loss')

plt.plot(epochs,val_loss,'b',color='blue',label='Validation Loss')

plt.title('Training and Validation Loss')

plt.legend()

plt.figure()

plt.plot(epochs,t_auc,'b',color='red',label='Training AUC')

plt.plot(epochs,v_auc,'b',color='blue',label='Validation AUC')

plt.title('Training and Validation Loss')

plt.legend()

plt.figure()

plt.show()