#import libraries



import numpy as np

import tensorflow as tf

import imageio

import glob

from IPython.display import Image, display,FileLink,clear_output



import os

from scipy import misc, ndimage, stats

import random

#base settings



batch_size = 16 #how many images to run each iteration



zN = 128 #length of latent vector



sideX = 256 #sidelength in x dimension (numpy switches these around) (low res for testing)

sideY = 256
#utils/layers

#spec norm from Junho Kim https://github.com/taki0112/Spectral_Normalization-Tensorflow

def spectral_norm(w, iteration=1):

   w_shape = w.shape.as_list()

   w = tf.reshape(w, [-1, w_shape[-1]])



   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)



   u_hat = u

   v_hat = None

   for i in range(iteration):

       """

       power iteration

       Usually iteration = 1 will be enough

       """

       v_ = tf.matmul(u_hat, tf.transpose(w))

       v_hat = tf.nn.l2_normalize(v_)



       u_ = tf.matmul(v_hat, w)

       u_hat = tf.nn.l2_normalize(u_)



   u_hat = tf.stop_gradient(u_hat)

   v_hat = tf.stop_gradient(v_hat)



   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))



   with tf.control_dependencies([u.assign(u_hat)]):

       w_norm = w / sigma

       w_norm = tf.reshape(w_norm, w_shape)





   return w_norm





def upsamp(x,sc=2):

  x = tf.image.resize_image_with_pad(x,sc*x.get_shape().as_list()[1],sc*x.get_shape().as_list()[2])

#   x = conv2D(x,x.shape[-1],1,1)

#   x = tf.keras.layers.Conv2DTranspose(x.get_shape().as_list()[-1],4,2,padding='SAME')(x)

  return x



def conv2D(x,c,kd=3,strides=1,fin=False,spN = True):

  with tf.variable_scope(None,'c'):

    inFil = (x.shape[1]//strides,x.shape[2]//strides)

    

    

    w = tf.get_variable('kernel',[kd,kd,x.get_shape()[-1],c])

    if spN:

        w = spectral_norm(w)

    

    b = tf.get_variable('bias',[c],initializer=tf.constant_initializer(0.0))

    x = tf.nn.conv2d(input=x,filter=w,strides=[1,strides,strides,1],padding='SAME')+b

#     dif = ((inFil[0]-x.shape[1])//2,(inFil[1]-x.shape[2])//2) #set padding to valid

#     pd = [[0,0],[dif[0],dif[0]],[dif[1],dif[1]],[0,0]]

    

#     x = tf.pad(x,pd,'REFLECT')

    

    if fin:

      x = tf.tanh(x)

      print(x.shape)

    return x



def dense(x,units,spN=True):

  with tf.variable_scope(None,'d'):



#     fl = tf.keras.layers.Flatten()(x)

    ind = x.get_shape()[-1]

    

    w = tf.get_variable('weights',shape=[ind,units])

    if spN:

        w = spectral_norm(w)

    b = tf.get_variable('bias',[units],initializer=tf.constant_initializer(0.0))

    return tf.matmul(x,w)+b

                     

def relu(x):

  return tf.maximum(x,tf.zeros_like(x))



def leaky(x):

  return tf.maximum(x,.2*x)



def batchN(x):

    x = tf.contrib.layers.batch_norm(x,renorm=True)

    return x



def iN(x):

    return tf.contrib.layers.instance_norm(x)







def block(x,filters,k='l',up = False, down = False): 

  with tf.variable_scope(None,'resn'):

    inFil = x.shape.as_list()[-1]

    cur = filters//4

    

    if not down:

      a = batchN(x)

      a = leaky(a)

      

      a = conv2D(a,cur,1,1)

        

      a = batchN(a)

      a = leaky(a)

    

      a = conv2D(a,cur)

        

      a = batchN(a)

      a = leaky(a)

    

    else:

      a = leaky(x)

      a = conv2D(a,cur,1,1)

      a = leaky(a)

      a = conv2D(a,cur)

      a = leaky(a)

      

    

    if up:

      a = upsamp(a)

      x = x[:,:,:,:filters]

      x = upsamp(x)

      a = conv2D(a,filters,1,1)

    elif down:

#       pd = (filters-inFil)//2

#       x = tf.pad(x,[[0,0],[0,0],[0,0],[pd,pd]])

      x = conv2D(x,filters,2,2)

      a = conv2D(a,filters,2,2)

#       a = tf.nn.avg_pool(a,[1,2,2,1],[1,2,2,1],'SAME')



#       x = tf.nn.avg_pool(x,[1,2,2,1],[1,2,2,1],'SAME')

    else:

      a = conv2D(a,filters,1,1)

    a = tf.add(x,a)

    return a



def lN(x):

    return tf.contrib.layers.layer_norm(x)



def stdL(x,groups=4,features=1): #hardcoded?

    sh = x.shape

    y = tf.reshape(x,(groups,features,-1,x.shape[1],x.shape[2],x.shape[3]))

    y = tf.sqrt(tf.nn.moments(y,[0])[1])

    y = tf.reduce_mean(y,[2,3,4],keep_dims=True)

    y = tf.reduce_mean(y,[2])

    y = tf.reshape(y,(-1,1,1,1))

    y = tf.tile(y,(groups,x.shape[1],x.shape[2],features))

    y = tf.concat([x,y],axis=-1)

    return x
#setup data functions

def load():

    paths = []

    for filepath in glob.iglob('/kaggle/input/psychart/' + '**/*.jpg', recursive=True):

        try:

#             misc.imresize(filepath,(256,256))

            paths.append(filepath)

        except:

            print(filepath,'broke')

            

    tf.convert_to_tensor(paths, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices((paths))

    dataset = dataset.repeat().shuffle(len(paths))

    return dataset

def maper(path):

    image = tf.image.decode_jpeg(tf.read_file(path))

    

    #augment images

    

#     image = tf.image.random_crop(image,[sideY,sideX,3])

    #instead of cropping to a random x by y location, do the whole thing with a random up-to 20th shaved off each side

    rnds = tf.random.uniform([4],0,.1)

    image = tf.image.crop_and_resize(tf.expand_dims(image,axis=0),[[rnds[0],rnds[1],

                                             1-rnds[2],1-rnds[3]]]

                                     ,[0],(sideY,sideX))

    image = tf.squeeze(image)

#     image = tf.image.rot90(image,tf.random.uniform([],0,4,dtype=tf.int32))

    image = tf.image.random_hue(image,.5)

    image = tf.image.random_saturation(image,.9,2)

    image = tf.image.random_brightness(image,.8)



    

#     image = tf.image.resize_images(image,(sideY,sideX))

    image = (tf.cast(image,tf.float32)/255)*2-1 #normalize data

    

    return image    



#setup model



tf.reset_default_graph()#run this to restart



def generator(latent):#moves from noise (a latent vector) to images

    with tf.variable_scope('gen'):

        dep = 4

        x = dense(latent,sideY//2**dep*sideX//2**dep*512)

        x = tf.reshape(x,(-1,sideY//2**dep,sideX//2**dep,512))

        ch = 256

        for i in range(dep):

            x = upsamp(x)

            x = conv2D(x,ch)

            x = leaky(x)

            x = batchN(x) #renorm is on if batchN

            ch//=2

        x = conv2D(x,3,fin=True)

        return x



def discriminator(x): #tell generator how far it is from real images

    with tf.variable_scope('rdis',reuse=tf.AUTO_REUSE):

        ch = 64

        x = tf.reshape(x,(-1,sideY,sideX,3*2)) # pacgan

        for i in range(4):

            x = conv2D(x,ch,4,2)

            x = leaky(x)

            ch*=2

            if i == 3:

                x = stdL(x)

        x = tf.reshape(x,(batch_size,-1))

        x = dense(x,1)

        return x

    

latent = tf.placeholder(dtype=tf.float32,shape=(None,zN))

syn = generator(latent)



dataset = load()

dataset = dataset.map(maper, num_parallel_calls=4)

dataset = dataset.batch(batch_size)

dataset = dataset.prefetch(1)



img = dataset.make_one_shot_iterator().get_next()

img.set_shape((batch_size,sideY,sideX,3))





dreal = discriminator(img)

dfake = discriminator(syn)



tr = tf.trainable_variables()

genv = [var for var in tr if 'gen' in var.name]



discv = [var for var in tr if 'rdis' in var.name]



#hinge loss

d_loss = tf.reduce_mean(tf.maximum(0.,1.-dreal)) + tf.reduce_mean(tf.maximum(0.,1.+dfake))

g_loss = tf.reduce_mean(-dfake)





gopt = tf.train.AdamOptimizer(.0002,.0,.9).minimize(g_loss,var_list=genv)

  

dopt = tf.train.AdamOptimizer(.0002,.0,.9).minimize(d_loss,var_list=discv)



config = tf.ConfigProto()

config.gpu_options.allow_growth = True



sess = tf.Session(config=config)



sess.run(tf.global_variables_initializer())





its = 0

print('Ready')

#train model



#for comitting just change st in gen_batch to its and do the whole batch



#for watching

def gen_batch(st=3,rows=True,singles=False):

    out = syn.eval({latent:np.random.normal(size=(batch_size,zN))})

    cn=0

    if not rows:

        bimg = np.zeros((3*sideY,3*sideX,3)) #only using part of the full 20-image batches

        for x in range(3):

            for y in range(3):

                bimg[x*sideY:(x+1)*sideY,y*sideX:(y+1)*sideX,:] = out[cn]

                cn+=1

        imageio.imwrite(str(st)+'.jpg', bimg)

        display(Image(str(st)+'.jpg'))

    elif singles:

        for ims in range(3):

            imageio.imwrite(str(st)+'.jpg', out[ims])

            display(Image(str(st)+'.jpg'))

            

    else:

        bimg = np.zeros((1*sideY,5*sideX,3)) #should use all of batches,b ut idc rn

        for x in range(1):

            for y in range(5):

                bimg[x*sideY:(x+1)*sideY,y*sideX:(y+1)*sideX,:] = out[cn]

                cn+=1

        imageio.imwrite(str(st)+'.jpg', bimg)

        display(Image(str(st)+'.jpg'))

def train():

  global its

  with sess.as_default():

    for i in range(10000):

      

      for n in range(1):

        dopt.run({latent:np.random.normal(size=(batch_size,zN))})

      

      

      gopt.run({latent:np.random.normal(size=(batch_size,zN))})

      its+=1

      

      if its%100 == 0 or i == 0:

        done = np.resize(img.eval()[0],(sideX,sideY,3))

        imageio.imwrite('3.jpg', done)

        display(Image('3.jpg'))

        

        for j in range(1):

            gen_batch(st=its)

        

        de = d_loss.eval({latent:np.random.normal(size=(batch_size,zN))})

        ge = g_loss.eval({latent:np.random.normal(size=(batch_size,zN))})

        print('\nG',ge,'D',de,'at',its,'\n')      

       

with sess.as_default():

    train()



#get to at least 2000

saver = tf.train.Saver()

saver.save(sess, './model.ckpt')
#generate

with sess.as_default():

    for i in range(2):

        gen_batch(rows=False,singles=True)
#interpolate (may want to implement a circle about the distribution vs cutting through)



#batch_size may screw with this...



def interp(shape,steps,trunc=False):

    if trunc:

        start = stats.truncnorm.rvs(-1.4,1.4,size=shape)

    else:

        start = np.random.normal(size=shape)

    



    stepA = np.zeros((start.shape))

    for i in range(points-1):

      stepA[i] = np.subtract(start[i+1],start[i])/steps

    stepA[points-1] = np.subtract(start[0],start[points-1])/steps



    ind = 0

    exampIn = np.zeros(([shape[0]*steps]+list(shape[1:])))



    for qt in range(points):

      for zt in range(steps):

        exampIn[ind] = start[qt]+stepA[qt]*(zt+1)

        ind+=1

    return exampIn





with sess.as_default(): #does order not hold over multiple bathes for some reason?

    steps = 9

    points = 5



#     np.random.seed(87)



    lat = interp((points,zN),steps)

    

    fed = {latent:lat}

    out = syn.eval(fed)

    print(out.shape)

    out = np.reshape(out,(points*steps,sideY,sideX,3))

    images = []

    for im in out:

        images.append(im)

    

    imageio.mimsave('g.gif', images)

    display(Image('g.gif'))
FileLink('g.gif')