import numpy as np, pandas as pd, os, time, gc

import xml.etree.ElementTree as ET , time

import matplotlib.pyplot as plt, zipfile 

from PIL import Image 



# STOP KERNEL IF IT RUNS FOR MORE THAN 8 HOURS

kernel_start = time.time()

LIMIT = 8



# PREPROCESS IMAGES

ComputeLB = True

DogsOnly = True

ROOT = '../input/generative-dog-images/'

if not ComputeLB: ROOT = '../input/'

IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')

breeds = os.listdir(ROOT + 'annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,80,80,3))



# CROP WITH BOUNDING BOXES TO GET DOGS ONLY

# https://www.kaggle.com/paulorzp/show-annotations-and-breeds

if DogsOnly:

    for breed in breeds:

        for dog in os.listdir(ROOT+'annotation/Annotation/'+breed):

            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

            except: continue           

            ww,hh = img.size

            tree = ET.parse(ROOT+'annotation/Annotation/'+breed+'/'+dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                w = np.min((xmax - xmin, ymax - ymin))

                # ADD PADDING TO CROPS

                EXTRA = w//8

                a1 = EXTRA; a2 = EXTRA; b1 = EXTRA; b2 = EXTRA

                a1 = np.min((a1,xmin)); a2 = np.min((a2,ww-xmin-w))

                b1 = np.min((b1,ymin)); b2 = np.min((b2,hh-ymin-w))

                img2 = img.crop((xmin-a1, ymin-b1, xmin+w+a2, ymin+w+b2))

                img2 = img2.resize((80,80), Image.ANTIALIAS)

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                namesIn.append(breed)                

                #if idxIn%1000==0: print(idxIn)

                idxIn += 1

                

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    namesIn = np.array(namesIn)[idx]

    

# RANDOMLY CROP FULL IMAGES

else:

    for k in range(len(IMAGES)):

        img = Image.open(ROOT + 'all-dogs/all-dogs/' + IMAGES[k])

        w = img.size[0]

        h = img.size[1]

        sz = np.min((w,h))

        a=0; b=0

        if w<h: b = (h-sz)//2

        else: a = (w-sz)//2

        img = img.crop((0+a, 0+b, sz+a, sz+b))  

        img = img.resize((64,64), Image.ANTIALIAS)   

        imagesIn[idxIn,:,:,:] = np.asarray(img)

        namesIn.append(IMAGES[k])               

        #if (idxIn%1000==0): print(idxIn)

        idxIn += 1 

    

# DISPLAY CROPPED IMAGES

x = np.random.randint(0,idxIn,25)

for k in range(3):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )

        plt.axis('off')

        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)

        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)

        plt.imshow(img)

    plt.show()
import tensorflow as tf

tf.enable_eager_execution()



# FUNCTION FOR DATA AUGMENTATION

def flip(x: tf.Tensor, y:tf.Tensor) -> (tf.Tensor,tf.Tensor):

    x = tf.image.random_flip_left_right(x)

    return (x,y)



# FUNCTION FOR DATA AUGMENTATION

def crop(x: tf.Tensor, y:tf.Tensor) -> (tf.Tensor,tf.Tensor):

    x = tf.random_crop(x,size=[64,64,3])

    return (x,y)
BATCH_SIZE = 32



from sklearn import preprocessing



for i in range(len(namesIn)):

    namesIn[i] = namesIn[i].split('-')[1].lower()

le = preprocessing.LabelEncoder()

namesIn = le.fit_transform(namesIn)



imagesIn = (imagesIn[:idxIn,:,:,:]-127.5)/127.5

namesIn = namesIn[:idxIn]

imagesIn = imagesIn.astype('float32')

namesIn = namesIn.astype('int8')

ds = tf.data.Dataset.from_tensor_slices((imagesIn,namesIn)).map(flip).map(crop).batch(BATCH_SIZE,drop_remainder=True)

print('TF Version',tf.__version__); print()

print('Our TF data pipeline has been built')

print(ds)
MAPS = 128

noise_dim = 128



from tensorflow.keras import layers

from tensorflow.keras.initializers import RandomNormal

init = RandomNormal(mean=0.0, stddev=0.02)



def make_generator():

    seed = tf.keras.Input(shape=((noise_dim,)))

    label = tf.keras.Input(shape=((1,)))

    x = layers.Embedding(120, 120, input_length=1,name='emb')(label)

    x = layers.Flatten()(x)

    x = layers.concatenate([seed,x])

    x = layers.Dense(4*4*MAPS*8, use_bias=False)(x)

    x = layers.Reshape((4, 4, MAPS*8))(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(MAPS*4, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(MAPS*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(MAPS, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    x = layers.ReLU()(x)

    

    x = layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False, activation='tanh')(x)



    model = tf.keras.Model(inputs=[seed,label], outputs=x)    

    return model



generator = make_generator()

#generator.summary()
GNOISE = 0.25



def make_discriminator():

    image = tf.keras.Input(shape=((64,64,3)))

    label = tf.keras.Input(shape=((1,)))

    x = layers.Embedding(120, 64*64, input_length=1)(label)

    x = layers.Reshape((64,64,1))(x)

    x = layers.concatenate([image,x])

    

    x = layers.Conv2D(MAPS, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    #x = layers.GaussianNoise(GNOISE)(x)

    x = layers.LeakyReLU()(x)



    x = layers.Conv2D(MAPS*2, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    #x = layers.GaussianNoise(GNOISE)(x)

    x = layers.LeakyReLU()(x)



    x = layers.Conv2D(MAPS*4, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    #x = layers.GaussianNoise(GNOISE)(x)

    x = layers.LeakyReLU()(x)



    x = layers.Conv2D(MAPS*8, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init, use_bias=False)(x)

    x = layers.BatchNormalization()(x)

    #x = layers.GaussianNoise(GNOISE)(x)

    x = layers.LeakyReLU()(x)

    

    x = layers.Flatten()(x)

    x = layers.Dense(121, activation='sigmoid')(x)

    x2 = layers.Dense(1, activation='linear')(x)

    

    model = tf.keras.Model(inputs=[image,label], outputs=[x,x2])

    return model



discriminator = make_discriminator()

#discriminator.summary()
# THESE LOSS FUNCTIONS ARE UNUSED. THEY ARE REWRITTEN INSIDE TRAINING LOOP BELOW

from tensorflow.contrib.eager.python import tfe



# RaLS Discriminator Loss

def RaLS_errD(fake,real):

    return (tf.reduce_mean( (real - tf.reduce_mean(fake,0) - tf.ones_like(real))**2,0 )

        + tf.reduce_mean( (fake - tf.reduce_mean(real,0) + tf.ones_like(real))**2,0 ) )/2.



# RaLS Generator Loss

def RaLS_errG(fake,real):

    return (tf.reduce_mean( (real - tf.reduce_mean(fake,0) + tf.ones_like(real))**2,0 )

        + tf.reduce_mean( (fake - tf.reduce_mean(real,0) - tf.ones_like(real))**2,0 ) )/2.



# OPTIMIZER - ADAM

learning_rate = tfe.Variable(0.0002)

generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)

discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5)
DISPLAY_EVERY = 10



def display_images(model, test_input, labs):

    predictions = model([test_input,labs], training=False)

    fig = plt.figure(figsize=(16,4))

    for i in range(predictions.shape[0]):

        plt.subplot(2, 8, i+1)

        plt.imshow( (predictions[i, :, :, :]+1.)/2. )

        plt.axis('off')

    plt.show()

    

def generate_latent_points(latent_dim, n_samples):

    return tf.random.truncated_normal((n_samples,latent_dim))



def train(dataset, epochs):

    all_gl = np.array([]); all_dl = np.array([])

    

    for epoch in range(epochs):

        start = time.time()

        gl = []; dl = []

           

        # TENSOR FLOW DATA.DATASET HAS A BUG AND WONT SHUFFLE ON ITS OWN :-(

        # https://github.com/tensorflow/tensorflow/issues/27680

        idx = np.arange(idxIn)

        np.random.shuffle(idx)

        dataset = (tf.data.Dataset.from_tensor_slices((imagesIn[idx,:,:,:],namesIn[idx]))

            .map(flip).map(crop).batch(BATCH_SIZE,drop_remainder=True))

        

        # TRAIN ACGAN

        for i,image_batch in enumerate(dataset):

            gg,dd = train_step(image_batch,generator,discriminator,

                        generator_optimizer, discriminator_optimizer)

            gl.append(gg); dl.append(dd)

        all_gl = np.append(all_gl,np.array([gl]))

        all_dl = np.append(all_dl,np.array([dl]))

        

        # EXPONENTIALLY DECAY LEARNING RATES

        if epoch>180: learning_rate.assign(learning_rate*0.95)

        

        # DISPLAY PROGRESS

        if epoch%DISPLAY_EVERY==0:

            # PLOT EPOCH LOSS

            plt.figure(figsize=(16,2))

            plt.plot(np.arange(len(gl)),gl,label='Gen_loss')

            plt.plot(np.arange(len(dl)),dl,label='Disc_loss')

            plt.legend()

            plt.title('Epoch '+str(epoch)+' Loss')

            ymax = plt.ylim()[1]

            plt.show()

            

            # PLOT ALL TIME LOSS

            plt.figure(figsize=(16,2))

            plt.plot(np.arange(len(all_gl)),all_gl,label='Gen_loss')

            plt.plot(np.arange(len(all_dl)),all_dl,label='Disc_loss')

            plt.legend()

            plt.ylim((0,np.min([1.1*np.max(all_gl),2*ymax])))

            plt.title('All Time Loss')

            plt.show()



            # DISPLAY IMAGES FROM TRAIN PROGRESS

            seed = generate_latent_points(noise_dim, num_examples)

            labs = tf.cast(120*tf.random.uniform((num_examples,1)),tf.int8)

            display_images(generator, seed, labs)

            

            # PRINT STATS

            print('EPOCH',epoch,'took',np.round(time.time()-start,1),'sec')

            print('Gen_loss mean=',np.mean(gl),'std=',np.std(gl))

            print('Disc_loss mean=',np.mean(dl),'std=',np.std(dl))

            print('Learning rate = ',end='')

            tf.print(discriminator_optimizer._lr)

            

        x = gc.collect()

        tt = np.round( (time.time() - kernel_start)/60,1 )

        if tt > LIMIT*60: break
EPOCHS = 250

num_examples = 16



@ tf.function

def train_step(images,generator,discriminator,generator_optimizer,discriminator_optimizer):

        

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True,label_smoothing=0.4)

    bce2 = tf.keras.losses.BinaryCrossentropy(from_logits=False,label_smoothing=0.4)

    noise = tf.random.normal((32,128)) # update noise_dim here

    labs = tf.cast(120*tf.random.uniform((32,)),tf.int32)

    

    # USE GRADIENT TAPE TO CALCULATE GRADIENTS

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:       

        generated_images = generator([noise,labs], training=True)

        real_cat, real_output = discriminator([images[0],images[1]], training=True)

        fake_cat, fake_output = discriminator([generated_images,labs], training=True)

    

        # GENERATOR LOSS 

        gen_loss = (tf.reduce_mean( (real_output - tf.reduce_mean(fake_output,0) + tf.ones_like(real_output))**2,0 )

        + tf.reduce_mean( (fake_output - tf.reduce_mean(real_output,0) - tf.ones_like(real_output))**2,0 ) )/2.

        

        # DISCRIMINATOR LOSS

        disc_loss = bce(tf.ones_like(real_output), real_output) + bce(tf.zeros_like(fake_output), fake_output)           

        real_cat2 = tf.one_hot(tf.cast(images[1],tf.int32),121,dtype=tf.int32)

        fake_cat2 = tf.one_hot(120*tf.ones((32,),tf.int32),121,dtype=tf.int32)

        disc_loss += bce2(real_cat2,real_cat) + bce2(fake_cat2,fake_cat) 

        

    # BACK PROPAGATE ERROR

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)

    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))

    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

       

    return gen_loss, disc_loss



print('Training started. Displaying every '+str(DISPLAY_EVERY)+'th epoch.')

train(ds, EPOCHS)
mse = tf.keras.losses.MeanSquaredError()



print('Display Random Dogs by Breed')

print()

for j in np.random.randint(0,120,25):

    # GENERATE DOGS

    seed = generate_latent_points(noise_dim, 10)

    labs = tf.cast( j*np.ones((10,1)), tf.int8)

    predictions = generator([seed,labs], training=False); d = 0   

    # GET BREED NAME    

    br = np.argwhere( namesIn==j ).flatten()

    bd = le.inverse_transform(np.array([j]))[0].capitalize()

    # CALCULATE VARIETY

    for k in range(4): d += mse(predictions[k,:,:,:],predictions[k+1,:,:,:]) 

    d = np.round( np.array(d),1 )

    if d<1.0: 

        print(bd,'had mode collapse. No display. (variety =',d,')')

        continue

    # DISPLAY DOGS

    print(bd,'REAL DOGS on top. FAKE DOGS on bottom. (variety =',d,')')

    plt.figure(figsize=(15,9))

    for i in range(5):

        plt.subplot(3,5,i+1)

        plt.imshow( (imagesIn[br[i],:,:,:]+1.)/2. )

        plt.axis('off')

    for i in range(10):

        plt.subplot(3,5,i+6)

        plt.imshow( (predictions[i,:,:,:]+1.)/2. )

        plt.axis('off')

    plt.show()
seed = generate_latent_points(noise_dim, 100)

labs = tf.cast(120*tf.random.uniform((100,1)),tf.int8)

predictions = generator([seed,labs], training=False)

plt.figure(figsize=(20,20))

plt.subplots_adjust(wspace=0,hspace=0)

for k in range(100):

    plt.subplot(10,10,k+1)

    plt.imshow( (predictions[k,:,:,:]+1.)/2. )

    plt.axis('off')

plt.show()
# SUBMIT 84 IMAGES OF EACH OF 119 BREEDS and 4 of breed 120

z = zipfile.PyZipFile('images.zip', mode='w')

for i in range(120):

    ct = 84

    if i==119: ct=4

    seed = generate_latent_points(noise_dim, ct)

    labs = tf.cast( i*np.ones((ct,1)), tf.int8)

    predictions = generator([seed,labs], training=False)

    predictions = 255*((np.array(predictions)+1.)/2.)

    for j in range(ct):

        img = Image.fromarray( predictions[j,:,:,:].astype('uint8') )

        f = str(i*84+j)+'.png'

        img.save(f,'PNG'); z.write(f); os.remove(f)

    #if (i%10==0)|(i==119): print(i*84)

z.close()
from __future__ import absolute_import, division, print_function

import numpy as np

import os

import gzip, pickle

import tensorflow as tf

from scipy import linalg

import pathlib

import urllib

import warnings

from tqdm import tqdm

from PIL import Image



class KernelEvalException(Exception):

    pass



model_params = {

    'Inception': {

        'name': 'Inception', 

        'imsize': 64,

        'output_layer': 'Pretrained_Net/pool_3:0', 

        'input_layer': 'Pretrained_Net/ExpandDims:0',

        'output_shape': 2048,

        'cosine_distance_eps': 0.1

        }

}



def create_model_graph(pth):

    """Creates a graph from saved GraphDef file."""

    # Creates graph from saved graph_def.pb.

    with tf.gfile.FastGFile( pth, 'rb') as f:

        graph_def = tf.GraphDef()

        graph_def.ParseFromString( f.read())

        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')



def _get_model_layer(sess, model_name):

    # layername = 'Pretrained_Net/final_layer/Mean:0'

    layername = model_params[model_name]['output_layer']

    layer = sess.graph.get_tensor_by_name(layername)

    ops = layer.graph.get_operations()

    for op_idx, op in enumerate(ops):

        for o in op.outputs:

            shape = o.get_shape()

            if shape._dims != []:

              shape = [s.value for s in shape]

              new_shape = []

              for j, s in enumerate(shape):

                if s == 1 and j == 0:

                  new_shape.append(None)

                else:

                  new_shape.append(s)

              o.__dict__['_shape_val'] = tf.TensorShape(new_shape)

    return layer



def get_activations(images, sess, model_name, batch_size=50, verbose=False):

    """Calculates the activations of the pool_3 layer for all images.



    Params:

    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values

                     must lie between 0 and 256.

    -- sess        : current session

    -- batch_size  : the images numpy array is split into batches with batch size

                     batch_size. A reasonable batch size depends on the disposable hardware.

    -- verbose    : If set to True and parameter out_step is given, the number of calculated

                     batches is reported.

    Returns:

    -- A numpy array of dimension (num images, 2048) that contains the

       activations of the given tensor when feeding inception with the query tensor.

    """

    inception_layer = _get_model_layer(sess, model_name)

    n_images = images.shape[0]

    if batch_size > n_images:

        print("warning: batch size is bigger than the data size. setting batch size to data size")

        batch_size = n_images

    n_batches = n_images//batch_size + 1

    pred_arr = np.empty((n_images,model_params[model_name]['output_shape']))

    for i in tqdm(range(n_batches)):

        if verbose:

            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)

        start = i*batch_size

        if start+batch_size < n_images:

            end = start+batch_size

        else:

            end = n_images

                    

        batch = images[start:end]

        pred = sess.run(inception_layer, {model_params[model_name]['input_layer']: batch})

        pred_arr[start:end] = pred.reshape(-1,model_params[model_name]['output_shape'])

    if verbose:

        print(" done")

    return pred_arr





# def calculate_memorization_distance(features1, features2):

#     neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', metric='euclidean')

#     neigh.fit(features2) 

#     d, _ = neigh.kneighbors(features1, return_distance=True)

#     print('d.shape=',d.shape)

#     return np.mean(d)



def normalize_rows(x: np.ndarray):

    """

    function that normalizes each row of the matrix x to have unit length.



    Args:

     ``x``: A numpy matrix of shape (n, m)



    Returns:

     ``x``: The normalized (by row) numpy matrix.

    """

    return np.nan_to_num(x/np.linalg.norm(x, ord=2, axis=1, keepdims=True))





def cosine_distance(features1, features2):

    # print('rows of zeros in features1 = ',sum(np.sum(features1, axis=1) == 0))

    # print('rows of zeros in features2 = ',sum(np.sum(features2, axis=1) == 0))

    features1_nozero = features1[np.sum(features1, axis=1) != 0]

    features2_nozero = features2[np.sum(features2, axis=1) != 0]

    norm_f1 = normalize_rows(features1_nozero)

    norm_f2 = normalize_rows(features2_nozero)



    d = 1.0-np.abs(np.matmul(norm_f1, norm_f2.T))

    print('d.shape=',d.shape)

    print('np.min(d, axis=1).shape=',np.min(d, axis=1).shape)

    mean_min_d = np.mean(np.min(d, axis=1))

    print('distance=',mean_min_d)

    return mean_min_d





def distance_thresholding(d, eps):

    if d < eps:

        return d

    else:

        return 1



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    """Numpy implementation of the Frechet Distance.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)

    and X_2 ~ N(mu_2, C_2) is

            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

            

    Stable version by Dougal J. Sutherland.



    Params:

    -- mu1 : Numpy array containing the activations of the pool_3 layer of the

             inception net ( like returned by the function 'get_predictions')

             for generated samples.

    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted

               on an representive data set.

    -- sigma1: The covariance matrix over activations of the pool_3 layer for

               generated samples.

    -- sigma2: The covariance matrix over activations of the pool_3 layer,

               precalcualted on an representive data set.



    Returns:

    --   : The Frechet Distance.

    """



    mu1 = np.atleast_1d(mu1)

    mu2 = np.atleast_1d(mu2)



    sigma1 = np.atleast_2d(sigma1)

    sigma2 = np.atleast_2d(sigma2)



    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"

    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"



    diff = mu1 - mu2



    # product might be almost singular

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    if not np.isfinite(covmean).all():

        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps

        warnings.warn(msg)

        offset = np.eye(sigma1.shape[0]) * eps

        # covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    

    # numerical error might give slight imaginary component

    if np.iscomplexobj(covmean):

        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):

            m = np.max(np.abs(covmean.imag))

            raise ValueError("Imaginary component {}".format(m))

        covmean = covmean.real



    # covmean = tf.linalg.sqrtm(tf.linalg.matmul(sigma1,sigma2))



    print('covmean.shape=',covmean.shape)

    # tr_covmean = tf.linalg.trace(covmean)



    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    # return diff.dot(diff) + tf.linalg.trace(sigma1) + tf.linalg.trace(sigma2) - 2 * tr_covmean

#-------------------------------------------------------------------------------





def calculate_activation_statistics(images, sess, model_name, batch_size=50, verbose=False):

    """Calculation of the statistics used by the FID.

    Params:

    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values

                     must lie between 0 and 255.

    -- sess        : current session

    -- batch_size  : the images numpy array is split into batches with batch size

                     batch_size. A reasonable batch size depends on the available hardware.

    -- verbose     : If set to True and parameter out_step is given, the number of calculated

                     batches is reported.

    Returns:

    -- mu    : The mean over samples of the activations of the pool_3 layer of

               the incption model.

    -- sigma : The covariance matrix of the activations of the pool_3 layer of

               the incption model.

    """

    act = get_activations(images, sess, model_name, batch_size, verbose)

    mu = np.mean(act, axis=0)

    sigma = np.cov(act, rowvar=False)

    return mu, sigma, act

    

def _handle_path_memorization(path, sess, model_name, is_checksize, is_check_png):

    path = pathlib.Path(path)

    files = list(path.glob('*.jpg')) + list(path.glob('*.png'))

    imsize = model_params[model_name]['imsize']



    # In production we don't resize input images. This is just for demo purpose. 

    x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])

    m, s, features = calculate_activation_statistics(x, sess, model_name)

    del x #clean up memory

    return m, s, features



# check for image size

def img_read_checks(filename, resize_to, is_checksize=False, check_imsize = 64, is_check_png = False):

    im = Image.open(str(filename))

    if is_checksize and im.size != (check_imsize,check_imsize):

        raise KernelEvalException('The images are not of size '+str(check_imsize))

    

    if is_check_png and im.format != 'PNG':

        raise KernelEvalException('Only PNG images should be submitted.')



    if resize_to is None:

        return im

    else:

        return im.resize((resize_to,resize_to),Image.ANTIALIAS)



def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):

    ''' Calculates the KID of two paths. '''

    tf.reset_default_graph()

    create_model_graph(str(model_path))

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        m1, s1, features1 = _handle_path_memorization(paths[0], sess, model_name, is_checksize = True, is_check_png = True)

        if len(mm) != 0:

            m2 = mm

            s2 = ss

            features2 = ff

        elif feature_path is None:

            m2, s2, features2 = _handle_path_memorization(paths[1], sess, model_name, is_checksize = False, is_check_png = False)

        else:

            with np.load(feature_path) as f:

                m2, s2, features2 = f['m'], f['s'], f['features']



        print('m1,m2 shape=',(m1.shape,m2.shape),'s1,s2=',(s1.shape,s2.shape))

        print('starting calculating FID')

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)

        print('done with FID, starting distance calculation')

        distance = cosine_distance(features1, features2)        

        return fid_value, distance, m2, s2, features2
if ComputeLB:

  

    # UNCOMPRESS OUR IMGAES

    with zipfile.ZipFile("../working/images.zip","r") as z:

        z.extractall("../tmp/images2/")



    # COMPUTE LB SCORE

    m2 = []; s2 =[]; f2 = []

    user_images_unzipped_path = '../tmp/images2/'

    images_path = [user_images_unzipped_path,'../input/generative-dog-images/all-dogs/all-dogs/']

    public_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'



    fid_epsilon = 10e-15



    fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)

    distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])

    print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",

            fid_value_public /(distance_public + fid_epsilon))

    

    # REMOVE FILES TO PREVENT KERNEL ERROR OF TOO MANY FILES

    ! rm -r ../tmp