from __future__ import absolute_import, division, print_function

import gzip, pickle

import pathlib

import urllib



import random

import time

import glob

import shutil

import warnings

import os

from tqdm import tqdm



import cv2

import scipy

from scipy import linalg



import numpy as np

import pandas as pd



import xml.etree.ElementTree as ET 

import matplotlib.pyplot as plt, zipfile 



import tensorflow as tf

from tensorflow.python.framework import ops

import albumentations as A

from albumentations.pytorch import ToTensor

from sklearn.preprocessing import LabelEncoder

 

from keras.models import Model

from keras.layers import Input, Dense, Conv2D, Reshape, Flatten, concatenate, UpSampling2D

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from keras.optimizers import SGD, Adam



from PIL import Image

%matplotlib inline

warnings.filterwarnings('ignore', category=FutureWarning)
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



seed_everything(43)
ROOT = '../input/' # директория, в которую подгружаются файлы

IMAGES = os.listdir(ROOT + 'all-dogs/all-dogs/')

breeds = os.listdir(ROOT + 'Annotation/Annotation/') 



idxIn = 0; namesIn = []

imagesIn = np.zeros((25000,64,64,3))
ComputeLB = True # флаг для подсчета метрики

DogsOnly = True # флаг для выделение собачек
%%time

if DogsOnly:

    for breed in breeds:

        for dog in os.listdir(ROOT+'Annotation/Annotation/'+breed):

            try: img = Image.open(ROOT+'all-dogs/all-dogs/'+dog+'.jpg') 

            except: continue           

            tree = ET.parse(ROOT+'Annotation/Annotation/'+breed+'/'+dog)

            root = tree.getroot()

            objects = root.findall('object')

            for o in objects:

                bndbox = o.find('bndbox') 

                xmin = int(bndbox.find('xmin').text)

                ymin = int(bndbox.find('ymin').text)

                xmax = int(bndbox.find('xmax').text)

                ymax = int(bndbox.find('ymax').text)

                w = np.min((xmax - xmin, ymax - ymin))

                img2 = img.crop((xmin, ymin, xmin+w, ymin+w))

                img2 = img2.resize((64,64), Image.ANTIALIAS)

                imagesIn[idxIn,:,:,:] = np.asarray(img2)

                #if idxIn%1000==0: print(idxIn)

                namesIn.append(breed)

                idxIn += 1

    idx = np.arange(idxIn)

    np.random.shuffle(idx)

    imagesIn = imagesIn[idx,:,:,:]

    namesIn = np.array(namesIn)[idx]
x = np.random.randint(0,idxIn,25)

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = Image.fromarray( imagesIn[x[k*5+j],:,:,:].astype('uint8') )

        plt.axis('off')

        if not DogsOnly: plt.title(namesIn[x[k*5+j]],fontsize=11)

        else: plt.title(namesIn[x[k*5+j]].split('-')[1],fontsize=11)

        plt.imshow(img)

    plt.show()
# Построим наш Дискриминатор

dog = Input((12288,))

dogName = Input((10000,))

x = Dense(12288, activation='sigmoid')(dogName) 

x = Reshape((2,12288,1))(concatenate([dog,x]))

x = Conv2D(1,(2,1),use_bias=False,name='conv')(x)

discriminated = Flatten()(x)



# Скомпилируем

discriminator = Model([dog,dogName], discriminated)

discriminator.get_layer('conv').trainable = False

discriminator.get_layer('conv').set_weights([np.array([[[[-1.0 ]]],[[[1.0]]]])])

discriminator.compile(optimizer='adam', loss='binary_crossentropy')



# Выведем архитектуру

discriminator.summary()
%%time

# TRAINING DATA

BATCH_SIZE = 256

train_y = (imagesIn[:10000,:,:,:]/255.).reshape((-1,12288))

train_X = np.zeros((10000,10000))

for i in range(10000): train_X[i,i] = 1

zeros = np.zeros((10000,12288))



# TRAIN NETWORK

lr = 0.5

for k in range(6):

    annealer = LearningRateScheduler(lambda x: lr)

    # This function keeps the learning rate at 0.5 for the first ten epochs

    # and decreases it exponentially after that.

    h = discriminator.fit([zeros,train_X], train_y, epochs = 10, batch_size=BATCH_SIZE, callbacks=[annealer], verbose=0)

    print('Epoch',(k+1)*10,'/60 - loss =',h.history['loss'][-1] )

    if h.history['loss'][-1]<0.533: lr = 0.1
del train_X, train_y, imagesIn
print('Discriminator Recalls from Memory Dogs')    

for k in range(5):

    plt.figure(figsize=(15,3))

    for j in range(5):

        xx = np.zeros((10000))

        xx[np.random.randint(10000)] = 1

        plt.subplot(1,5,j+1)

        img = discriminator.predict([zeros[0,:].reshape((-1,12288)),xx.reshape((-1,10000))]).reshape((-1,64,64,3))

        img = Image.fromarray( (255*img).astype('uint8').reshape((64,64,3)))

        plt.axis('off')

        plt.imshow(img)

    plt.show()
#BadMemory = True

#with tpu_strategy.scope():

BadMemory = True



if BadMemory:

    seed = Input((10000,))

    x = Dense(2048, activation='elu')(seed)

    x = Reshape((8,8,32))(x)

    x = Conv2D(128, (3, 3), activation='elu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='elu', padding='same')(x)

    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='elu', padding='same')(x)

    x = UpSampling2D((2, 2))(x)

    x = Conv2D(8, (3, 3), activation='elu', padding='same')(x)

    x = Conv2D(4, (3, 3), activation='elu', padding='same')(x)

    x = Conv2D(3, (3, 3), activation='linear', padding='same')(x)

    generated = Flatten()(x)

else:

    seed = Input((10000,))

    generated = Dense(12288, activation='linear')(seed)



# COMPILE

generator = Model(seed, [generated,Reshape((10000,))(seed)])



# DISPLAY ARCHITECTURE

generator.summary()

#Посторим сеть

discriminator.trainable=False    

gan_input = Input(shape=(10000,))

x = generator(gan_input)

gan_output = discriminator(x)



# скомпилируем GAN

gan = Model(gan_input, gan_output)

gan.get_layer('model_1').get_layer('conv').set_weights([np.array([[[[-1 ]]],[[[255.]]]])])

gan.compile(optimizer=Adam(5), loss='mean_squared_error')



# выведем архитектуру

gan.summary()
%%time

# TRAINING DATA

train = np.zeros((10000,10000))

for i in range(10000): train[i,i] = 1

zeros = np.zeros((10000,12288))



# TRAIN NETWORKS

ep = 1; it = 9



if BadMemory: lr = 0.005

else: lr = 5.

    

for k in range(it):  



    # BEGIN DISCRIMINATOR COACHES GENERATOR

    annealer = LearningRateScheduler(lambda x: lr)

    h = gan.fit(train, zeros, epochs = ep, batch_size=256, callbacks=[annealer], verbose=0)



    # DISPLAY GENERATOR LEARNING PROGRESS 

    print('Epoch',(k+1),'/'+str(it)+' - loss =',h.history['loss'][-1] )

    plt.figure(figsize=(15,3))

    for j in range(5):

        xx = np.zeros((10000))

        xx[np.random.randint(10000)] = 1

        plt.subplot(1,5,j+1)

        img = generator.predict(xx.reshape((-1,10000)))[0].reshape((-1,64,64,3))

        img = Image.fromarray( (img).astype('uint8').reshape((64,64,3)))

        plt.axis('off')

        plt.imshow(img)

    plt.show()  

            

    # ADJUST LEARNING RATES

    if BadMemory:

        ep *= 2

        if ep>=32: lr = 0.001

        if ep>256: ep = 256

    else:

        if h.history['loss'][-1] < 25: lr = 1.

        if h.history['loss'][-1] < 1.5: lr = 0.5

class DogGenerator:

    index = 0   

    def getDog(self,seed):

        xx = np.zeros((10000))

        xx[self.index] = 0.70

        xx[np.random.randint(10000)] = 0.30

        img = generator.predict(xx.reshape((-1,10000)))[0].reshape((64,64,3))

        self.index = (self.index+1)%10000

        return Image.fromarray( img.astype('uint8') ) 
# Выведем сгенерированные изображения

d = DogGenerator()

for k in range(3):

    plt.figure(figsize=(15,3))

    for j in range(5):

        plt.subplot(1,5,j+1)

        img = d.getDog(np.random.normal(0,1,100))

        plt.axis('off')

        plt.imshow(img)

    plt.show() 
%%time

# сохраним в зип формате изображения

z = zipfile.PyZipFile('images2.zip', mode='w')

d = DogGenerator()

for k in range(10000):

    img = d.getDog(np.random.normal(0,1,100))

    f = str(k)+'.png'

    img.save(f,'PNG'); z.write(f); os.remove(f)

    #if k % 1000==0: print(k)

z.close()
class KernelEvalException(Exception):

    pass



model_params = {

    'Inception': {

        'name': 'Inception', 

        'imsize': 64,

        'output_layer': 'pool_3:0', 

        'input_layer': 'ExpandDims:0',

        'output_shape': 2048,

        'cosine_distance_eps': 0.1

        }

}



def create_model_graph(pth):

    """Creates a graph from saved GraphDef file."""

    # Creates graph from saved graph_def.pb.

    with tf.io.gfile.GFile( pth, 'rb') as f:

        graph_def = tf.compat.v1.GraphDef()

        graph_def.ParseFromString( f.read())

        _ = tf.import_graph_def( graph_def, name='Pretrained_Net')



def _get_model_layer(sess, model_name):

    # layername = 'Pretrained_Net/final_layer/Mean:0'

    layername = model_params[model_name]['output_layer']

    layer = sess.graph.get_tensor_by_name(layername)

    print(layer)

#     layer.__dict__['_shape_val'] = tf.TensorShape([1, 1, 1, 2048])

    ops = layer.graph.get_operations()

    for op_idx, op in enumerate(ops):

        for o in op.outputs:

            shape = o.get_shape()

#             print(shape._dims)

            if shape._dims != []:

              shape = [s if s != None else None for s in shape]

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

    print(imsize)

    # In production we don't resize input images. This is just for demo purpose. 

    x_t = [np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files]

    print(len(x_t))

    x = np.array(x_t)

    #x = np.array([np.array(img_read_checks(fn, imsize, is_checksize, imsize, is_check_png)) for fn in files])

    

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

        #print(Image.ANTIALIAS)

        return im.resize((resize_to,resize_to),Image.ANTIALIAS)



def calculate_kid_given_paths(paths, model_name, model_path, feature_path=None, mm=[], ss=[], ff=[]):

    ''' Calculates the KID of two paths. '''

    ops.reset_default_graph()

    create_model_graph(str(model_path))

    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

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
# UNCOMPRESS OUR IMGAES

with zipfile.ZipFile("../working/images2.zip","r") as z:

    z.extractall("../working/images2/")
# COMPUTE LB SCORE

m2 = []; s2 =[]; f2 = []

user_images_unzipped_path = '../working/images2/'

images_path = [user_images_unzipped_path,'/kaggle/input/all-dogs/all-dogs/']

public_path = '/kaggle/input/classify_image_graph_def.pb'



fid_epsilon = 10e-15
%%time

fid_value_public, distance_public, m2, s2, f2 = calculate_kid_given_paths(images_path, 'Inception', public_path, mm=m2, ss=s2, ff=f2)
distance_public = distance_thresholding(distance_public, model_params['Inception']['cosine_distance_eps'])

print("FID_public: ", fid_value_public, "distance_public: ", distance_public, "multiplied_public: ",

        fid_value_public /(distance_public + fid_epsilon))
