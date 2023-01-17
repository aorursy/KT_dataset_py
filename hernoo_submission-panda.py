import tensorflow as tf

import numpy as np

class QuadraticWeightedKappa(tf.keras.metrics.Metric):

    def __init__(self, maxClassesCount=6, name='Kappa', **kwargs):        

        super(QuadraticWeightedKappa, self).__init__(name=name, **kwargs)

        self.M = maxClassesCount



        self.O = self.add_weight(name='O', initializer='zeros',shape=(self.M,self.M,), dtype=tf.int64)

        self.W = self.add_weight(name='W', initializer='zeros',shape=(self.M,self.M,), dtype=tf.float32)

        self.actualHist = self.add_weight(name='actHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)

        self.predictedHist = self.add_weight(name='predHist', initializer='zeros',shape=(self.M,), dtype=tf.int64)

        

        # filling up the content of W once

        w = np.zeros((self.M,self.M),dtype=np.float32)

        for i in range(0,self.M):

            for j in range(0,self.M):

                w[i,j] = (i-j)*(i-j) / ((self.M - 1)*(self.M - 1))

        self.W.assign(w)

    

    def reset_states(self):

        

        #Resets all of the metric state variables.

        #This function is called between epochs/steps,

        #when a metric is evaluated during training.

        

        

        # value should be a Numpy array

        zeros1D = np.zeros(self.M)

        zeros2D = np.zeros((self.M,self.M))

        tf.keras.backend.batch_set_value([

            (self.O, zeros2D),

            (self.actualHist, zeros1D),

            (self.predictedHist,zeros1D)

        ])







    def update_state(self, y_true, y_pred, sample_weight=None):

        # shape is: Batch x 1

        y_true = tf.reshape(y_true, [-1])

        y_pred = tf.reshape(y_pred, [-1])



        y_true_int = tf.cast(tf.math.round(y_true), dtype=tf.int64)

        y_pred_int = tf.cast(tf.math.round(y_pred), dtype=tf.int64)



        confM = tf.math.confusion_matrix(y_true_int, y_pred_int, dtype=tf.int64, num_classes=self.M)



        # incremeting confusion matrix and standalone histograms

        self.O.assign_add(confM)



        cur_act_hist = tf.math.reduce_sum(confM, 0)

        self.actualHist.assign_add(cur_act_hist)



        cur_pred_hist = tf.math.reduce_sum(confM, 1)

        self.predictedHist.assign_add(cur_pred_hist)



    def result(self):

        EFloat = tf.cast(tf.tensordot(self.actualHist,self.predictedHist, axes=0),dtype=tf.float32)

        OFloat = tf.cast(self.O,dtype=tf.float32)

        

        # E must be normalized "such that E and O have the same sum"

        ENormalizedFloat = EFloat / tf.math.reduce_sum(EFloat) * tf.math.reduce_sum(OFloat)



        

        return 1.0 - tf.math.reduce_sum(tf.math.multiply(self.W, OFloat))/tf.math.reduce_sum(tf.multiply(self.W, ENormalizedFloat))



from tensorflow import keras

tf.compat.v1.disable_eager_execution()

model = keras.models.load_model('../input/panda-resnet50/resnet50/',custom_objects={'QuadraticWeightedKappa':QuadraticWeightedKappa})



import os

import cv2

import PIL

import random

import openslide

import skimage.io

import matplotlib

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from IPython.display import Image, display

from os import walk

import random

from random import randrange



from sklearn.model_selection import train_test_split

import tensorflow as tf

height = 512

width = 512
"""%load_ext line_profiler"""
#Functions

## to get the % of different colors 

def compute_statistics(image):

    

    #Args:

    #    image                  numpy.array   multi-dimensional array of the form WxHxC

    #

    #Returns:

    #    ratio_white_pixels     float         ratio of white pixels over total pixels in the image 

    

    width, height = image.shape[0], image.shape[1]

    num_pixels = width * height

    

    num_white_pixels = 0

    

    summed_matrix = np.sum(image, axis=-1)

    # Note: A 3-channel white pixel has RGB (255, 255, 255)

    num_white_pixels = np.count_nonzero((summed_matrix > 620)) #avoid too white and/or too blank

    

    ratio_white_pixels = num_white_pixels / num_pixels

    

    green_concentration = np.mean(image[1])

    blue_concentration = np.mean(image[2])

#     red_median = np.percentile(image[0],50)

#     green_median = np.percentile(image[1],50)

#     blue_median = np.percentile(image[2],50)

#     return ratio_white_pixels, green_concentration, blue_concentration, red_median, green_median, blue_median

    return ratio_white_pixels, green_concentration, blue_concentration

#selection of the k-best regions

def select_k_best_regions(regions, k=20):

    

    #Args:

    #    regions               list           list of 2-component tuples first component the region, 

    #                                         second component the ratio of white pixels

    #                                         

    #    k                     int            number of regions to select

    

    red_penalty=0

    #regions = [x for x in regions if ((x[3] > 180 and x[4] > 180) and (((x[5]-red_penalty)>x[6]) or ((x[5]-red_penalty)>x[7])))] # x[3] is green concentration and 4 is blue 

    regions = [x for x in regions if (x[3] > 180 and x[4] > 180)] # x[3] is green concentration and 4 is blue 

    k_best_regions = sorted(regions, key=lambda tup: tup[2])[:k]

    return k_best_regions

#to retrieve from the top left pixel the full region



def get_k_best_regions(coordinates, image, window_size=512):

    regions = {}

    for i, tup in enumerate(coordinates):

        x, y = tup[0], tup[1]

        regions[i] = image[x : x+window_size, y : y+window_size, :]

    

    return regions



#the slider

def generate_patches(slide_path, window_size=200, stride=128, k=20):

    

    image = np.array(skimage.io.MultiImage(slide_path)[-2])

#     image = image

    

    max_width, max_height = image.shape[0], image.shape[1]

    regions_container = []

    i = 0

    

    while window_size + stride*i <= max_height:

        j = 0

        

        while window_size + stride*j <= max_width:            

            x_top_left_pixel = j * stride

            y_top_left_pixel = i * stride

            

            patch = image[

                x_top_left_pixel : x_top_left_pixel + window_size,

                y_top_left_pixel : y_top_left_pixel + window_size,

                :

            ]

            

            ratio_white_pixels, green_concentration, blue_concentration = compute_statistics(patch)

            

            region_tuple = (x_top_left_pixel, y_top_left_pixel, ratio_white_pixels, green_concentration, blue_concentration)

            regions_container.append(region_tuple)

            #print(f' DEBUG : rmed :{red_median} gmed :{green_median} bmed : {blue_median}')

            j += 1

        

        i += 1

    

    k_best_region_coordinates = select_k_best_regions(regions_container, k=k)

    k_best_regions = get_k_best_regions(k_best_region_coordinates, image, window_size)

    

    return image, k_best_region_coordinates, k_best_regions





#showing results 

def display_images(regions, title):

    fig, ax = plt.subplots(5, 4, figsize=(15, 15))

    

    for i, region in regions.items():

        ax[i//4, i%4].imshow(region)

    

    fig.suptitle(title)

    

## glue to a unique picture

def glue_to_one_picture(image_patches, window_size=200, k=16):

    side = int(np.sqrt(k))

    image = np.zeros((side*window_size, side*window_size, 3), dtype=np.int16)

        

    for i, patch in image_patches.items():

        x = i // side

        y = i % side

        image[

            x * window_size : (x+1) * window_size,

            y * window_size : (y+1) * window_size,

            :

        ] = patch

    

    return image

def _parse_function(filename, label, h=height, w=width, rotating=True): 

    image = tf.io.read_file(filename) #reading the image in the memory

    #image = tfio.experimental.image.decode_tiff(image, index=0, name=None) # tiff to numpy 1/2

    image = tf.io.decode_png(image,dtype=tf.dtypes.uint8, name=None)

    if rotating==True :

        sh=tf.shape(tf.io.read_file(filename))

        print(f"{sh}")

        width, height = image.shape[0],image.shape[1]

        print(f"w={width}, h={height}")

        image = tf.image.convert_image_dtype(image, tf.float32) # converting it 2/2

#     

    image = image/255 #normalisation

    #image = tf.image.resize(image, [h, w]) #resize

    image = tf.cast(image, tf.float32) # transforming into a tf.float object



    return image, label

"""#V57

import pandas as pd

import gc #garbage collection



WINDOW_SIZE = 128

STRIDE = 75

K = 16





if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

    data_dir = '../input/prostate-cancer-grade-assessment/test_images/'

else:

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=10)

    data_dir = '../input/prostate-cancer-grade-assessment/train_images/'

    

images = list(final_validation['image_id'])





df_np=pd.DataFrame(columns=['np_image'], index=np.arange(len(images))).fillna(0)

df_np['np_image']=df_np['np_image'].astype(object)

for i, img in enumerate(images):

    url = data_dir + img + '.tiff'

    image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

    glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)





    #df_np.at[i,'np_image']=glued_image.reshape([-1, 1,512,512,3])

    df_np.at[i,'np_image']=glued_image.reshape([-1,512,512,3])

np_image_from_df=np.stack(df_np['np_image'])/255

final_validation_dataset = (

tf.data.Dataset.from_tensor_slices(

    (

        np_image_from_df

    )

    )

)

final_validation_dataset_tens=final_validation_dataset



isup_grade_data = model.predict(final_validation_dataset_tens).argmax(axis=-1).astype(int)

my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': isup_grade_data})

    

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)



"""





import pandas as pd

import gc #garbage collection



WINDOW_SIZE = 128

STRIDE = 128

K = 16







if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

     #test to try out the prediction

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

    data_dir = '../input/prostate-cancer-grade-assessment/test_images/'

    """

    test_dataset=pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

    test_count=len(test_dataset.index)

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=test_count)

    data_dir = '../input/prostate-cancer-grade-assessment/train_images/'"""

else:

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=50)

    data_dir = '../input/prostate-cancer-grade-assessment/train_images/'



images = list(final_validation['image_id'])





df_np=pd.DataFrame(columns=['np_image'], index=np.arange(len(images))).fillna(0)

df_np['np_image']=df_np['np_image'].astype(object)

def treat_images():

    for i, img in enumerate(images):

        url = data_dir + img + '.tiff'

        image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

        glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)





        #df_np.at[i,'np_image']=glued_image.reshape([-1, 1,512,512,3])

        df_np.at[i,'np_image']=glued_image.reshape([-1,512,512,3])

    np_image_from_df=np.stack(df_np['np_image']/255)

    final_validation_dataset = (

    tf.data.Dataset.from_tensor_slices(

        (

            np_image_from_df

        )

        )

    )

    final_validation_dataset_tens=final_validation_dataset

    isup_grade_data = model.predict(final_validation_dataset_tens).argmax(axis=-1).astype(int)

    my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': isup_grade_data})

    # you could use any filename. We choose submission here

    my_submission.to_csv('submission.csv', index=False)

treat_images()

"""test to try out the image processing

    #temporary test

#     isup_grade_data = model.predict(final_validation_dataset_tens).argmax(axis=-1).astype(int)

#     my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': isup_grade_data})

    my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': 1})

""" """

    

    



#%lprun -f treat_images treat_images()

"""
"""pd.read_csv('submission.csv')"""



my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': 1})

my_submission.to_csv('submission.csv', index=False)
"""#V58 direct TF

import numpy as np

import pandas as pd

import gc #garbage collection

import skimage.io

WINDOW_SIZE = 128

STRIDE = 128

K = 16

def parse(s):

    return s, d[s]



def process_path(file_path):

    print()

#     a=lambda x: tf.py_function(parse, [x], [tf.string])

    url=file_path.numpy()

    print(url)

    image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

    glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)

    

    # load the raw data from the file as a string

#     img = tf.io.read_file(file_path)

#     img = decode_img(img)

    img = glued_image/255

    img=tf.reshape(img, [1,512,512,3])

#     img = img.reshape(-1,512,512,3)

    

    return img



def process_path(file_path):

    url=file_path.numpy()

    print(url)

    glued_image=np.array(skimage.io.MultiImage('../input/prostate-cancer-grade-assessment/train_images/0005f7aaab2800f6170c399693a96917.tiff'))

    img = glued_image/255

    img=tf.reshape(img, [1,512,512,3])

#     image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

#     glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)

    return img



if os.path.exists('../input/prostate-cancer-grade-assessment/test_images'):

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/test.csv')

    data_dir = '../input/prostate-cancer-grade-assessment/test_images/'

else:

    final_validation = pd.read_csv('../input/prostate-cancer-grade-assessment/train.csv').sample(n=10)

    data_dir = '../input/prostate-cancer-grade-assessment/train_images/'

images = list(final_validation['image_id']) 

# images = list(data_dir+final_validation['image_id']+'.tiff')

# train_ds = tf.data.Dataset.from_generator(process_path, args=images, output_types=tf.float32, output_shapes = (1,512,512,3), )

list_ds = tf.data.Dataset.from_tensor_slices(data_dir+final_validation.image_id+'.tiff')



#train_ds = list_ds.map(process_path)

# train_ds = list_ds.map(lambda x: tf.numpy_function(func=process_path, inp=[x], Tout=tf.float32))

train_ds = list_ds.map(

    lambda path: tf.py_function(

            func=process_path,

            inp=[path],

            Tout=[tf.string],

    )

)

final_validation_dataset_tens=train_ds



isup_grade_data = model.predict(final_validation_dataset_tens).argmax(axis=-1).astype(int)

my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': isup_grade_data})

    

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

"""
"""

df_np=pd.DataFrame(columns=['np_image'], index=np.arange(len(images))).fillna(0)

df_np['np_image']=df_np['np_image'].astype(object)

for i, img in enumerate(images):

    url = data_dir + img + '.tiff'

    image, best_coordinates, best_regions = generate_patches(url, window_size=WINDOW_SIZE, stride=STRIDE, k=K)

    glued_image = glue_to_one_picture(best_regions, window_size=WINDOW_SIZE, k=K)





    #df_np.at[i,'np_image']=glued_image.reshape([-1, 1,512,512,3])

    df_np.at[i,'np_image']=glued_image.reshape([-1,512,512,3])

np_image_from_df=np.stack(df_np['np_image'])/255

final_validation_dataset = (

tf.data.Dataset.from_tensor_slices(

    (

        np_image_from_df

    )

    )

)



final_validation_dataset_tens=final_validation_dataset



isup_grade_data = model.predict(final_validation_dataset_tens).argmax(axis=-1).astype(int)

my_submission = pd.DataFrame({'image_id': final_validation.image_id, 'isup_grade': isup_grade_data})

    

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)

"""

# 
# import matplotlib.pyplot as plt

# for f in final_validation_dataset.take(1):

#     print(f.numpy())

#     plt.imshow(f.numpy().reshape(512,512,3))