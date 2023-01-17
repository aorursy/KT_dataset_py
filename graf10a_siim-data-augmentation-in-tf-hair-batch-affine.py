import os

import gc

import cv2

import math

import random

import numpy as np

import pandas as pd

from glob import glob

import tensorflow as tf

from pathlib import Path

import matplotlib.pyplot as plt

import tensorflow.keras.backend as K

from kaggle_datasets import KaggleDatasets



print("Tensorflow version " + tf.__version__)

AUTO = tf.data.experimental.AUTOTUNE
n_max=6     # the maximum number of hairs to augment

im_size=512  # all images are resized to this size



hair_images=glob('/kaggle/input/melanoma-hairs/*.png')

train_images=glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/train/*.jpg')

test_images=glob('/kaggle/input/siim-isic-melanoma-classification/jpeg/test/*.jpg')



len(hair_images), len(train_images), len(test_images)
def hair_aug_ocv(input_img):

    

    img=input_img.copy()

    # Randomly choose the number of hairs to augment (up to n_max)

    n_hairs = random.randint(0, n_max)



    # If the number of hairs is zero then do nothing

    if not n_hairs:

        return img, n_hairs



    # The image height and width (ignore the number of color channels)

    im_height, im_width, _ = img.shape 



    for _ in range(n_hairs):



        # Read a random hair image

        hair = cv2.imread(random.choice(hair_images)) 

        

        # Rescale the hair image to the right size (256 -- original size)

        scale=im_size/256

        hair = cv2.resize(hair, (int(scale*hair.shape[1]), int(scale*hair.shape[0])), 

                          interpolation=cv2.INTER_AREA)       



        # Flip it

        # flipcode = 0: flip vertically

        # flipcode > 0: flip horizontally

        # flipcode < 0: flip vertically and horizontally    

        hair = cv2.flip(hair, flipCode=random.choice([-1, 0, 1]))



        # Rotate it

        hair = cv2.rotate(hair, rotateCode=random.choice([cv2.ROTATE_90_CLOCKWISE,

                                                          cv2.ROTATE_90_COUNTERCLOCKWISE,

                                                          cv2.ROTATE_180

                                                         ])

                         )

        

        

        # The hair image height and width (ignore the number of color channels)

        h_height, h_width, _ = hair.shape



        # The top left coord's of the region of interest (roi)  

        # where the augmentation will be performed

        roi_h0 = random.randint(0, im_height - h_height)

        roi_w0 = random.randint(0, im_width - h_width)



        # The region of interest

        roi = img[roi_h0:(roi_h0 + h_height), roi_w0:(roi_w0 + h_width)]



        # Convert the hair image to grayscale

        hair2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)



        # If the pixel value is smaller than the threshold (10), it is set to 0 (black), 

        # otherwise it is set to a maximum value (255, white).

        # ret -- the list of thresholds (10 in this case)

        # mask -- the thresholded image

        # The original image must be a grayscale image

        # https://docs.opencv.org/master/d7/d4d/tutorial_py_thresholding.html

        ret, mask = cv2.threshold(hair2gray, 10, 255, cv2.THRESH_BINARY)



        # Invert the mask

        mask_inv = cv2.bitwise_not(mask)



        # Bitwise AND won't be performed where mask=0

        img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

        hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

        # Fixing colors

        hair_fg = cv2.cvtColor(hair_fg, cv2.COLOR_BGR2RGB)

        # Overlapping the image with the hair in the region of interest

        dst = cv2.add(img_bg, hair_fg)

        # Inserting the result in the original image

        img[roi_h0:roi_h0 + h_height, roi_w0:roi_w0 + h_width] = dst

        

    return img, n_hairs
def aug_examples(paths):



    for img_path in paths:

        # Read the image

        img=cv2.imread(img_path)

        # Fixing colors

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to the desired size

        img = cv2.resize(img , (im_size, im_size), interpolation = cv2.INTER_AREA )

        # Creating an augmented image

        img_aug, n_hairs = hair_aug_ocv(img)

        

        _, (ax1,ax2) = plt.subplots(1, 2)

        

        im_name=img_path.split('/')[-1].split('.')[0]    

        ax1.set_title(f"{im_name}")            

        ax2.set_title(f"{im_name} with {n_hairs} {'hair' if n_hairs==1 else 'hairs'}")

        

        ax1.imshow(img)

        ax2.imshow(img_aug)

        

        plt.tight_layout()

        plt.show()
aug_examples(train_images[6:9])
aug_examples(test_images[6:9])
%%time



# Select a sample image

fname=test_images[7]

# Read and decode the image

bits = tf.io.read_file(fname)

# dct_method='INTEGER_ACCURATE' produces the same result as OpenCV

img0 = tf.image.decode_jpeg(bits, channels=3, dct_method='INTEGER_ACCURATE')



plt.imshow(img0.numpy() / 255)

plt.show()
def resize_and_crop_image(input_img):

    

    img=tf.identity(input_img)

    # Resize and crop using "fill" algorithm:

    # always make sure the resulting image

    # is cut out from the source image so that

    # it fills the TARGET_SIZE entirely with no

    # black bars and a preserved aspect ratio.

    w = tf.shape(img)[0] 

    h = tf.shape(img)[1]

    tw = im_size

    th = im_size

    resize_crit = (w * th) / (h * tw)

    img = tf.cond(resize_crit < 1,

                  # if true

                  lambda: tf.image.resize(img, [w*tw/w, h*tw/w],

                                          #method='lanczos3',

                                          #antialias=True

                                         ),

                  

                  # if false

                  lambda: tf.image.resize(img, [w*th/h, h*th/h],

                                          #method='lanczos3',

                                          #antialias=True

                                         )

                 )

    

    nw = tf.shape(img)[0]

    nh = tf.shape(img)[1]

    img = tf.image.crop_to_bounding_box(img,

                                        (nw - tw) // 2,

                                        (nh - th) // 2,

                                        tw, th

                                       )

    

    return img
img_cropped=resize_and_crop_image(input_img=img0)



# Divide by 255 to bring it in the 0-1 range. plt.imshow()  

# expects either 0 to 1 floats or 0 to 255 integers.

plt.imshow(img_cropped.numpy()/255)

plt.show()
hair_images_tf=tf.convert_to_tensor(hair_images)

scale=tf.cast(im_size/256, dtype=tf.int32)

tf.random.set_seed(42)
def hair_aug_tf(input_img):

    # Copy the input image, so it won't be changed

    img=tf.identity(input_img) 

    # Randomly choose the number of hairs to augment (up to n_max)

    n_hairs = tf.random.uniform(shape=[], maxval=tf.constant(n_max)+1, 

                                dtype=tf.int32)



    im_height = tf.shape(img)[0]

    im_width = tf.shape(img)[1]

    

    if n_hairs == 0:

        return img, n_hairs



    for _ in tf.range(n_hairs):



        # Read a random hair image

        i=tf.random.uniform(shape=[], maxval=tf.shape(hair_images_tf)[0], 

                            dtype=tf.int32)

        fname=hair_images_tf[i]



        bits = tf.io.read_file(fname)

        hair = tf.image.decode_jpeg(bits)

        

        # Rescale the hair image to the right size (256 -- original size)

        new_width=scale*tf.shape(hair)[1]

        new_height=scale*tf.shape(hair)[0]

        hair = tf.image.resize(hair, [new_height, new_width])



        

        # Random flips of the hair image

        hair = tf.image.random_flip_left_right(hair)

        hair = tf.image.random_flip_up_down(hair)

        # Random number of 90 degree rotations

        n_rot=tf.random.uniform(shape=[], maxval=4,

                                dtype=tf.int32)

        hair = tf.image.rot90(hair, k=n_rot)



        h_height = tf.shape(hair)[0]

        h_width = tf.shape(hair)[1]



        roi_h0 = tf.random.uniform(shape=[], maxval=im_height - h_height + 1, 

                                    dtype=tf.int32)

        roi_w0 = tf.random.uniform(shape=[], maxval=im_width - h_width + 1, 

                                    dtype=tf.int32)





        roi = img[roi_h0:(roi_h0 + h_height), roi_w0:(roi_w0 + h_width)]  



        # Convert the hair image to grayscale 

        # (slice to remove the trainsparency channel)

        hair2gray = tf.image.rgb_to_grayscale(hair[:, :, :3])



        mask=hair2gray>10



        img_bg = tf.multiply(roi, tf.cast(tf.image.grayscale_to_rgb(~mask),

                                          dtype=tf.float32))

        hair_fg = tf.multiply(tf.cast(hair[:, :, :3], dtype=tf.int32),

                              tf.cast(tf.image.grayscale_to_rgb(mask), dtype=tf.int32)#uint8)

                             )



        dst = tf.add(img_bg, tf.cast(hair_fg, dtype=tf.float32))



        paddings = tf.stack([

            [roi_h0, im_height-(roi_h0 + h_height)], 

            [roi_w0, im_width-(roi_w0 + h_width)],

            [0, 0]

        ])



        # Pad dst with zeros to make it the same shape as image.

        dst_padded=tf.pad(dst, paddings, "CONSTANT")

        # Create a boolean mask with zeros at the pixels of

        # the augmentation segment and ones everywhere else

        mask_img=tf.pad(tf.ones_like(dst), paddings, "CONSTANT")

        mask_img=~tf.cast(mask_img, dtype=tf.bool)

        # Make a hole in the original image at the location

        # of the augmentation segment

        img_hole=tf.multiply(img, tf.cast(mask_img, dtype=tf.float32))

        # Inserting the augmentation segment in place of the hole

        img=tf.add(img_hole, dst_padded)

        

    return img, n_hairs
def aug_examples_tf(paths):



    for img_path in paths:

        

        # Read and decode the image

        bits = tf.io.read_file(img_path)

        # dct_method='INTEGER_ACCURATE' produces the same result as OpenCV

        img = tf.image.decode_jpeg(bits, channels=3, dct_method='INTEGER_ACCURATE')

        

        # Resize and crop the image

        img=resize_and_crop_image(img)  

        # Creating an augmented image

        img_aug, n_hairs = hair_aug_tf(img)

        

        _, (ax1,ax2) = plt.subplots(1, 2)

        

        im_name=img_path.split('/')[-1].split('.')[0]    

        ax1.set_title(f"{im_name}")            

        ax2.set_title(f"{im_name} with {n_hairs} {'hair' if n_hairs==1 else 'hairs'}")

        

        ax1.imshow(img/255)

        ax2.imshow(img_aug/255)

        

        plt.tight_layout()

        plt.show()
aug_examples_tf(train_images[6:9])
aug_examples(test_images[6:9])
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment 

    # variable is set. On Kaggle this is always the case.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy() 



print("REPLICAS: ", strategy.num_replicas_in_sync)
CLASSES = ['benign', 'malignant']

IMAGE_SIZE = [512, 512]

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

NFOLDS=5
tab_feats=['age_scaled',

           'sex_female', 

           'sex_male', 

           'sex_unknown', 

           'site_head/neck', 

           'site_lower extremity', 

           'site_oral/genital',

           'site_palms/soles',

           'site_torso',

           'site_unknown',

           'site_upper extremity',

#            'height',

#            'width',

          ]



N_TAB_FEATS=len(tab_feats)



print(f"The number of tabular features is {N_TAB_FEATS}.")
GCS_PATH={}



# GCS_PATH['train']=KaggleDatasets().get_gcs_path('siim-tfrec-cc-512-train')

# GCS_PATH['test']=KaggleDatasets().get_gcs_path('siim-tfrec-cc-512-test')

GCS_PATH['train']=KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95')

GCS_PATH['test']=KaggleDatasets().get_gcs_path('siim-512x512-tfrec-q95-test')

# Roman's images of hairs

GCS_PATH['hairs']=KaggleDatasets().get_gcs_path('melanoma-hairs')



print(GCS_PATH['train'])

print(GCS_PATH['test'])

print(GCS_PATH['hairs'])
ALL_TRAIN=tf.io.gfile.glob(GCS_PATH['train'] + '/*.tfrec')



VAL_FNAMES={}

for fn in range(1, NFOLDS+1):

    VAL_FNAMES[f"fold_{fn}"]=[path for path in ALL_TRAIN if f"fold_{fn}" in path]    

    print("Fold", f'{fn}:', len(VAL_FNAMES[f'fold_{fn}']), "elements in total.")

    

TRAIN_FNAMES={f'fold_{i}': list(set(ALL_TRAIN)-set(VAL_FNAMES[f'fold_{i}']))

              for i in range(1, NFOLDS+1)}



TEST_FNAMES = tf.io.gfile.glob(GCS_PATH['test'] + '/*.tfrec')



# Roman's images of hairs

hair_images=tf.io.gfile.glob(GCS_PATH['hairs'] + '/*.png')
def decode_image(image_data):

    image = tf.image.decode_jpeg(image_data, channels=3)

    # convert image to floats in [0, 1] range

    image = tf.cast(image, tf.float32) / 255.0 

    # explicit size needed for TPU

    image = tf.reshape(image, [*IMAGE_SIZE, 3])

    

    return image
def read_labeled_tfrecord(example):

    LABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        # shape [] means single element

        ################################

        # bytestring features

        "image": tf.io.FixedLenFeature([], tf.string), 

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "patient_id": tf.io.FixedLenFeature([], tf.string),

        "benign_malignant": tf.io.FixedLenFeature([], tf.string),

        # integer features

        "age": tf.io.FixedLenFeature([], tf.int64),

        "sex_female": tf.io.FixedLenFeature([], tf.int64),        

        "sex_male": tf.io.FixedLenFeature([], tf.int64),

        "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_head/neck": tf.io.FixedLenFeature([], tf.int64),

        "site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

        "site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

        "site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

        "site_torso": tf.io.FixedLenFeature([], tf.int64),

        "site_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

        "height": tf.io.FixedLenFeature([], tf.int64),

        "width": tf.io.FixedLenFeature([], tf.int64),

        "target": tf.io.FixedLenFeature([], tf.int64), 

        # float features

        "age_scaled": tf.io.FixedLenFeature([], tf.float32),

    }



    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)

    # image data

    image = decode_image(example['image']) 

    data={}

    # bytestring features

    data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    data['patient_id']=tf.cast(example['patient_id'], tf.string)

    # integer features

    data['age']=tf.cast(example['age'], tf.int32)

    data['sex_female']=tf.cast(example['sex_female'], tf.int32)

    data['sex_male']=tf.cast(example['sex_male'], tf.int32)

    data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

    data['site_head/neck']=tf.cast(example['site_head/neck'], tf.int32)

    data['site_lower extremity']=tf.cast(example['site_lower extremity'], tf.int32)

    data['site_oral/genital']=tf.cast(example['site_oral/genital'], tf.int32)

    data['site_palms/soles']=tf.cast(example['site_palms/soles'], tf.int32)

    data['site_torso']=tf.cast(example['site_torso'], tf.int32)

    data['site_unknown']=tf.cast(example['site_unknown'], tf.int32)

    data['site_upper extremity']=tf.cast(example['site_upper extremity'], tf.int32)

    # float features

    data['age_scaled']=tf.cast(example['age_scaled'], tf.float32)

    # target (integer)

    label=tf.cast(example['target'], tf.int32)

     # target (string)

    label_name=tf.cast(example['benign_malignant'], tf.string)



    return image, label, data, label_name
def read_unlabeled_tfrecord(example):

    UNLABELED_TFREC_FORMAT = {

        # tf.string means bytestring

        # shape [] means single element

        ################################

        # bytestring features

        "image": tf.io.FixedLenFeature([], tf.string), 

        "image_name": tf.io.FixedLenFeature([], tf.string),

        "patient_id": tf.io.FixedLenFeature([], tf.string),

        # integer features

        "age": tf.io.FixedLenFeature([], tf.int64),

        "sex_female": tf.io.FixedLenFeature([], tf.int64),        

        "sex_male": tf.io.FixedLenFeature([], tf.int64),

        "sex_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_head/neck": tf.io.FixedLenFeature([], tf.int64),

        "site_lower extremity": tf.io.FixedLenFeature([], tf.int64),

        "site_oral/genital": tf.io.FixedLenFeature([], tf.int64),

        "site_palms/soles": tf.io.FixedLenFeature([], tf.int64),

        "site_torso": tf.io.FixedLenFeature([], tf.int64),

        "site_unknown": tf.io.FixedLenFeature([], tf.int64),

        "site_upper extremity": tf.io.FixedLenFeature([], tf.int64),

        "height": tf.io.FixedLenFeature([], tf.int64),

        "width": tf.io.FixedLenFeature([], tf.int64), 

        # float features

        "age_scaled": tf.io.FixedLenFeature([], tf.float32),

    }



    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)

    # image data

    image = decode_image(example['image']) 

    data={}

    # bytestring features

    data['image_name']=image_name=tf.cast(example['image_name'], tf.string)

    data['patient_id']=tf.cast(example['patient_id'], tf.string)

    # integer features

    data['age']=tf.cast(example['age'], tf.int32)

    data['sex_female']=tf.cast(example['sex_female'], tf.int32)

    data['sex_male']=tf.cast(example['sex_male'], tf.int32)

    data['sex_unknown']=tf.cast(example['sex_unknown'], tf.int32)

    data['site_head/neck']=tf.cast(example['site_head/neck'], tf.int32)

    data['site_lower extremity']=tf.cast(example['site_lower extremity'], tf.int32)

    data['site_oral/genital']=tf.cast(example['site_oral/genital'], tf.int32)

    data['site_palms/soles']=tf.cast(example['site_palms/soles'], tf.int32)

    data['site_torso']=tf.cast(example['site_torso'], tf.int32)

    data['site_unknown']=tf.cast(example['site_unknown'], tf.int32)

    data['site_upper extremity']=tf.cast(example['site_upper extremity'], tf.int32)

    # float features

    data['age_scaled']=tf.cast(example['age_scaled'], tf.float32)



    return image, data
def load_dataset(filenames, labeled=True, ordered=False):

    # Read from TFRecords. For optimal performance, reading from multiple files 

    # at once and disregarding data order. Order does not matter since we will 

    # be shuffling the data anyway.



    ignore_order = tf.data.Options()

    if not ordered:

        # disable order, increase speed

        ignore_order.experimental_deterministic = False



    # automatically interleaves reads from multiple files

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    # uses data as soon as it streams in, rather than in its original order

    dataset = dataset.with_options(ignore_order)

    # returns a dataset of (image, label) pairs if labeled=True 

    # or (image, id) pairs if labeled=False

    dataset = dataset.map(read_labeled_tfrecord if labeled 

                          else read_unlabeled_tfrecord, num_parallel_calls=AUTO)

    

    return dataset
def setup_input(image, label, data, label_name):

    

    tab_data=[tf.cast(data[tfeat], dtype=tf.float32) for tfeat in tab_feats]

    

    tabular=tf.stack(tab_data)

    

    return {'inp1': image, 'inp2':  tabular}, label
def data_augment(data, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement 

    # in the next function (below), this happens essentially for free on TPU. 

    # Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    image = tf.image.random_flip_left_right(data['inp1'])

    image = tf.image.random_flip_up_down(image)

    

    return {'inp1': image, 'inp2':  data['inp2']}, label
hair_images_tf=tf.convert_to_tensor(hair_images)

scale=tf.cast(IMAGE_SIZE[0]/256, dtype=tf.int32)
def hair_aug(data, label):

    # Copy the input image, so it won't be changed

    img=tf.identity(data['inp1']) 

    # Randomly choose the number of hairs to augment (up to n_max)

    n_hairs = tf.random.uniform(shape=[], maxval=tf.constant(n_max)+1, 

                                dtype=tf.int32)

    

    im_height=tf.shape(img)[1]

    im_width=tf.shape(img)[0]

    

    if n_hairs == 0:

        return data, label



    for _ in tf.range(n_hairs):



        # Read a random hair image

        i=tf.random.uniform(shape=[], maxval=tf.shape(hair_images_tf)[0], 

                            dtype=tf.int32)

        fname=hair_images_tf[i]



        bits = tf.io.read_file(fname)

        hair = tf.image.decode_jpeg(bits)

        

        # Rescale the hair image to the right size (256 -- original size)

        new_width=scale*tf.shape(hair)[1]

        new_height=scale*tf.shape(hair)[0]

        hair = tf.image.resize(hair, [new_height, new_width])



        

        # Random flips of the hair image

        hair = tf.image.random_flip_left_right(hair)

        hair = tf.image.random_flip_up_down(hair)

        # Random number of 90 degree rotations

        n_rot=tf.random.uniform(shape=[], maxval=4,

                                dtype=tf.int32)

        hair = tf.image.rot90(hair, k=n_rot)

        

        h_height=tf.shape(hair)[0]

        h_width=tf.shape(hair)[1]

        

        roi_h0 = tf.random.uniform(shape=[], maxval=im_height - h_height + 1, 

                                    dtype=tf.int32)

        roi_w0 = tf.random.uniform(shape=[], maxval=im_width - h_width + 1, 

                                    dtype=tf.int32)





        roi = img[roi_h0:(roi_h0 + h_height), roi_w0:(roi_w0 + h_width)]  



        # Convert the hair image to grayscale 

        # (slice to remove the trainsparency channel)

        hair2gray = tf.image.rgb_to_grayscale(hair[:, :, :3])



        mask=hair2gray>10



        img_bg = tf.multiply(roi, tf.cast(tf.image.grayscale_to_rgb(~mask),

                                          dtype=tf.float32))

        hair_fg = tf.multiply(tf.cast(hair[:, :, :3], dtype=tf.int32),

                              tf.cast(tf.image.grayscale_to_rgb(mask), 

                                      dtype=tf.int32

                                      )

                             )



        dst = tf.add(img_bg, tf.cast(hair_fg, dtype=tf.float32)/255)



        paddings = tf.stack([

            [roi_h0, im_height-(roi_h0 + h_height)], 

            [roi_w0, im_width-(roi_w0 + h_width)],

            [0, 0]

        ])



        # Pad dst with zeros to make it the same shape as image.

        dst_padded=tf.pad(dst, paddings, "CONSTANT")

        # Create a boolean mask with zeros at the pixels of

        # the augmentation segment and ones everywhere else

        mask_img=tf.pad(tf.ones_like(dst), paddings, "CONSTANT")

        mask_img=~tf.cast(mask_img, dtype=tf.bool)

        # Make a hole in the original image at the location

        # of the augmentation segment

        img_hole=tf.multiply(img, tf.cast(mask_img, dtype=tf.float32))

        # Inserting the augmentation segment in place of the hole

        img=tf.add(img_hole, dst_padded)

        

    return {'inp1': img, 'inp2':  data['inp2']}, label
def get_training_dataset(dataset):

    # horizontal and vertical random flips

    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)

    # advanced hair augmentation

    dataset = dataset.map(hair_aug, num_parallel_calls=AUTO)

    # the training dataset must repeat for several epochs

    dataset = dataset.repeat()

    dataset = dataset.shuffle(512)

    dataset = dataset.batch(BATCH_SIZE)

    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO)

    

    return dataset
# numpy and matplotlib defaults

np.set_printoptions(threshold=15, linewidth=80)
def batch_to_numpy_images_and_labels(databatch, ds='train'):

    if ds=='train':

        data, labels = databatch

        numpy_images = data['inp1'].numpy()

        numpy_labels = labels.numpy()

    else:

        data = databatch

        numpy_images = data['inp1'].numpy()

        numpy_labels = [None for _ in enumerate(numpy_images)]



    # If no labels, only image IDs, return None for labels (this is the case for test data)

    return numpy_images, numpy_labels
def title_from_label_and_target(label, correct_label):

    if correct_label is None:

        return CLASSES[label], True

    correct = (label == correct_label)

    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" 

                                if not correct else '', 

                                CLASSES[correct_label] if not correct else ''), correct
def display_one_image(image, title, subplot, red=False, titlesize=16):

    plt.subplot(*subplot)

    plt.axis('off')

    plt.imshow(image)

    if len(title) > 0:

        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), 

                  color='red' if red else 'black', fontdict={'verticalalignment':'center'}, 

                  pad=int(titlesize/1.5)

                 )

    return (subplot[0], subplot[1], subplot[2]+1)
def display_batch_of_images(databatch, predictions=None, ds='train'):

    """This will work with:

    display_batch_of_images(images)

    display_batch_of_images(images, predictions)

    display_batch_of_images((images, labels))

    display_batch_of_images((images, labels), predictions)

    """

    # data

    images, labels = batch_to_numpy_images_and_labels(databatch, ds=ds)

    if labels is None:

        labels = [None for _ in enumerate(images)]

        

    # auto-squaring: this will drop data that does  

    # not fit into square or square-ish rectangle

    rows = int(math.sqrt(len(images)))

    cols = len(images)//rows

        

    # size and spacing

    FIGSIZE = 13.0

    SPACING = 0.1

    subplot=(rows,cols,1)

    if rows < cols:

        plt.figure(figsize=(FIGSIZE,FIGSIZE/cols*rows))

    else:

        plt.figure(figsize=(FIGSIZE/rows*cols,FIGSIZE))

    

    # display

    for i, (image, label) in enumerate(zip(images[:rows*cols], labels[:rows*cols])):

        title = '' if label is None else CLASSES[label]

        correct = True

        if predictions is not None:

            title, correct = title_from_label_and_target(predictions[i], label)

        # magic formula tested to work from 1x1 to 10x10 images

        dynamic_titlesize = FIGSIZE*SPACING/max(rows,cols)*40+3

        subplot = display_one_image(image, title, subplot, 

                                     not correct, titlesize=dynamic_titlesize)

    

    #layout

    plt.tight_layout()

    if label is None and predictions is None:

        plt.subplots_adjust(wspace=0, hspace=0)

    else:

        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)

    plt.show()
# Peek at training data



training_dataset = load_dataset(TRAIN_FNAMES['fold_2'])

training_dataset = training_dataset.map(setup_input, num_parallel_calls=AUTO)   

training_dataset = get_training_dataset(training_dataset)

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
%%time



# run this cell again for next set of images

display_batch_of_images(next(train_batch))
ROT_ = 180.0

SHR_ = 2.0

HZOOM_ = 8.0

WZOOM_ = 8.0

HSHIFT_ = 8.0

WSHIFT_ = 8.0
def get_batch_transformatioin_matrix(rotation, shear, 

                                     height_zoom, width_zoom, 

                                     height_shift, width_shift):

  

    """Returns a tf.Tensor of shape (batch_size, 3, 3) with each element along the 1st axis being

       an image transformation matrix (which transforms indicies).



    Args:

        rotation: 1-D Tensor with shape [batch_size].

        shear: 1-D Tensor with shape [batch_size].

        height_zoom: 1-D Tensor with shape [batch_size].

        width_zoom: 1-D Tensor with shape [batch_size].

        height_shift: 1-D Tensor with shape [batch_size].

        width_shift: 1-D Tensor with shape [batch_size].

        

    Returns:

        A 3-D Tensor with shape [batch_size, 3, 3].

    """    



    # A trick to get batch_size

    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(rotation)), tf.int64)    

    

    # CONVERT DEGREES TO RADIANS

    rotation = tf.constant(math.pi) * rotation / 180.0

    shear = tf.constant(math.pi) * shear / 180.0



    # shape = (batch_size,)

    one = tf.ones_like(rotation, dtype=tf.float32)

    zero = tf.zeros_like(rotation, dtype=tf.float32)

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation) # shape = (batch_size,)

    s1 = tf.math.sin(rotation) # shape = (batch_size,)



    # Intermediate matrix for rotation, shape = (9, batch_size) 

    rotation_matrix_temp = tf.stack([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0)

    # shape = (batch_size, 9)

    rotation_matrix_temp = tf.transpose(rotation_matrix_temp)

    # Fianl rotation matrix, shape = (batch_size, 3, 3)

    rotation_matrix = tf.reshape(rotation_matrix_temp, shape=(batch_size, 3, 3))

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear) # shape = (batch_size,)

    s2 = tf.math.sin(shear) # shape = (batch_size,)

    

    # Intermediate matrix for shear, shape = (9, batch_size) 

    shear_matrix_temp = tf.stack([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0)

    # shape = (batch_size, 9)

    shear_matrix_temp = tf.transpose(shear_matrix_temp)

    # Fianl shear matrix, shape = (batch_size, 3, 3)

    shear_matrix = tf.reshape(shear_matrix_temp, shape=(batch_size, 3, 3))    

    

    # ZOOM MATRIX

    

    # Intermediate matrix for zoom, shape = (9, batch_size) 

    zoom_matrix_temp = tf.stack([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0)

    # shape = (batch_size, 9)

    zoom_matrix_temp = tf.transpose(zoom_matrix_temp)

    # Fianl zoom matrix, shape = (batch_size, 3, 3)

    zoom_matrix = tf.reshape(zoom_matrix_temp, shape=(batch_size, 3, 3))

    

    # SHIFT MATRIX

    

    # Intermediate matrix for shift, shape = (9, batch_size) 

    shift_matrix_temp = tf.stack([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0)

    # shape = (batch_size, 9)

    shift_matrix_temp = tf.transpose(shift_matrix_temp)

    # Fianl shift matrix, shape = (batch_size, 3, 3)

    shift_matrix = tf.reshape(shift_matrix_temp, shape=(batch_size, 3, 3))    

        

    return tf.linalg.matmul(tf.linalg.matmul(rotation_matrix, shear_matrix), tf.linalg.matmul(zoom_matrix, shift_matrix))
def affine_aug(data, label):

    """Returns a tf.Tensor of the same shape as `images`, represented a batch of randomly transformed images.



    Args:

        images: 4-D Tensor with shape (batch_size, width, hight, depth).

            Currently, `depth` can only be 3.

        

    Returns:

        A 4-D Tensor with the same shape as `images`.

    """ 

    images=data['inp1']



    # input `images`: a batch of images [batch_size, dim, dim, 3]

    # output: images randomly rotated, sheared, zoomed, and shifted

    DIM = images.shape[1]

    XDIM = DIM % 2  # fix for size 331

    

    # A trick to get batch_size

    batch_size = tf.cast(tf.reduce_sum(tf.ones_like(images)) / (images.shape[1] * images.shape[2] * images.shape[3]), tf.int64)

    

    rot = ROT_ * tf.random.normal([batch_size], dtype='float32')

    shr = SHR_ * tf.random.normal([batch_size], dtype='float32') 

    h_zoom = 1.0 #+ tf.random.normal([batch_size], dtype='float32') / HZOOM_

    w_zoom = 1.0 #+ tf.random.normal([batch_size], dtype='float32') / WZOOM_

    h_shift = HSHIFT_ * tf.random.normal([batch_size], dtype='float32') 

    w_shift = WSHIFT_ * tf.random.normal([batch_size], dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    # shape = (batch_size, 3, 3)

    m = get_batch_transformatioin_matrix(rot, shr, h_zoom, w_zoom, h_shift, w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)  # shape = (DIM * DIM,)

    y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])  # shape = (DIM * DIM,)

    z = tf.ones([DIM * DIM], dtype='int32')  # shape = (DIM * DIM,)

    idx = tf.stack([x, y, z])  # shape = (3, DIM * DIM)

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = tf.linalg.matmul(m, tf.cast(idx, dtype='float32'))  # shape = (batch_size, 3, DIM ** 2)

    idx2 = K.cast(idx2, dtype='int32')  # shape = (batch_size, 3, DIM ** 2)

    idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)  # shape = (batch_size, 3, DIM ** 2)

    

    # FIND ORIGIN PIXEL VALUES

    # shape = (batch_size, 2, DIM ** 2)

    idx3 = tf.stack([DIM // 2 - idx2[:, 0, ], DIM // 2 - 1 + idx2[:, 1, ]], axis=1)  

    

    # shape = (batch_size, DIM ** 2, 3)

    d = tf.gather_nd(images, tf.transpose(idx3, perm=[0, 2, 1]), batch_dims=1)

        

    # shape = (batch_size, DIM, DIM, 3)

    new_images = tf.reshape(d, (batch_size, DIM, DIM, 3))



    return {'inp1': new_images, 'inp2':  data['inp2']}, label
def flip_aug(data, label):

    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement 

    # in the next function (below), this happens essentially for free on TPU. 

    # Data pipeline code is executed on the "CPU" part

    # of the TPU while the TPU itself is computing gradients.

    images = tf.image.random_flip_left_right(data['inp1'])

    images = tf.image.random_flip_up_down(images)

    

    return {'inp1': images, 'inp2':  data['inp2']}, label
def get_training_dataset(dataset, do_flip_aug=True, do_affine_aug=True, do_hair_aug=True):



    if do_hair_aug:

      # advanced hair augmentation

      dataset = dataset.map(hair_aug, num_parallel_calls=AUTO)

  

    # the training dataset must repeat for several epochs    

    dataset = dataset.repeat()

    dataset = dataset.shuffle(2048)

    dataset = dataset.batch(BATCH_SIZE)



    if do_flip_aug:

      # horizontal and vertical random flips

      dataset = dataset.map(flip_aug, num_parallel_calls=AUTO)



    if do_affine_aug:

      # affine transformations

      dataset = dataset.map(affine_aug, num_parallel_calls=AUTO)



    # prefetch next batch while training (autotune prefetch buffer size)

    dataset = dataset.prefetch(AUTO)

  

    return dataset
# Peek at training data



training_dataset = load_dataset(TRAIN_FNAMES['fold_4'])

training_dataset = training_dataset.map(setup_input, num_parallel_calls=AUTO)   

training_dataset = get_training_dataset(training_dataset)

training_dataset = training_dataset.unbatch().batch(20)

train_batch = iter(training_dataset)
%%time



# run this cell again for next set of images

display_batch_of_images(next(train_batch))
%%time



row = 3; col = 4;



training_dataset = load_dataset(TRAIN_FNAMES['fold_1'])

training_dataset = training_dataset.map(setup_input, num_parallel_calls=AUTO)   



all_elements = get_training_dataset(training_dataset,

                                    do_flip_aug=False, 

                                    do_affine_aug=False, 

                                    do_hair_aug=False).unbatch()



one_element = tf.data.Dataset.from_tensors( next(iter(all_elements)) )

augmented_element = one_element.repeat().map(hair_aug).batch(row*col).map(flip_aug).map(affine_aug)

for (data, label) in augmented_element:

    plt.figure(figsize=(15,int(15*row/col)))

    for j in range(row*col):

        plt.subplot(row,col,j+1)

        plt.axis('off')

        plt.imshow(data['inp1'][j,])

    plt.show()

    break