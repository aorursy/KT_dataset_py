import time

start_time = time.time()

import os; import gc; import math

gc.enable()

import skimage.io

import numpy as np

import pandas as pd

import cv2

import PIL.Image

import tensorflow as tf

import albumentations

import matplotlib.pyplot as plt

%matplotlib inline

%config InlineBackend.figure_format = 'retina'

from tqdm import tqdm_notebook as tqdm

print("Libraries Imported.! Time step {:.2f}".format(time.time()-start_time))
start_time = time.time()

data_dir = '../input/prostate-cancer-grade-assessment'

df_train = pd.read_csv(os.path.join(data_dir, 'train.csv'))

image_folder = os.path.join(data_dir, 'train_images')



tile_size = 256

image_size = 256

n_tiles = 36

num_workers = 4

print("Configuration Done.! Time step {:.2f}".format(time.time()-start_time))
# Simple Augmentation

transforms_train = albumentations.Compose([

    albumentations.Transpose(p=0.5),

    albumentations.VerticalFlip(p=0.5),

    albumentations.HorizontalFlip(p=0.5),

])

transforms_val = albumentations.Compose([])



# Heavy Augmentation

# transforms_train = albumentations.Compose([

#     albumentations.OneOf([

#         albumentations.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,

#                                         rotate_limit=15,

#                                         border_mode=cv2.BORDER_CONSTANT, value=0),

#         albumentations.OpticalDistortion(distort_limit=0.11, shift_limit=0.15,

#                                          border_mode=cv2.BORDER_CONSTANT,

#                                          value=0),

#         albumentations.NoOp(),

#     ]),

#     albumentations.OneOf([

#         albumentations.RandomGamma(gamma_limit=(50, 150)),

#         albumentations.NoOp()

#     ]),

#     albumentations.OneOf([

#         albumentations.Blur(),

#         albumentations.Transpose(),

#         albumentations.ElasticTransform(),

#         albumentations.GridDistortion(),

#         albumentations.CoarseDropout(),

#         albumentations.NoOp()

#     ]),

#     albumentations.OneOf([

#         albumentations.HorizontalFlip(),

#         albumentations.VerticalFlip(),

#         albumentations.NoOp()

#     ])     

# ])
def get_tiles(img, mode=0):

    result = []

    h, w, c = img.shape

    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)

    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)



    img2 = np.pad(img,[[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2,pad_w - pad_w//2], [0,0]], constant_values=255)

    img3 = img2.reshape(

        img2.shape[0] // tile_size,

        tile_size,

        img2.shape[1] // tile_size,

        tile_size,

        3

    )

    img3 = img3.transpose(0,2,1,3,4).reshape(-1, tile_size, tile_size,3)

    n_tiles_with_info = (img3.reshape(img3.shape[0],-1).sum(1) < tile_size ** 2 * 3 * 255).sum()

    if len(img) < n_tiles:

        img3 = np.pad(img3,[[0,N-len(img3)],[0,0],[0,0],[0,0]], constant_values=255)

    idxs = np.argsort(img3.reshape(img3.shape[0],-1).sum(-1))[:n_tiles]

    img3 = img3[idxs]

    for i in range(len(img3)):

        result.append({'img':img3[i], 'idx':i})

    return result, n_tiles_with_info >= n_tiles
start_1, end_1 = 0, 500

start_2, end_2 = 500, 1000

shard_1, shard_2 = 0, 1
start_time = time.time()

save_dir = "kaggle/train_images/"

os.makedirs(save_dir, exist_ok=True)



def covert_tiles(start_records, end_records):

    # select the number of data samples here



    for i in tqdm(range(start_records, end_records)):



        row = df_train.iloc[i]

        img_id = row.image_id



        save_path = save_dir + img_id + '.png'



        tiff_file = os.path.join(image_folder, f'{img_id}.tiff')

        image = skimage.io.MultiImage(tiff_file)[1]



        tiles, OK = get_tiles(image)



        idxes = list(range(n_tiles))

        n_row_tiles = int(np.sqrt(n_tiles))

        images = np.zeros((image_size * n_row_tiles, image_size * n_row_tiles, 3))



        for h in range(n_row_tiles):

            for w in range(n_row_tiles):

                i = h * n_row_tiles + w

                if len(tiles) > idxes[i]:

                    this_img = tiles[idxes[i]]['img']

                else:

                    this_img = np.ones((image_size, image_size, 3)).astype(np.uint8) * 255

                this_img = 255 - this_img

                if transforms_train is not None:

                    # apply augmentation

                    this_img = transforms_train(image=this_img)['image']

                h1 = h * image_size

                w1 = w * image_size

                images[h1:h1+image_size, w1:w1+image_size] = this_img

        if transforms_train is not None:

            images = transforms_train(image=images)['image']

        images = images.astype(np.float32)



        #images = cv2.resize(images, (512, 512))



        cv2.imwrite(save_path, images)

    print("Coversion of Image to Tiles Complete.! Time step {:.2f}".format(time.time()-start_time))
data_root = "kaggle/train_images/"

tf_record_dir = "kaggle/tfrecord_data/"



def get_paths_and_labels(first_index=0, last_index=1000):

    # utility function to return image and label

    first_index=first_index

    last_index=last_index

    return [(os.path.join(save_dir, df_train.iloc[i].image_id+".png"), df_train.iloc[i].isup_grade)  for i in range(len(df_train.iloc[first_index:last_index]))]
def write_to_tfrecords(num, start, end):

    start_time = time.time()

    

    record_dir = tf_record_dir



    if os.path.exists(record_dir):

        return

    os.makedirs(record_dir, exist_ok=True)



    print("Converting images to TFRecords...")

    

    # number of records per shard

    records_per_shard = 250

    shard_number = num

    # start index

    start_index = start

    # end index

    end_index = end

    path_template = os.path.join(record_dir, "shard_{0:04d}.tfrecords")

    writer = tf.io.TFRecordWriter(path_template.format(shard_number))

    

    for i, (image_path, label) in enumerate(get_paths_and_labels(first_index=start_index, last_index=end_index)):

        if i and not (i % records_per_shard):

            shard_number += 1

            writer.close()

            writer = tf.io.TFRecordWriter(path_template.format(shard_number))



        with open(image_path, "rb") as f:

            image_bytes = f.read()



        record_bytes = tf.train.Example(features=tf.train.Features(feature={

                            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),

                            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

                        })).SerializeToString()



        writer.write(record_bytes)



    writer.close()

    print("TFRecord conversion complete.")

    print('Conversion to TF-Records is Complete.! Time step {:.2f}'.format(time.time()-start_time))
covert_tiles(start_1, end_1)

write_to_tfrecords(shard_1, start_1, end_1)
import os, shutil

folder = 'kaggle/train_images/'

for filename in os.listdir(folder):

    file_path = os.path.join(folder, filename)

    try:

        if os.path.isfile(file_path) or os.path.islink(file_path):

            os.unlink(file_path)

        elif os.path.isdir(file_path):

            shutil.rmtree(file_path)

    except Exception as e:

        print('Failed to delete %s. Reason: %s' % (file_path, e))

os.removedirs('kaggle/train_images/')
IMAGE_SIZE = [1536, 1536]

def decode_image(image_data):

    image = tf.image.decode_png(image_data, channels=3)

    image = tf.cast(image, tf.float32)  

    return image



def read_labeled_tfrecord(record):

    record = tf.io.parse_single_example(record, RECORD_SCHEMA)

    image = decode_image(record['image'])

    label = tf.cast(record['label'], tf.int32)

    return image, label 



RECORD_PATTERN = os.path.join('kaggle/tfrecord_data/', "*.tfrecords")

RECORD_SCHEMA = {

    "image": tf.io.FixedLenFeature([], dtype=tf.string),

    "label": tf.io.FixedLenFeature([1], dtype=tf.int64)

}
dataset = tf.data.Dataset.list_files(RECORD_PATTERN)

dataset = dataset.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.shuffle(100)

dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)

dataset = dataset.batch(500, drop_remainder=True)

dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)