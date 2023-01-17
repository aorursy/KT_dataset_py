import numpy as np 

import pandas as pd

import tensorflow as tf

from tqdm.notebook import tqdm
all_images_meta = pd.read_csv('../input/landmark-retrieval-2020/train.csv')
images_meta_sample = all_images_meta.groupby('landmark_id').head(2).reset_index(drop=True)
images_meta_sample.head(10)
images_meta_sample.info()
landmark_groups = all_images_meta.groupby('landmark_id')
landmark_group = landmark_groups.get_group(1)

landmark_group
def get_positive(anchor_landmark_id):    

    landmark_group = landmark_groups.get_group(anchor_landmark_id)

    indexes = landmark_group.index.values

    

    rand_index = np.random.choice(indexes)

    pos_img_id = landmark_group.loc[rand_index].id

        

    return pos_img_id
images_meta_sample['positive_id'] = images_meta_sample['landmark_id'].apply(get_positive)
def get_negative(landmark_id):        

    indexes = images_meta_sample.index.values

    

    for i in range(len(images_meta_sample)):

        rand_index = np.random.choice(indexes)

        

        neg_img_id = images_meta_sample.loc[rand_index].id

        neg_landmark_id = images_meta_sample.loc[rand_index].landmark_id

        

        if neg_landmark_id != landmark_id:

            return neg_img_id

    

    return neg_img_id
images_meta_sample['negative_id'] = images_meta_sample['landmark_id'].apply(get_negative)
images_meta_sample = images_meta_sample.rename({'id': 'anchor_id'}, axis='columns')
images_meta_sample.head()
def _bytes_feature(value):

    if isinstance(value, type(tf.constant(0))):

        value = value.numpy() 

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def get_image(img_id):

    chars = tf.strings.bytes_split(img_id)

    dir_1, dir_2, dir_3 = chars[0], chars[1], chars[2]

    

    path = '../input/landmark-retrieval-2020/train/' + dir_1 + '/' + dir_2 + '/' + dir_3 + '/' + img_id + '.jpg'

    image = tf.io.read_file(path)

    

    image = tf.image.decode_jpeg(image)

    image = tf.image.resize(image, size=(128, 128), method='nearest')

    image = tf.image.convert_image_dtype(image, tf.uint8)

    

    image = tf.image.encode_jpeg(image, quality=94, optimize_size=True)

    

    return image
def serialize_example(example):    

    anchor_img = get_image(example.anchor_id)

    positive_img = get_image(example.positive_id)

    negative_img = get_image(example.negative_id)

    

    feature = {

        'anchor_img': _bytes_feature(anchor_img),

        'positive_img': _bytes_feature(positive_img),

        'negative_img': _bytes_feature(negative_img),

    }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()
image_indexes = images_meta_sample.index.values
file_size = len(image_indexes) // 15

file_count = len(image_indexes) // file_size + int(len(image_indexes) % file_size != 0)
def write_tfrecord_file(file_index, file_size, image_indexes):

    with tf.io.TFRecordWriter('train%.2i.tfrec'%(file_index)) as writer:

        start = file_size * file_index

        end = file_size * (file_index + 1)

        

        for i in tqdm(image_indexes[start:end]):



            example = serialize_example(

                images_meta_sample.loc[i]

            )

            

            writer.write(example)
for file_index in range(file_count):

    print('Writing TFRecord %i of %i...'%(file_index, file_count))

    write_tfrecord_file(file_index, file_size, image_indexes)