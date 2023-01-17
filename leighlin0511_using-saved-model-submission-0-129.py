from __future__ import absolute_import

from __future__ import division

from __future__ import print_function



import os

import time



import numpy as np

from skimage.io import imread

import pandas as pd



import tensorflow as tf



import sys

import matplotlib.pyplot as plt



%matplotlib inline
#DEFAULT_BASE_PATH = 'https://storage.cloud.google.com/rxrx1-us-central1'

DEFAULT_BASE_PATH = '../input'

DEFAULT_METADATA_BASE_PATH = os.path.join(DEFAULT_BASE_PATH, 'recursion-cellular-image-classification')

DEFAULT_TRAIN_BASE_PATH = os.path.join(DEFAULT_METADATA_BASE_PATH, 'train')

DEFAULT_TEST_BASE_PATH = os.path.join(DEFAULT_METADATA_BASE_PATH, 'test')

DEFAULT_CHANNELS = (1, 2, 3, 4, 5, 6)

saved_model_path= DEFAULT_BASE_PATH+"/leighlin-recur-cell-saved-model/saved_model/saved_model"
def image_path(experiment,

               plate,

               well,

               site,

               channel,

               base_path=DEFAULT_TRAIN_BASE_PATH):

    

    

    return os.path.join(base_path, experiment, "Plate{}".format(plate),

                        "{}_s{}_w{}.png".format(well, site, channel))
#test/HEPG2-08/Plate1/B02_s1_w1.png

image_path01 = image_path("HUVEC-01","1","K17",'1','3')

print(image_path01)
def load_image(image_path):

    with tf.io.gfile.GFile(image_path, 'rb') as f:

        return imread(f, format='png')
def load_images_as_matrix(image_paths, dtype=np.uint8):

    n_channels = len(image_paths)



    data = np.ndarray(shape=(512, 512, n_channels), dtype=dtype)



    for ix, img_path in enumerate(image_paths):

        data[:, :, ix] = load_image(img_path)



    return data
def load_site(experiment,

              plate,

              well,

              site,

              channels=DEFAULT_CHANNELS,

              base_path=DEFAULT_TRAIN_BASE_PATH):

   

    channel_paths = [

        image_path(

            experiment, plate, well, site, c, base_path=base_path)

        for c in channels

    ]

    return load_images_as_matrix(channel_paths)
image_shape = [512, 512, 6]

# The mean and stds for each of the channels

GLOBAL_PIXEL_STATS = (np.array([6.74696984, 14.74640167, 10.51260864,

                                10.45369445,  5.49959796, 9.81545561]),

                       np.array([7.95876312, 12.17305868, 5.86172946,

                                 7.83451711, 4.701167, 5.43130431]))

def process_image(image, pixel_stats=GLOBAL_PIXEL_STATS):    

    if pixel_stats is not None:

        mean, std = pixel_stats

        image = (image - mean) / std

        

    image = image[np.newaxis, :]

    return image
def _tf_read_csv(path):

    with tf.io.gfile.GFile(path, 'rb') as f:

        return pd.read_csv(f)



def _load_dataset(base_path, dataset, include_controls=True):

    df = _tf_read_csv(os.path.join(base_path, dataset + '.csv'))

    if include_controls:

        controls = _tf_read_csv(

            os.path.join(base_path, dataset + '_controls.csv'))

        df['well_type'] = 'treatment'

        df = pd.concat([controls, df], sort=True)

    df['cell_type'] = df.experiment.str.split("-").apply(lambda a: a[0])

    df['dataset'] = dataset

    return df



def load_test_metadata(base_path=DEFAULT_METADATA_BASE_PATH,

                     include_controls=False):

    df = _load_dataset(base_path, "test", include_controls=include_controls)

    return df
metadata_test = load_test_metadata()

metadata_test.head()
def get_id_code(experiment,plate,well):

    return "{}_{}_{}".format(experiment,plate,well)



#HEPG2-08_1_B03

print(get_id_code("HEPG2-08","1","B03"))
start_time = time.time()

#df = pd.DataFrame(columns=('id_code','sirna','prob') )

df = pd.DataFrame(columns=('id_code','sirna') )



with tf.Session(graph=tf.Graph()) as sess:

    signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    input_key = 'feature'

    output_key_classes = 'classes'

    output_key_p = 'probabilities'

    meta_graph_def = tf.saved_model.loader.load(

           sess,

          [tf.saved_model.tag_constants.SERVING],

          saved_model_path)

    signature = meta_graph_def.signature_def

    

    x_tensor_name = signature[signature_key].inputs[input_key].name

    print(signature[signature_key].outputs)

    y_classes_tensor_name = signature[signature_key].outputs[output_key_classes].name

    #y_p_tensor_name = signature[signature_key].outputs[output_key_p].name

    

    x = sess.graph.get_tensor_by_name(x_tensor_name)

    y_classes = sess.graph.get_tensor_by_name(y_classes_tensor_name)    

    #y_p = sess.graph.get_tensor_by_name(y_p_tensor_name)

        

    for index, row in metadata_test.iterrows():

        #experiment  plate well cell_type

        experiment = row["experiment"]

        plate = row["plate"]

        well = row["well"]

        cell_type = row["cell_type"]

        #I only test for site 1, you should include site 2 data as well

        image_raw = load_site(experiment, plate, well, 1, base_path=DEFAULT_TEST_BASE_PATH)

        

        image = process_image(image_raw)

        #y_out = sess.run([y_classes, y_p], {x: image})

        y_out = sess.run(y_classes, {x: image})

        #sirna = y_out[0][0]

        sirna = y_out[0]

        #prob = y_out[1][0,sirna]

        

        id_code = get_id_code(experiment,plate,well)

        #df.loc[index] = [id_code, sirna, prob]

        df.loc[index] = [id_code, sirna]

        if index % 100 == 0:            

            elapsed_time = time.time() - start_time

            print("processed row {} with {} minutes".format(index, (elapsed_time/60)))   

        #Only test for first 200. Remove this code for the whole dataset

        if index>20000:

            break
submission = df

submission.to_csv('submission.csv', index=False)