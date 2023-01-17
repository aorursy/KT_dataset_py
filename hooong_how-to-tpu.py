import tensorflow as tf 

import tensorflow.keras as tfk

import numpy as np 

import glob

import pandas as pd 

from skimage import io, transform

from tqdm.notebook import tqdm









metadata = pd.read_csv('/kaggle/input/deepfake-first-frames-and-labels/train_df.csv')

metadata['path'] = '/kaggle/input/deepfake-first-frames-and-labels/reduced_train_stills/first_stills/' + metadata['index']

metadata['path'] = metadata.path.str.replace('.mp4', '.jpg')

metadata.head(5)
#bytes feature is the easiest feature type to use. It is also extremely inefficient in terms of storage. This one is configured for images

#you can save on storage by compressing your tensor into a jpg byte stream 

#you can also compress the whole tfrecord file, but I haven't tried that yet 



def _bytes_feature(value):

    value = tf.io.serialize_tensor(tf.convert_to_tensor(value, dtype = tf.uint8))

    if tf.executing_eagerly: value = value.numpy()

    else: value = value.eval() 

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



# this method takes an image and label and converts it to a bytes feature then serializes it 

# you can customize this and make it more efficient.

def serialize_example(image, label):

    

    feature = {'image' : _bytes_feature(image)

              ,'label' : _bytes_feature(label)

              }

    

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))

    return example_proto.SerializeToString()



#this feature map is very important. It allows the TPU to parse the tfrecord and will be used later 

feature_map = {'image' :tf.io.FixedLenFeature([], tf.string, default_value='')

              ,'label' : tf.io.FixedLenFeature([], tf.string, default_value='')

              }



def get_xy(metadatarow):

    x = io.imread(metadatarow.path)

    

    #you can transform your image here 

    x = transform.resize(x, (256,256))

    

    #please remember that fakes = 1 and reals = 0

    y = 1

    if metadatarow.label == 'REAL': y = 0

    return x, y 

    

#these record files are really small. I recommend using a sample number that gets you to ~150mb per tfrecord file

samples = 64





#you can generate you own filenames 

filenames = ['test1.tfrecord', 'test2.tfrecord']



for filename in filenames:     

    with tf.io.TFRecordWriter(filename) as writer:         

        for i, row in  tqdm(metadata.sample(n = samples).iterrows(), total = samples): 

            X, y = get_xy(row)

            example = serialize_example(X, y)

            writer.write(example)

    


#parsing an example requires a feature map, as defined earlier

#remember that whatever things you did to get the X, y pairs into the tfrecord, you will need to undo them so you get your xy pairs back 

def parse_example(stringlike): 

    parsed = tf.io.parse_single_example(stringlike, feature_map)

    return (tf.io.parse_tensor(parsed['image'], out_type = tf.uint8)

            , tf.io.parse_tensor(parsed['label'], out_type = tf.uint8))



#tfrecords are streams of data, so you will need to remind the TPU what shape the data is in 

def force_shape(tx, ty): 

    xshape = (256, 256, 3)

    yshape = (1, 1)

    tx = tf.reshape(tx, xshape)

    ty = tf.reshape(ty, yshape)

    return tx, ty



#this is a helper function that I made that allows you to wrap a mappable function, so that you can pass parameters to it 

def get_mappable(func, **kwargs):

    #sets kwargs for a function that should map across a dataset

    default_kwargs = kwargs

    def map_wrap(*args):

        return func(*args, **default_kwargs)

    

    return map_wrap



def force_shape_parametered(tx, ty, xshape, yshape): 

    tx = tf.reshape(tx, xshape)

    ty = tf.reshape(ty, yshape)

    return tx, ty



#VERY IMPORTANT. OMG THIS WAS SO HARD TO FIND. TPUs don't take your regular data types. they are very picky. For this comp, int32 is fine. 

def force_int32(tx, ty): 

    tx = tf.dtypes.cast(tx, tf.int32)

    ty = tf.dtypes.cast(ty, tf.int32)

    return tx, ty 



#read the record. You should upload it to a cloud storage bucket

#storage bucket paths look like gs://bucketname/something/something/test1.tfrecord 

#you can pass a lot of filenames for optimal performance 

dataset = tf.data.TFRecordDataset(filenames)



#the map method applies a function across the whole dataset (technically it only applies as it reads the data)

dataset = dataset.map(parse_example)

dataset = dataset.map(force_int32)



# a basic force shape 

# dataset = dataset.map(force_shape)



# a parametered force shape, so you can define the parameters 

dataset = dataset.map(get_mappable(force_shape_parametered, xshape = [256, 256, 3], yshape = [1]))





#you need to batch the dataset. remember that this is the batch size passed to the tpu. For this dataset, 64 per TPU core is fine. You can probably do 128. 

#remember to drop the remainder or else you will get a silly error that stops your tpu, but I believe this drop remainder is automatic now 

dataset = dataset.batch(16, drop_remainder = True)

#prefetch also seems to be automatically tuned 

dataset = dataset.prefetch(2)

#don't forget to repeat your datasets or else they will be exhausted

dataset = dataset.repeat()



# unfortuantely, you can only read tfrecord when it is in the cloud. below is a for loop that I use to verify the shape, before passing it on 

# you should run this loop in TPU off mode when generating data 



# for record in dataset.take(1): 

#     print('batch shape', record[0].shape, record[1].shape)    

#     break
# detect and init the TPU

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

strategy = tf.distribute.experimental.TPUStrategy(tpu)





    

    
learning_rate = 0.01        



with strategy.scope():

    

    model = tfk.models.Sequential()

    

    pt = tfk.applications.resnet.ResNet50 



    ptmod = pt(include_top=False

                , weights='imagenet'

                , input_tensor=None

                , input_shape=(256, 256, 3)

                , pooling = 'avg')





    model.add(ptmod)

    model.add(tfk.layers.Dropout(rate = 0.5))

    model.add(tfk.layers.Dense(1))

    model.add(tfk.layers.Activation('sigmoid'))

    

    optimizer = tfk.optimizers.Adam(learning_rate = learning_rate)





    model.compile(loss='binary_crossentropy', optimizer=optimizer)



    model.summary()





# for an unknown reason, the kernel dies at this point. 

# this could be because it is trying access a local tfrecord. 



# model.fit(dataset,

#           epochs = 10, 

#           steps_per_epoch = 10, 

# #               validation_data = val_ds, 

# #               validation_steps = val_steps//strategy.num_replicas_in_sync,

# #               callbacks = callbacks 

#          )
