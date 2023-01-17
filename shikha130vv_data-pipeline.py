!pip install -q efficientnet

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.core.display import display, HTML

import tensorflow as tf

import matplotlib.pyplot as plt

from kaggle_datasets import KaggleDatasets

import tensorflow_addons as tfa

import tensorflow.keras.backend as K

import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

AUTO = tf.data.experimental.AUTOTUNE



import tensorflow as tf, re, math

def get_strategy():

    # Detect hardware, return appropriate distribution strategy

    gpu = ""

    try:

        # TPU detection. No parameters necessary if TPU_NAME environment variable is

        # set: this is always the case on Kaggle.

        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

        print('Running on TPU ', tpu.master())

        

        # GPU detection

        

    except ValueError:

        tpu = None

        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        gpu = tf.config.list_physical_devices("GPU")

        if len(gpu) == 1:

            print('Running on GPU ', gpu)

    if tpu:

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        tf.config.experimental_connect_to_cluster(tpu)

        tf.tpu.experimental.initialize_tpu_system(tpu)

        strategy = tf.distribute.experimental.TPUStrategy(tpu)

        GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

    elif len(gpu) == 1:

        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

        tf.config.optimizer.set_experimental_options({"auto_mixed_precision":True})

        GCS_PATH = "/kaggle/input/siim-isic-melanoma-classification/"

    else:

        # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

        strategy = tf.distribute.get_strategy()

        GCS_PATH = "/kaggle/input/siim-isic-melanoma-classification/"



    print("REPLICAS: ", strategy.num_replicas_in_sync)

    base_dir = "/kaggle/input/siim-isic-melanoma-classification/"

    return strategy, GCS_PATH, base_dir



strategy,GCS_PATH, base_dir = get_strategy()
mm = 1; rr= 1

f = open("log-{mm}-{rr}.txt","a")

f.write("LR, Val Score")

f.close()
print("Content of base directory: {}".format(",".join(os.listdir(base_dir))))
train_data = pd.read_csv("/kaggle/input/siim-isic-melanoma-classification/train.csv")

print("Num Rows in train_data: {}".format(train_data.shape[0]))

print(display(HTML(train_data.head(1).to_html())))

print(train_data["target"].value_counts())
train_data["anatom_site_general_challenge"].fillna("Unknown", inplace=True)

group_data = train_data.groupby(["anatom_site_general_challenge"])["benign_malignant"].value_counts().unstack(-1)

group_data["perc_malignant"] = np.round((group_data["malignant"] * 100) /(group_data["benign"] + group_data["malignant"]),2)

group_data
IMAGE_SIZE = [224,224]

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):

    # returns 3x3 transformmatrix which transforms indicies

        

    # CONVERT DEGREES TO RADIANS

    rotation = math.pi * rotation / 180.

    shear = math.pi * shear / 180.

    

    # ROTATION MATRIX

    c1 = tf.math.cos(rotation)

    s1 = tf.math.sin(rotation)

    one = tf.constant([1],dtype='float32')

    zero = tf.constant([0],dtype='float32')

    rotation_matrix = tf.reshape( tf.concat([c1,s1,zero, -s1,c1,zero, zero,zero,one],axis=0),[3,3] )

        

    # SHEAR MATRIX

    c2 = tf.math.cos(shear)

    s2 = tf.math.sin(shear)

    shear_matrix = tf.reshape( tf.concat([one,s2,zero, zero,c2,zero, zero,zero,one],axis=0),[3,3] )    

    

    # ZOOM MATRIX

    zoom_matrix = tf.reshape( tf.concat([one/height_zoom,zero,zero, zero,one/width_zoom,zero, zero,zero,one],axis=0),[3,3] )

    

    # SHIFT MATRIX

    shift_matrix = tf.reshape( tf.concat([one,zero,height_shift, zero,one,width_shift, zero,zero,one],axis=0),[3,3] )

    

    return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))



def transform(image):

    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]

    # output - image randomly rotated, sheared, zoomed, and shifted

    DIM = IMAGE_SIZE[0]

    XDIM = DIM%2 #fix for size 331

    

    rot = 15. * tf.random.normal([1],dtype='float32')

    shr = 5. * tf.random.normal([1],dtype='float32') 

    h_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    w_zoom = 1.0 + tf.random.normal([1],dtype='float32')/10.

    h_shift = 16. * tf.random.normal([1],dtype='float32') 

    w_shift = 16. * tf.random.normal([1],dtype='float32') 

  

    # GET TRANSFORMATION MATRIX

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 



    # LIST DESTINATION PIXEL INDICES

    x = tf.repeat( tf.range(DIM//2,-DIM//2,-1), DIM )

    y = tf.tile( tf.range(-DIM//2,DIM//2),[DIM] )

    z = tf.ones([DIM*DIM],dtype='int32')

    idx = tf.stack( [x,y,z] )

    

    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS

    idx2 = K.dot(m,tf.cast(idx,dtype='float32'))

    idx2 = K.cast(idx2,dtype='int32')

    idx2 = K.clip(idx2,-DIM//2+XDIM+1,DIM//2)

    

    # FIND ORIGIN PIXEL VALUES           

    idx3 = tf.stack( [DIM//2-idx2[0,], DIM//2-1+idx2[1,]] )

    d = tf.gather_nd(image,tf.transpose(idx3))

        

    return tf.reshape(d,[DIM,DIM,3])



import tensorflow_probability as tfp



def gaussian_kernel(size: int,

                    mean: float,

                    std: float,

                   ):

    """Makes 2D gaussian Kernel for convolution."""



    d = tfp.distributions.Normal(mean, std)



    vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))



    gauss_kernel = tf.einsum('i,j->ij',

                                  vals,

                                  vals)



    return gauss_kernel / tf.reduce_sum(gauss_kernel)





def smoothing(img):

    gauss_kernel = gaussian_kernel(7,0.5,1)



    # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.

    gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]



    # Convolve.

    img1 = tf.nn.conv2d(tf.reshape(img[:,:,0], [1,224,224,1]), gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

    img2 = tf.nn.conv2d(tf.reshape(img[:,:,1], [1,224,224,1]), gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

    img3 = tf.nn.conv2d(tf.reshape(img[:,:,2], [1,224,224,1]), gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")

    img = tf.reshape(tf.concat([img1,img2,img3], axis=3), [224,224,3])

    return img
FEATURE_SET =  {

      

}





def parse_rec_train(data):           

    feature_set = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'target': tf.io.FixedLenFeature([], tf.int64)

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    return features



def parse_rec_validate(data):           

    feature_set = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'target': tf.io.FixedLenFeature([], tf.int64),

        'image_name': tf.io.FixedLenFeature([], tf.string)

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    return features



def parse_rec_test(data):           

    feature_set = {

        'image': tf.io.FixedLenFeature([], tf.string),

        'image_name': tf.io.FixedLenFeature([], tf.string)

    }

    features = tf.io.parse_single_example(data, features= feature_set )

    return features



seed = 42

def process_img(img):

    img = tf.image.decode_image(img)

    #img = tf.ensure_shape(img, (1024,1024,3))

    img = tf.ensure_shape(img, (224,224,3))

    img = tf.image.resize(img, [224,224])

    img = tf.ensure_shape(img, (224,224,3))

    #img = tf.keras.preprocessing.image.random_rotation(img, np.random.randint(360))

    img = float(img)/255.00

    return tf.cast(img, tf.float32)



def get_img_label(features):

    target = features["target"]

    features.pop("target")

    img = process_img(features["image"])

    return img, target



def aug(img):

    return transform(img)



def aug1(img):

    angle_list = [15, 30, 45, 60, 75, 90]

    img = tfa.image.rotate(img, angle_list[np.random.randint(6)])

    #img = tf.image.rot90(img,k=np.random.randint(4))

    img = tf.image.random_flip_left_right(img, seed=seed)

    img = tf.image.random_flip_up_down(img, seed=seed)

    return img



def aug_img_label(img, label):

    img = aug(img)

    return img, label



def aug_img(img):

    img = aug(img)

    return img



def get_img(features, label=None):

    img = process_img(features["image"])

    return img



def get_img_and_name(features, label=None):

    img = process_img(features["image"])

    image_name = features["image_name"]

    return img, image_name



def get_img_name(features):

    image_name = features["image_name"]

    return image_name
from sklearn.model_selection import train_test_split

tfrec_dir = base_dir + "tfrecords/"

tfrec_dir = base_dir + "../croppedskincancerimagestrain/"

tfrec_files = os.listdir(tfrec_dir)

tfrec_files_train = [GCS_PATH + "/tfrecords/" + file for file in tfrec_files if "test" not in file]

tfrec_files_train = [GCS_PATH + "../croppedskincancerimagestrain/" + file for file in tfrec_files if file != "train_"]

tfrec_files_train, tfrec_files_valid = train_test_split(tfrec_files_train, test_size=0.2)

tfrec_files_test = [GCS_PATH + "/tfrecords/" + file for file in tfrec_files if "test" in file]
dataset_train = tf.data.TFRecordDataset(tfrec_files_train)
for data in dataset_train.take(1):

    example = tf.train.Example()

    example.ParseFromString(data.numpy())

    print(str(example)[-600:])
dataset_train = dataset_train.map(parse_rec_train).map(get_img_label)
def get_img_list(dataset):

    arr_img = []

    for img, label in dataset.take(12):

        arr_img.append(img)

    return arr_img

    

def show_img(img_list):

    row=4; col=12;

    plt.figure(figsize=(20,row*12/col))

    x = 1

    for k in range(2):

        if k == 0:

            for img in img_list:

                plt.subplot(row,col,x)

                plt.imshow(img)

                x = x + 1

        elif k==1:

            for img in img_list:

                #img = tf.image.rgb_to_grayscale(img)

                img = smoothing(img)

                #img = tf.image.grayscale_to_rgb(img)

                plt.subplot(row,col,x)

                plt.imshow(img)

                x = x + 1

        else:

            for img in img_list:

                img = aug_img(img)

                plt.subplot(row,col,x)

                plt.imshow(img)

                x = x + 1



show_img(get_img_list(dataset_train)) 
from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import Conv2D, Dense, Input, Flatten, AveragePooling2D, GlobalAveragePooling2D
if 1==2:

    with strategy.scope():

        model = Sequential([

            Conv2D(filters=4, kernel_size=(3, 3), input_shape=(224,224,3), name="image", activation="relu"),

            AveragePooling2D(),

            Conv2D(filters=16, kernel_size=(3, 3), activation="relu"),

            AveragePooling2D(),

            Flatten(),

            Dense(128, activation="relu"),

            Dense(1, activation="sigmoid")

        ])

        model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])

        model.summary()


with strategy.scope():

    if 1==2:

        base_model = efn.EfficientNetB6(weights='imagenet', include_top=False, input_shape=(224,224,3))

        x = base_model.output

        x = GlobalAveragePooling2D()(x)

        x = Dense(1024, activation='relu')(x)

        predictions = Dense(1, activation='sigmoid')(x)



        model = Model(inputs=base_model.input, outputs=predictions)

    else:

        model_path = "/kaggle/input/model-data-pipeline-v18/model.h5"

        model = tf.keras.models.load_model(model_path)

       

    opt = tf.keras.optimizers.Adam(lr=0.00001)

    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05, name='binary_crossentropy')

    model.compile(optimizer=opt, loss=loss, metrics=['accuracy',tf.keras.metrics.AUC()])

    #model.summary()

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

DATASET_SIZE = 33126

steps_per_epoch=DATASET_SIZE//BATCH_SIZE

train_size = steps_per_epoch * BATCH_SIZE



def scheduler(epoch, lr):

    if epoch < 4:

        return lr

    else:

        return lr * tf.math.exp(-0.1)

    

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
if 1==1:

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5)



    bln_aug = False

    with strategy.scope():

        tfrec_files_train_all = np.array([GCS_PATH + "../croppedskincancerimagestrain/" + file for file in tfrec_files if file != "train_"])

        for train_idx, val_idx in kf.split(range(len(tfrec_files_train_all))):

            print(train_idx, val_idx)

            tfrec_files_train = tfrec_files_train_all[list(train_idx)]

            tfrec_files_valid = tfrec_files_train_all[list(val_idx)]

            temp_dataset = tf.data.TFRecordDataset(tfrec_files_train).repeat().shuffle(1024).map(parse_rec_train).map(get_img_label)

            if bln_aug:

                temp_dataset = temp_dataset.map(aug_img_label, num_parallel_calls=AUTO)

            dataset_train = temp_dataset.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

            dataset_valid = tf.data.TFRecordDataset(tfrec_files_valid).map(parse_rec_train).map(get_img_label).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

            for data,label in dataset_train.take(1):

                print(data.shape, label.shape)



            model.fit(dataset_train, epochs=1, verbose=1, steps_per_epoch=steps_per_epoch, callbacks=callback, validation_data = dataset_valid)    



    model.save("model_tpu.h5")
def filter_malign(data):

    return tf.equal(data["target"], 1)
tfrec_files_train_all = np.array([GCS_PATH + "../croppedskincancerimagestrain/" + file for file in tfrec_files if file != "train_"])



malign_dataset = tf.data.TFRecordDataset(tfrec_files_train_all).map(parse_rec_validate).filter(filter_malign)



#batch_data = dataset_train.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
name_list = list(malign_dataset.map(get_img_name).as_numpy_iterator())

img_list = []

for img in malign_dataset.map(get_img):

    img_list.append(np.array(img))

  

show_img(img_list[:12])
pred = model.predict(malign_dataset.map(get_img).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE))

pred=pred.flatten()

print(pred[pred>0.5].shape)

print(pred[pred<0.5].shape)
arr_pred = []

for i in range(5):

    dataset_test = malign_dataset.map(get_img).map(aug_img).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    arr_pred.append( model.predict(dataset_test, verbose=1) )

all_pred = np.concatenate(arr_pred, axis=1)
pred_mean = all_pred.mean(axis=1)

pred_mean=pred_mean.flatten()

print(pred_mean[pred_mean>0.5].shape)

print(pred_mean[pred_mean<0.5].shape)



pred_max = all_pred.max(axis=1)

pred_max=pred_max.flatten()

print(pred_max[pred_max>0.5].shape)

print(pred_max[pred_max<0.5].shape)
df = pd.DataFrame({"img":img_list, "image_name":name_list, "pred": list(pred), "pred_mean":list(pred_mean), "pred_max":list(pred_max)})
df["image_name"] = df["image_name"].map(lambda x: x.decode("utf-8"))

df.to_csv("malign.csv",index=False)



df = pd.merge(df, train_data, on="image_name")



df["Incorrect"] = df["pred_max"].map(lambda x: 1 if x >= 0.5 else 0)
df.head(1)
df[["Incorrect","sex","image_name"]].groupby(["Incorrect","sex"]).count().unstack(-1)
df[["Incorrect","anatom_site_general_challenge","image_name"]].groupby(["Incorrect","anatom_site_general_challenge"]).count().unstack(-1)
df.sort_values("pred_max", inplace=True)

img_list = df.head(12)["img"].values

show_img(img_list)
dataset_test = tf.data.TFRecordDataset(tfrec_files_test).take(200).map(parse_rec_test)

image_name_list = list(dataset_test.map(get_img_name).as_numpy_iterator())
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
dataset_test_raw = dataset_test.map(get_img)

arr_pred = []



for i in range(5):

    dataset_test = dataset_test_raw.map(aug_img).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    arr_pred.append( model.predict(dataset_test, verbose=1) )

    

all_pred = np.concatenate(arr_pred, axis=1)

pred = all_pred.max(axis=1)

pred_mean = all_pred.mean(axis=1)
df = pd.DataFrame({"image_name":image_name_list,"target":list(pred)})

df["image_name"] = df["image_name"].map(lambda x: x.decode("utf-8"))

df.to_csv("submission.csv", index=False)
df_mean = pd.DataFrame({"image_name":image_name_list,"target":list(pred_mean)})

df_mean["image_name"] = df_mean["image_name"].map(lambda x: x.decode("utf-8"))

df_mean.to_csv("submission_mean.csv", index=False)
print(df.sort_values("target").tail(5))
print(df_mean.sort_values("target").tail(5))
df[["target"]].hist()
df_mean[["target"]].hist()