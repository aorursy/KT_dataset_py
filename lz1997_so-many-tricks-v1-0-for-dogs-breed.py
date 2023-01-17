# 以狗分类为例子，可以自由使用所有方法以及其组合，可以自由更改数据集

# baseline：EfficientNetB0，可以自由更改网络



# 可以选择启用哪些方法



bool_random_flip_left_right=0

bool_random_flip_up_down=0

bool_random_brightness=0

bool_random_contrast=0

bool_random_hue=0

bool_random_saturation=0

cutmix_rate=0.

mixup_rate= 0.

gridmask_rate = 0.



pre_trained='imagenet' # None,'imagenet','noisy-student'

dense_activation='softmax' #'softmax','sigmoid'

bool_lr_scheduler=0



tta_times=0



#focal_loss和label_smoothing最多同时使用一个

bool_focal_loss = 0

label_smoothing_rate=0.



# 暂未引入，后续引入

# 旋转

# 交叉验证

# adversarial validation
#tta只有在使用了data_aug时才允许启用

bool_tta =  tta_times and max(  bool_random_flip_left_right,

                                bool_random_flip_up_down,

                                bool_random_brightness,

                                bool_random_contrast,

                                bool_random_hue,

                                bool_random_saturation)



print(bool_tta)



assert (bool_focal_loss and label_smoothing_rate) == 0 , 'focal_loss和label_smoothing最多同时使用一个'
#导入和安装包

!pip install -U efficientnet

!pip install tensorflow_addons

import numpy as np

import pandas as pd

import os

import tensorflow as tf

import random, re, math

from tensorflow.keras.layers import Dense

from tensorflow.keras.models import Model

from tensorflow.keras import optimizers

import tensorflow_addons as tfa



from kaggle_datasets import KaggleDatasets

import efficientnet.tfkeras as efn



print(tf.__version__)

print(tf.keras.__version__)
#针对不同硬件产生不同配置

AUTO = tf.data.experimental.AUTOTUNE

# Detect hardware, return appropriate distribution strategy

try:

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

    GCS_DS_PATH = KaggleDatasets().get_gcs_path()

    print(GCS_DS_PATH)

    # 如果连接失败，直接令GCS_DS_PATH='gs://kds-5a2f14e545dc319d4c7dadd6e05fe609d2674d9e2dea10e40102f4ea'

else:

    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)



#针对狗分类数据集，更换数据集时更换整个这大段



#超参数，根据数据和策略调参

BATCH_SIZE = 8 * strategy.num_replicas_in_sync

img_size = 299

EPOCHS = 20

lr_if_without_scheduler = 0.0003





path = '../input/dog-breed-identification/'



#训练集部分

df = pd.read_csv(path+'labels.csv')

df2 = pd.read_csv(path+'sample_submission.csv')

print(df.head())



breed = set(df['breed'])

#类别数

nb_classes = len(breed)



n = len(df)

class_to_num = dict(zip(breed, range(nb_classes)))

train_labels = np.zeros((n, nb_classes), dtype=np.uint8)

for i in range(n):

    train_labels[i][class_to_num[df['breed'][i]]] = 1



print('\n',train_labels[0])



train_paths = df[:1].id.apply(lambda x: (path+'train/') + x + '.jpg').values

from matplotlib import pyplot as plt

#随便看一张

img = plt.imread(train_paths[0])

print('\n',img.shape)

plt.imshow(img)





if tpu:

    path = GCS_DS_PATH+'/'



train_paths = df.id.apply(lambda x: (path+'train/') + x + '.jpg').values







test_paths = df2.id.apply(lambda x: (path+'test/') + x + '.jpg').values
def decode_image(filename, label=None, image_size=(img_size, img_size)):

    bits = tf.io.read_file(filename)

    image = tf.image.decode_jpeg(bits, channels=3)

    image = tf.cast(image, tf.float32) / 255.0

    image = tf.image.resize(image, image_size)

    if label is None:

        return image

    else:

        return image, label



# 只能写入test data 也能用的 aug   

def data_aug(image, label=None):

    if bool_random_flip_left_right:

        image = tf.image.random_flip_left_right(image)

    if bool_random_flip_up_down:    

        image = tf.image.random_flip_up_down(image)

    if bool_random_brightness:

        image = tf.image.random_brightness(image,0.2)

    if bool_random_contrast:

        image = tf.image.random_contrast(image,0.6,1.4)

    if bool_random_hue:

        image = tf.image.random_hue(image,0.07)

    if bool_random_saturation:

        image = tf.image.random_saturation(image,0.5,1.5)

    

    if label is None:

        return image

    else:

        return image, label
# 在batch内部互相随机取图

def cutmix(image, label, PROBABILITY = cutmix_rate):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with cutmix applied

    

    DIM = img_size    

    imgs = []; labs = []

    

    for j in range(BATCH_SIZE):

        

        #random_uniform( shape, minval=0, maxval=None)        

        # DO CUTMIX WITH PROBABILITY DEFINED ABOVE

        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

        

        # CHOOSE RANDOM IMAGE TO CUTMIX WITH

        k = tf.cast(tf.random.uniform([], 0, BATCH_SIZE), tf.int32)

        

        # CHOOSE RANDOM LOCATION

        #选一个随机的中心点

        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)

        

        # Beta(1, 1)等于均匀分布

        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0

        

        #P只随机出0或1，就是裁剪或是不裁剪

        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P

        ya = tf.math.maximum(0,y-WIDTH//2)

        yb = tf.math.minimum(DIM,y+WIDTH//2)

        xa = tf.math.maximum(0,x-WIDTH//2)

        xb = tf.math.minimum(DIM,x+WIDTH//2)

        

        # MAKE CUTMIX IMAGE

        one = image[j,ya:yb,0:xa,:]

        two = image[k,ya:yb,xa:xb,:]

        three = image[j,ya:yb,xb:DIM,:]        

        #得出了ya:yb区间内的输出图像

        middle = tf.concat([one,two,three],axis=1)

        #得到了完整输出图像

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)

        imgs.append(img)

        

        # MAKE CUTMIX LABEL

        #按面积来加权的

        a = tf.cast(WIDTH*WIDTH/DIM/DIM,tf.float32)

        lab1 = label[j,]

        lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)



    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE, nb_classes))

    return image2,label2
def mixup(image, label, PROBABILITY = mixup_rate):

    # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]

    # output - a batch of images with mixup applied

    DIM = img_size

    

    imgs = []; labs = []

    for j in range(BATCH_SIZE):

        

        # CHOOSE RANDOM

        k = tf.cast( tf.random.uniform([],0,BATCH_SIZE),tf.int32)

        a = tf.random.uniform([],0,1) # this is beta dist with alpha=1.0



        #根据概率抽取执不执行mixup

        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

        if P==1:

            a=0.

        

        # MAKE MIXUP IMAGE

        img1 = image[j,]

        img2 = image[k,]

        imgs.append((1-a)*img1 + a*img2)

        

        # MAKE CUTMIX LABEL

        lab1 = label[j,]

        lab2 = label[k,]

        labs.append((1-a)*lab1 + a*lab2)

            

    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)

    image2 = tf.reshape(tf.stack(imgs),(BATCH_SIZE,DIM,DIM,3))

    label2 = tf.reshape(tf.stack(labs),(BATCH_SIZE,nb_classes))

    return image2,label2
def transform(image, inv_mat, image_shape):

    h, w, c = image_shape

    cx, cy = w//2, h//2

    new_xs = tf.repeat( tf.range(-cx, cx, 1), h)

    new_ys = tf.tile( tf.range(-cy, cy, 1), [w])

    new_zs = tf.ones([h*w], dtype=tf.int32)

    old_coords = tf.matmul(inv_mat, tf.cast(tf.stack([new_xs, new_ys, new_zs]), tf.float32))

    old_coords_x, old_coords_y = tf.round(old_coords[0, :] + w//2), tf.round(old_coords[1, :] + h//2)

    clip_mask_x = tf.logical_or(old_coords_x<0, old_coords_x>w-1)

    clip_mask_y = tf.logical_or(old_coords_y<0, old_coords_y>h-1)

    clip_mask = tf.logical_or(clip_mask_x, clip_mask_y)

    old_coords_x = tf.boolean_mask(old_coords_x, tf.logical_not(clip_mask))

    old_coords_y = tf.boolean_mask(old_coords_y, tf.logical_not(clip_mask))

    new_coords_x = tf.boolean_mask(new_xs+cx, tf.logical_not(clip_mask))

    new_coords_y = tf.boolean_mask(new_ys+cy, tf.logical_not(clip_mask))

    old_coords = tf.cast(tf.stack([old_coords_y, old_coords_x]), tf.int32)

    new_coords = tf.cast(tf.stack([new_coords_y, new_coords_x]), tf.int64)

    rotated_image_values = tf.gather_nd(image, tf.transpose(old_coords))

    rotated_image_channel = list()

    for i in range(c):

        vals = rotated_image_values[:,i]

        sparse_channel = tf.SparseTensor(tf.transpose(new_coords), vals, [h, w])

        rotated_image_channel.append(tf.sparse.to_dense(sparse_channel, default_value=0, validate_indices=False))

    return tf.transpose(tf.stack(rotated_image_channel), [1,2,0])



def random_rotate(image, angle, image_shape):

    def get_rotation_mat_inv(angle):

        # transform to radian

        angle = math.pi * angle / 180

        cos_val = tf.math.cos(angle)

        sin_val = tf.math.sin(angle)

        one = tf.constant([1], tf.float32)

        zero = tf.constant([0], tf.float32)

        rot_mat_inv = tf.concat([cos_val, sin_val, zero, -sin_val, cos_val, zero, zero, zero, one], axis=0)

        rot_mat_inv = tf.reshape(rot_mat_inv, [3,3])

        return rot_mat_inv

    angle = float(angle) * tf.random.normal([1],dtype='float32')

    rot_mat_inv = get_rotation_mat_inv(angle)

    return transform(image, rot_mat_inv, image_shape)





def GridMask(image_height, image_width, d1, d2, rotate_angle=1, ratio=0.5):

    h, w = image_height, image_width

    hh = int(np.ceil(np.sqrt(h*h+w*w)))

    hh = hh+1 if hh%2==1 else hh

    d = tf.random.uniform(shape=[], minval=d1, maxval=d2, dtype=tf.int32)

    l = tf.cast(tf.cast(d,tf.float32)*ratio+0.5, tf.int32)



    st_h = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)

    st_w = tf.random.uniform(shape=[], minval=0, maxval=d, dtype=tf.int32)



    y_ranges = tf.range(-1 * d + st_h, -1 * d + st_h + l)

    x_ranges = tf.range(-1 * d + st_w, -1 * d + st_w + l)



    for i in range(0, hh//d+1):

        s1 = i * d + st_h

        s2 = i * d + st_w

        y_ranges = tf.concat([y_ranges, tf.range(s1,s1+l)], axis=0)

        x_ranges = tf.concat([x_ranges, tf.range(s2,s2+l)], axis=0)



    x_clip_mask = tf.logical_or(x_ranges < 0 , x_ranges > hh-1)

    y_clip_mask = tf.logical_or(y_ranges < 0 , y_ranges > hh-1)

    clip_mask = tf.logical_or(x_clip_mask, y_clip_mask)



    x_ranges = tf.boolean_mask(x_ranges, tf.logical_not(clip_mask))

    y_ranges = tf.boolean_mask(y_ranges, tf.logical_not(clip_mask))



    hh_ranges = tf.tile(tf.range(0,hh), [tf.cast(tf.reduce_sum(tf.ones_like(x_ranges)), tf.int32)])

    x_ranges = tf.repeat(x_ranges, hh)

    y_ranges = tf.repeat(y_ranges, hh)



    y_hh_indices = tf.transpose(tf.stack([y_ranges, hh_ranges]))

    x_hh_indices = tf.transpose(tf.stack([hh_ranges, x_ranges]))



    y_mask_sparse = tf.SparseTensor(tf.cast(y_hh_indices, tf.int64),  tf.zeros_like(y_ranges), [hh, hh])

    y_mask = tf.sparse.to_dense(y_mask_sparse, 1, False)



    x_mask_sparse = tf.SparseTensor(tf.cast(x_hh_indices, tf.int64), tf.zeros_like(x_ranges), [hh, hh])

    x_mask = tf.sparse.to_dense(x_mask_sparse, 1, False)



    mask = tf.expand_dims( tf.clip_by_value(x_mask + y_mask, 0, 1), axis=-1)



    mask = random_rotate(mask, rotate_angle, [hh, hh, 1])

    mask = tf.image.crop_to_bounding_box(mask, (hh-h)//2, (hh-w)//2, image_height, image_width)



    return mask



def apply_grid_mask(image, image_shape, PROBABILITY = gridmask_rate):

    AugParams = {

        'd1' : 100,

        'd2': 160,

        'rotate' : 45,

        'ratio' : 0.3

    }

    

        

    mask = GridMask(image_shape[0], image_shape[1], AugParams['d1'], AugParams['d2'], AugParams['rotate'], AugParams['ratio'])

    if image_shape[-1] == 3:

        mask = tf.concat([mask, mask, mask], axis=-1)

        mask = tf.cast(mask,tf.float32)

        #print(mask.shape) # (299,299,3)



# 会报错，放弃

#     imgs = []

#     BATCH_SIZE=len(image)

#     for j in range(BATCH_SIZE):

#         P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

#         if P==1:

#             imgs.append(image[j,]*mask)

#         else:

#             imgs.append(image[j,])

#     return tf.cast(imgs,tf.float32)



        

    # 整个batch启用或者不启用

    P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)

    if P==1:

        return image*mask

    else:

        return image



def gridmask(img_batch, label_batch):

    return apply_grid_mask(img_batch, (img_size,img_size, 3)), label_batch
# num_parallel_calls并发处理数据的并发数



train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((train_paths, train_labels.astype(np.float32)))

    .map(decode_image, num_parallel_calls=AUTO)

    .map(data_aug, num_parallel_calls=AUTO)

    .shuffle(512)

    .batch(BATCH_SIZE,drop_remainder=True)

    )





if cutmix_rate:  

    print('启用cutmix')

    train_dataset =train_dataset.map(cutmix, num_parallel_calls=AUTO)

if mixup_rate:  

    print('启用mixup')

    train_dataset =train_dataset.map(mixup, num_parallel_calls=AUTO)

if gridmask_rate:

    print('启用gridmask')

    train_dataset =train_dataset.map(gridmask, num_parallel_calls=AUTO)

if (cutmix_rate or mixup_rate):

    train_dataset =train_dataset.unbatch().shuffle(512).batch(BATCH_SIZE)





# repeat()代表无限制复制原始数据，这里可以用count指明复制份数，但要注意要比fit中的epochs大才可

# 直接调用repeat()的话，生成的序列就会无限重复下去

# prefetch: prefetch next batch while training (autotune prefetch buffer size)

train_dataset = train_dataset.repeat().prefetch(AUTO)





view_dataset = train_dataset.repeat(1)

it = view_dataset.__iter__()
#看看train_dataset 是否正常显示

show_x, show_y = it.next()

print(show_x.shape,'\n\n',show_y[0])



plt.figure(figsize=(12, 6))

for i in range(len(show_x[:8])):

    plt.subplot(2, 4, i+1)

    plt.imshow(show_x[i])
#生成测试集

def re_produce_test_dataset(test_paths):

    test_dataset = (

        tf.data.Dataset

        .from_tensor_slices(test_paths)

        .map(decode_image, num_parallel_calls=AUTO)

        )

    if bool_tta:

        test_dataset = test_dataset.map(data_aug, num_parallel_calls=AUTO)



    test_dataset = test_dataset.batch(BATCH_SIZE)

    return test_dataset
#tta时可重复运行这块观察是否多次运行时生成了不同的测试数据

test_dataset = re_produce_test_dataset(test_paths[:8])



view_dataset = test_dataset.repeat(1)

it = view_dataset.__iter__()

show_x= it.next()

print(show_x.shape,)



plt.figure(figsize=(12, 6))

for i in range(len(show_x)):

    plt.subplot(2, 4, i+1)

    plt.imshow(show_x[i])

    
#lr_scheduler

#数值按实际情况设置



LR_START = 0.00003

LR_MAX = 0.0003 * strategy.num_replicas_in_sync

LR_MIN = 0.00003

LR_RAMPUP_EPOCHS = 3

LR_SUSTAIN_EPOCHS = 4

LR_EXP_DECAY = .8



def lrfn(epoch):

    if epoch < LR_RAMPUP_EPOCHS:

        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START

    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:

        lr = LR_MAX

    else:

        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr

    

lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



rng = [i for i in range(EPOCHS)]

y = [lrfn(x) for x in rng]

if bool_lr_scheduler:

    plt.plot(rng, y)

    print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
#创建模型

def get_model():

    base_model =  efn.EfficientNetB0(weights=pre_trained, include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))

    x = base_model.output

    predictions = Dense(nb_classes, activation=dense_activation)(x)

    return Model(inputs=base_model.input, outputs=predictions)



with strategy.scope():

    model = get_model()
if label_smoothing_rate:

    print('启用label_smoothing')

    my_loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing_rate)

elif bool_focal_loss:

    

    my_loss = tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)



else:

    my_loss='categorical_crossentropy'

        

model.compile(optimizer=tf.keras.optimizers.Adam(lr_if_without_scheduler), 

              loss=my_loss,

              metrics=['accuracy'])
callbacks=[]

if bool_lr_scheduler:

    callbacks.append(lr_callback)
model.fit(

    train_dataset, 

    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,

    callbacks=callbacks,

    epochs=EPOCHS

)

if bool_tta:

    probabilities = []

    for i in range(tta_times+1):

        print('TTA Number: ',i,'\n')

        test_dataset = re_produce_test_dataset(test_paths)

        probabilities.append(model.predict(test_dataset))

    y_pred = np.mean(probabilities,axis =0)

    

else:

    test_dataset = re_produce_test_dataset(test_paths)

    y_pred = model.predict(test_dataset)
#针对不同数据不同后处理

for b in breed:

    df2[b] = y_pred[:,class_to_num[b]]

df2.to_csv('pred.csv', index=None)

df2.head()