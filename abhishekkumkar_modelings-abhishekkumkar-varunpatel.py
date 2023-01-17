!pip install -q efficientnet
!pip install image-classifiers==1.0.0b1
import pandas as pd
import numpy as np
import os , math , re , random


from kaggle_datasets import KaggleDatasets

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


import tensorflow as tf
import tensorflow.keras.layers as L
from classification_models.tfkeras import Classifiers
from tensorflow.keras.callbacks import EarlyStopping,LearningRateScheduler, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model


import efficientnet.tfkeras as efn
from tensorflow.keras.applications import DenseNet121, DenseNet201
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet , MobileNetV2
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras import optimizers

!pip install tensorflow-addons=='0.9.1'
import tensorflow_addons as tfa

import cv2

def seed_everything(seed=13):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_KERAS'] = '1'
    random.seed(seed)
    
seed_everything(42)
try :
    tpu=tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on :',tpu.master())
except ValueError :
    tpu = None

if tpu :    
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else :
    strategy = tf.distribute.get_strategy()
    
print('Replicas :',strategy.num_replicas_in_sync)    
AUTO  = tf.data.experimental.AUTOTUNE

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

EPOCHS = 5
BATCH_SIZE = 8 * strategy.num_replicas_in_sync 
img_size = 512
SEED =  42
nb_classes = 1
sub = pd.read_csv('../input/alaska2-image-steganalysis/sample_submission.csv')
files_name = np.array(os.listdir('../input/alaska2-image-steganalysis/Cover'))

path = '../input/alaska2-image-steganalysis/'
def append_path(pre) :
    return np.vectorize(lambda file : os.path.join(GCS_DS_PATH,pre,file))
def append_path2(pre) :
    return np.vectorize(lambda file : os.path.join(path,pre,file))
#reading file names and shuffling them
positives = files_name.copy()
negatives = files_name.copy()

np.random.shuffle(positives)
np.random.shuffle(negatives)
test_paths = append_path('Test')(sub.Id.values)
test_paths2 = append_path2('Test')(sub.Id.values)
#creating data so that i have 30k pos image (10k of each transformation) and 30k neg image
jmipod = append_path('JMiPOD')(positives[:74999])
#jmipod2 = append_path2('JMiPOD')(positives[:2000])
juniward = append_path('JUNIWARD')(positives[:74999])
#juniward2 = append_path2('JUNIWARD')(positives[2000:4000])
uerd = append_path('UERD')(positives[:74999])
#uerd2 = append_path2('UERD')(positives[4000:6000])

pos_paths = np.concatenate([jmipod,juniward,uerd])
#pos_paths2 = np.concatenate([jmipod2,juniward2,uerd2])
neg_paths = append_path('Cover')(negatives[:74999])
#neg_paths2 = append_path2('Cover')(negatives[:6000])
train_paths = np.concatenate([pos_paths,neg_paths])
#train_paths2 = np.concatenate([pos_paths2,neg_paths2])

train_labels = np.array([1] * len(pos_paths) + [0] * len(neg_paths))
train_paths , train_labels = shuffle(train_paths,train_labels)
#train_paths2 , train_labels = shuffle(train_paths2,train_labels)
#splitting data into train 85% / validation 15%
X_train , X_val, y_train , y_val = train_test_split(train_paths, train_labels , random_state=SEED,test_size = 0.15)
bool_random_brightness = False
bool_random_contrast = False
bool_random_hue = False
bool_random_saturation = False

cutmix_rate = 0
mixup_rate = 0
gridmask_rate = 0
! git clone https://github.com/dwgoon/jpegio
# Once downloaded install the package
!pip install jpegio/.
import jpegio as jio
import numpy as np
import jpegio as jpio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
def JPEGdecompressYCbCr(jpegStruct):
    
    nb_colors=len(jpegStruct.coef_arrays)
        
    [Col,Row] = np.meshgrid( range(8) , range(8) )
    T = 0.5 * np.cos(np.pi * (2*Col + 1) * Row / (2 * 8))
    T[0,:] = T[0,:] / np.sqrt(2)
    
    sz = np.array(jpegStruct.coef_arrays[0].shape)
    
    imDecompressYCbCr = np.zeros([sz[0], sz[1], nb_colors]);
    szDct = (sz/8).astype('int')
    
    
    
    for ColorChannel in range(nb_colors):
        tmpPixels = np.zeros(sz)
    
        DCTcoefs = jpegStruct.coef_arrays[ColorChannel];
        if ColorChannel==0:
            QM = jpegStruct.quant_tables[ColorChannel];
        else:
            QM = jpegStruct.quant_tables[1];
        
        for idxRow in range(szDct[0]):
            for idxCol in range(szDct[1]):
                D = DCTcoefs[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8]
                tmpPixels[idxRow*8:(idxRow+1)*8 , idxCol*8:(idxCol+1)*8] = np.dot( np.transpose(T) , np.dot( QM * D , T ) )
        imDecompressYCbCr[:,:,ColorChannel] = tmpPixels;
    return imDecompressYCbCr
for i, img in enumerate(os.listdir('../input/alaska2-image-steganalysis/Cover')[:10]):
    imgRGB = mpimg.imread('../input/alaska2-image-steganalysis/Cover/' + img)
    jpegStruct = jpio.read('../input/alaska2-image-steganalysis/Cover/' + img)
    print(type(jpegStruct))
    
    imDecompressYCbCr = JPEGdecompressYCbCr(jpegStruct)
    
    print(type(imDecompressYCbCr))
    print(imDecompressYCbCr.shape)
'''def decode_image(file,label=None,img_size=(img_size,img_size)) :
    
    func = tf.py_function(JPEGdecompressYCbCr,[file],[tf.float16,tf.float16,3])
    #bits = tf.io.read_file(file)
    #image = tf.image.decode_jpeg(bits, channels = 3)
    #image = tf.cast(image, tf.float16) # /255.0
    #image = np.float32(image)
    #image = image.eval(session=tf.compat.v1.Session())
    #image= cv2.cvtColor(image.numpy(), cv2.COLOR_BGR2YCR_CB) 
    #image = tf.convert_to_tensor(image, dtype = tf.float16)
    image = func(file)
    print('done')
    print(image.shape)
    print(image)
    image = tf.image.resize(image,img_size)
    
    if label is None :
        return image
    else :
        return image,label'''
def decode_image(filename,label=None, image_size=(img_size,img_size)) :
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits,channels=3)
    image = tf.cast(image, tf.float16) / 255.0
    image = tf.image.resize(image,image_size)
    if label == None :
        return image
    else :
        return image, label
    
def data_augment(image ,label = None,seed=2020) :
    
    image = tf.image.random_flip_left_right(image,seed=seed)
    image = tf.image.random_flip_up_down(image,seed=seed)
    if bool_random_brightness:
        image = tf.image.random_brightness(image,0.2,seed=seed)
    if bool_random_contrast:
        image = tf.image.random_contrast(image,0.6,1.4, seed=seed)
    if bool_random_hue:
        image = tf.image.random_hue(image,0.07,seed=seed)
    if bool_random_saturation:
        image = tf.image.random_saturation(image,0.5,1.5,seed=seed)
    
    if label is None :
        return image
    else :
        return image , label
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
        x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
        
        # Beta(1, 1)
        b = tf.random.uniform([], 0, 1) # this is beta dist with alpha=1.0
        

        WIDTH = tf.cast(DIM * tf.math.sqrt(1-b),tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        
        # MAKE CUTMIX IMAGE
        one = image[j,ya:yb,0:xa,:]
        two = image[k,ya:yb,xa:xb,:]
        three = image[j,ya:yb,xb:DIM,:]        
        #ya:yb
        middle = tf.concat([one,two,three],axis=1)

        img = tf.concat([image[j,0:ya,:,:],middle,image[j,yb:DIM,:,:]],axis=0)
        imgs.append(img)
        
        # MAKE CUTMIX LABEL
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

        #mixup
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
        P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
    if P==1:
        return image*mask
    else:
        return image

def gridmask(img_batch, label_batch):
    return apply_grid_mask(img_batch, (img_size,img_size, 3)), label_batch
def create_train_data(train_paths,train_labels) :
    train_data =  (
    tf.data.Dataset.from_tensor_slices((train_paths,train_labels))
    .map(decode_image, num_parallel_calls = AUTO)
    .map(data_augment ,num_parallel_calls = AUTO)
    .cache()
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )    
    
    if cutmix_rate :
        train_data = train_data.map(cutmix,num_parallel_calls = AUTO) 
    if mixup_rate : 
        train_data = train_data.map(mixup, num_parallel_calls = AUTO)
    if gridmask_rate :
        train_data = train_data.map(gridmask, num_parallel_calls = AUTO)
    
    return train_data

def create_validation_data(valid_paths,valid_labels):
    valid_data = (
        tf.data.Dataset.from_tensor_slices((valid_paths,valid_labels))
        .map(decode_image , num_parallel_calls = AUTO)
        .map(data_augment , num_parallel_calls = AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
        .cache()
    )
    return valid_data

def create_test_data(test_paths,aug=False):
    test_data = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(decode_image, num_parallel_calls = AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )
    
    if aug == True :
        test_data=test_data.map(data_augment ,num_parallel_calls = AUTO)
    return test_data
train_labels = tf.cast(train_labels,tf.float32)
train_data = create_train_data(train_paths,train_labels)
test_data = create_test_data(test_paths)
train_labels = tf.cast(train_labels,tf.float32)
train_data = create_train_data(train_paths,train_labels)
test_data = create_test_data(test_paths)
lr_start = 0.001

lr_max = 0.001 * strategy.num_replicas_in_sync
lr_min = 0.001 
lr_rampup_epochs = 1
lr_sustain_epochs = 2
lr_exp_decay = .8


def lrfn(epoch) :
    if epoch < lr_rampup_epochs :
        lr = lr_start + (lr_max-lr_min) / lr_rampup_epochs * epoch
    elif epoch < lr_rampup_epochs + lr_sustain_epochs :
        lr = lr_max
    else :
        lr = lr_min + (lr_max - lr_min) * lr_exp_decay**(epoch - lr_sustain_epochs - lr_rampup_epochs)
    return lr

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]

from matplotlib import pyplot as plt

plt.plot(rng,y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
lr_scheduler = LearningRateScheduler(lrfn , verbose=True)
mc = ModelCheckpoint('best_model.h5',monitor=tf.keras.metrics.AUC(),mode='max',save_best_only=True,verbose=1)
# https://www.kaggle.com/anokas/weighted-auc-metric-updated
from sklearn import metrics
import numpy as np

def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights =        [       2,   1]
    
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)
    
    # size of subsets
    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])
    
    # The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.
    normalization = np.dot(areas, weights)
    
    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric
        
    return competition_metric / normalization
focal_loss = False
label_smoothing = 0

def get_model_generalized(name,trainable_layers=20):
    if name == 'Inception' :
        base_model = InceptionV3(weights = 'imagenet',include_top=False,input_shape=(img_size,img_size,3))
        
    base_model.trainable = True
    for layer in base_model.layers[:-trainable_layers] :
        layer.trainable = True
    layer = base_model.output
    layer = L.GlobalAveragePooling2D()(layer)
    layer = L.Dense(1024,activation='relu')(layer)
    layer = L.Dropout(0.3)(layer,training=True)
    predictions = L.Dense(nb_classes,activation='sigmoid')(layer)
    predictions = tf.cast(predictions,tf.float32)
    model = Model(inputs = base_model.input, outputs=predictions)
    if focal_loss : 
        loss= tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
    if label_smoothing :
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    else :
        loss = 'binary_crossentropy'
    
    #opt = optimizers.Adam(learning_rate = 0.001)
    
    model.compile(optimizer='adam',loss=loss,metrics=['accuracy',tf.keras.metrics.AUC()])  
    return model
model_effnet = get_model_generalized('Inception')
model_effnet.summary()
steps_per_epoch = len(train_labels) // BATCH_SIZE
with strategy.scope() :
    model_effnet = get_model_generalized('Inception')
history = model_effnet.fit(
    train_data,
    steps_per_epoch = steps_per_epoch,
   # validation_data = valid_data,
    epochs = EPOCHS,
    #callbacks = [lr_scheduler]  # early stopping , Model checkpoint , schedulers
    )
predictions = model_effnet.predict(test_data , verbose=1)
sub['Label'] = predictions
sub.to_csv('e2_fulldata.csv',index=False)









