# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np
import pandas as pd 
import pydicom
import PIL
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from kaggle_datasets import KaggleDatasets

import tensorflow as tf
print("Tensorflow version " + tf.__version__)
from tensorflow.keras.applications import ResNet50,MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

! pip install -q efficientnet
import efficientnet.tfkeras as efn

# Input data files are available in the read-only "../input/" directory


#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DEVICE = 'TPU'
if DEVICE == "TPU":
    print("connecting to TPU...")
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        print("Could not connect to TPU")
        tpu = None

    if tpu:
        try:
            print("initializing  TPU ...")
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.experimental.TPUStrategy(tpu)
            print("TPU initialized")
        except _:
            print("failed to initialize TPU")


if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print('REPLICAS: {}'.format(REPLICAS))
# Global variables

DATA_PATH = '../input/siim-isic-melanoma-classification/'
IMAGE_PATH= DATA_PATH+'jpeg/train/'

DIM = 128
BATCH_SIZE = 16 * REPLICAS
EPOCHS = 5
VERBOSE =1
LR = 1e-3

#For using TPU's, you must read data from a GCS bucket. For using GPU's you can skip the GCS path and use ../input/{data folder}
GCS_PATH = KaggleDatasets().get_gcs_path("jpeg-melanoma-{}x{}".format(DIM,DIM)) # for resized JPEG images

GCS_PATH_2019 = KaggleDatasets().get_gcs_path("jpeg-isic2019-{}x{}".format(DIM,DIM))
train = pd.read_csv(DATA_PATH+'train.csv')
test = pd.read_csv(DATA_PATH+'test.csv')
train.head()
train.info()
train.benign_malignant.value_counts().plot.bar();
# number of unique patient id's
train.patient_id.nunique()
train.groupby('sex').size()
train_mel = train[train.target==1]
train_no_mel = train[train.target==0]

#img2arr = np.array(image)
image_names = np.random.choice(train_mel.image_name,size=6, replace=False)
plt.figure(figsize=(12,10))
for idx,image_name in enumerate(image_names):
    full_image_path = DATA_PATH+'jpeg/train/'+image_name+'.jpg'
    image = PIL.Image.open(full_image_path)
    plt.subplot(3,2,idx+1)
    plt.imshow(image);
plt.suptitle('Examples of Melanoma');

image_names = np.random.choice(train_no_mel.image_name,size=6, replace=False)
plt.figure(figsize=(12,10))
for idx,image_name in enumerate(image_names):
    full_image_path = DATA_PATH+'jpeg/train/'+image_name+'.jpg'
    image = PIL.Image.open(full_image_path)
    plt.subplot(3,2,idx+1)
    plt.imshow(image);
plt.suptitle('Examples of no Melanoma');
# Load the train and test files from the resized folder because it has information about duplicate images
train =pd.read_csv(GCS_PATH+'/train.csv')
test =pd.read_csv(GCS_PATH+'/test.csv')

# remove the 434 duplicate images
train = train.loc[~(train.tfrecord == -1), :].reset_index(drop=True)
def resize_images(image_name,size=(128,128)):
    '''
    Function to resize images using Pillow
    '''
    image = PIL.Image.open(os.path.join(IMAGE_PATH,image_name+'.jpg'))
    image = image.resize(size, resample = PIL.Image.LANCZOS)
    return image

train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=90,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            fill_mode='nearest')

idx = np.random.randint(train.shape[0])
sample_image_resized = resize_images(train.image_name[idx])
it = train_datagen.flow(np.expand_dims(sample_image_resized,axis=0), batch_size=1)

fig = plt.figure(figsize=(12,10))
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(it.next()[0,:,:,:])
    
ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0


def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    # returns 3x3 transformation matrix which transforms indices
        
    # CONVERT DEGREES TO RADIANS
    rotation = np.pi * rotation / 180.
    shear    = np.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    # ROTATION MATRIX
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    # SHEAR MATRIX
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    # ZOOM MATRIX
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    # SHIFT MATRIX
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, DIM=192):    
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    XDIM = DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    # GET TRANSFORMATION MATRIX
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    # LIST DESTINATION PIXEL INDICES
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    # FIND ORIGIN PIXEL VALUES           
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])
def parse_function(filename, label=None):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    # uncomment the line below if you wan't to resize on the fly, I'm using resized images already
    #image = tf.image.resize(image, [64, 64])
    if DEVICE== 'TPU':
        image = tf.reshape(image, [DIM,DIM, 3])  # explicit size needed for TPU
    return image, label

    
def augment_image(image, label):
#    image = transform(image, DIM=DIM)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    #Make sure the image is still in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def data_loader(filenames, labels, augment=False, repeat=True, shuffle=True):
    """
    Create tf Dataset for training and validation sets
    """
    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    
    dataset = dataset.map(parse_function, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    if shuffle:
        dataset = dataset.shuffle(2048)
    if repeat:
        dataset = dataset.repeat()
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size = AUTO)
    return dataset


#for i, label in data_loader(filenames, labels, augment=True).take(1):
#    print(i.shape)

def data_loader_unlabelled(filenames, augment=False, repeat=True):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.map(parse_function, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    if augment:
        dataset = dataset.map(augment_image, num_parallel_calls=AUTO)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size = AUTO)
    return dataset

#train_files = get_filenames(train['image_name'], GCS_PATH)

#for i, label in data_loader(train_files, labels=np.ones(len(train_files)), repeat=False, augment=True,shuffle=False).take(10):
#    print(i.shape)
def build_model(input_shape = (192,192,3), pretrained_model= ResNet50): 
    inp = tf.keras.layers.Input(shape=input_shape)
    base_model = pretrained_model(include_top=False, weights='imagenet', input_shape=input_shape)
    print("Using {} as the base model".format(base_model.name))
# weight freezing for Resnet50  
    if base_model.name == "resnet50":
        for layer in base_model.layers[:143]:
            layer.trainable = False
        for layer in base_model.layers[143:]:
            layer.trainable = True
    x = base_model(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1, activation = 'sigmoid')(x)
    model = tf.keras.Model(inputs= inp, outputs = x)
    optimizer = tf.keras.optimizers.Adam(learning_rate= LR)
    model.compile(optimizer= optimizer,
              loss= tfa.losses.SigmoidFocalCrossEntropy(gamma = 2.0, alpha = 0.80), #tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics= ['AUC'])
    return model



def get_lr_callback(batch_size=BATCH_SIZE):
    lr_start   = 0.000005
    lr_max     = 0.00000125  * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
model = build_model(input_shape = (DIM,DIM,3))
model.summary()


def get_filenames(df, path, train_or_test = 'train'):
    if train_or_test =='train':
        fnames_list = df.apply(lambda img_name: path+"/train/"+str(img_name)+'.jpg').values.tolist()
    elif train_or_test =='test':
        fnames_list = df.apply(lambda img_name: path+"/test/"+str(img_name)+'.jpg').values.tolist()
    else:
        print('Invalid argument')
        return None
    return fnames_list

## test file order

def _data_loader_test(filenames):
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset

test_files = get_filenames(test['image_name'], path = GCS_PATH, train_or_test='test')
test_data = _data_loader_test(test_files)
#list(test_data.as_numpy_iterator()
dload_files = []
for img in iter(test_data.unbatch()):
    dload_files.append(img.numpy().decode("utf-8"))
    
assert test_files[:10] == dload_files[:10]
## test input pipeline

def show_dataset(thumb_size, cols, rows, ds):
    mosaic = PIL.Image.new(mode='RGB', size=(thumb_size*cols + (cols-1), 
                                             thumb_size*rows + (rows-1)))
   
    for idx, data in enumerate(iter(ds)):
        img, _ = data[0], data[1]
        ix  = idx % cols
        iy  = idx // cols
        img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        img = img.resize((thumb_size, thumb_size), resample=PIL.Image.BILINEAR)
        mosaic.paste(img, (ix*thumb_size + ix, 
                           iy*thumb_size + iy))

    display(mosaic)
    
train_files = get_filenames(train.loc[np.arange(75), 'image_name'], GCS_PATH)
ds = data_loader(train_files, labels = None).unbatch().take(10*5)   
show_dataset(64, 10, 5, ds)

   
def display_training_curves(training, validation, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    #ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])
kfold_n_splits = 5
valid_set_aug=True 
test_set_aug=True
preds = []
skf = StratifiedKFold(n_splits=kfold_n_splits,shuffle=True,random_state=123123)

for fold,(idxT,idxV) in enumerate(skf.split(train.image_name,train.target.values)):

    print('#'*25)
    print('### FOLD {}'.format(fold+1))
    print('#'*25)

    if DEVICE=='TPU':
        if tpu: 
            tf.tpu.experimental.initialize_tpu_system(tpu)
            print('using TPU')
    elif DEVICE=='GPU': 
        print("Using GPU")
    K.clear_session()
    with strategy.scope():
        model = build_model(input_shape = (DIM,DIM,3), pretrained_model = efn.EfficientNetB1)
        
    train_files, valid_files = get_filenames(train.loc[idxT, 'image_name'], GCS_PATH), get_filenames(train.loc[idxV, 'image_name'], GCS_PATH) 
    train_labels, valid_labels = train.loc[idxT, 'target'].values.tolist(), train.loc[idxV]['target'].values.tolist()
    
    train_data = data_loader(train_files, labels = train_labels, augment=True, repeat=True, shuffle=False)
    valid_data = data_loader(valid_files, labels = valid_labels, augment=False, repeat=False, shuffle=False)

    sv_best_epoch = tf.keras.callbacks.ModelCheckpoint(
        "fold{}.h5".format(fold+1), monitor='val_loss', verbose=1, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')

    history = model.fit(train_data,epochs = EPOCHS,  class_weight = {0: 1, 1: 1.5},
                        steps_per_epoch= np.ceil(len(train_files)/BATCH_SIZE),
                        verbose= 1, callbacks=[sv_best_epoch,get_lr_callback(BATCH_SIZE)],validation_data = valid_data)
       
    print("#"*5+" Loading model weights from best epoch "+"#"*5)
    model.load_weights("fold{}.h5".format(fold+1))

    TTA = 5
    if valid_set_aug:
        print('#'*5+" With validation set augmentation size {}".format(TTA)+'#'*5)
        valid_data_tta = data_loader_unlabelled(valid_files, augment=True, repeat=True)
        ypred_valid = model.predict(valid_data_tta, steps = np.ceil(TTA*len(valid_files)/BATCH_SIZE), verbose=1)
        ypred_valid = ypred_valid[:len(valid_files)*TTA].reshape((len(valid_files),TTA), order = 'F') # Fortran like indexing, augmentations in columns
        ypred_valid = ypred_valid.mean(axis = 1) # take the average across number of augmentations
    else:
        print('#'*5+" Without validation set augmentation "+'#'*5)
        valid_data_no_tta = data_loader_unlabelled(valid_files, augment=False, repeat=False)
        ypred_valid = model.predict(valid_data_no_tta, verbose =1)

    auc_valid = roc_auc_score(valid_labels, ypred_valid)        
    print('AUC of validation fold {} = {}'.format(fold+1, auc_valid))

    print('Predicting Test image class')
    test_files = get_filenames(test['image_name'], path = GCS_PATH, train_or_test='test')

    if test_set_aug:
        print("Using test set augmentation")
        test_data = data_loader_unlabelled(test_files, augment=True, repeat=True)
        ypred_test = model.predict(test_data, steps = np.ceil(TTA*len(test_files)/BATCH_SIZE), verbose=1)
        ypred_test = ypred_test[:len(test_files)*TTA].reshape((len(test_files),TTA), order = 'F')
        ypred_test = ypred_test.mean(axis = 1)
    else:
        print("Without test set augmentation")
        test_data  = data_loader_unlabelled(test_files, augment=False, repeat=False)
        ypred_test = model.predict(test_data, verbose=1)
    preds += ypred_test/kfold_n_splits
    
    display_training_curves(
    history.history['loss'],
    history.history['val_loss'],
    'loss',
    211,
    )
    display_training_curves(
    history.history['auc'],
    history.history['val_auc'],
    'AUC',
    212,
    )

#filenames = train['image_name'].apply(lambda img_name: GCS_PATH+"train/"+str(img_name)+'.jpg').values.tolist()
#labels = train['target'].values.tolist()



def train_model(augment=True, kfold_n_splits = 5, valid_set_aug=True, test_set_aug=True):
    preds = []
    skf = StratifiedKFold(n_splits=kfold_n_splits,shuffle=True,random_state=1212)
    
    for fold,(idxT,idxV) in enumerate(skf.split(train.image_name,train.target.values)):

        print('#'*25)
        print('### FOLD {}'.format(fold+1))
        print('#'*25)
    
        K.clear_session()
        with strategy.scope():
            model = build_model()
        
        train_files, valid_files = get_filenames(train.loc[idxT, 'image_name'], GCS_PATH), get_filenames(train.loc[idxV, 'image_name'], GCS_PATH) 
        train_labels, valid_labels = train.loc[idxT, 'target'].values.tolist(), train.loc[idxV]['target'].values.tolist()

        train_data = data_loader(train_files, labels = train_labels, augment=True, repeat=True, shuffle=True)
        valid_data = data_loader(valid_files, labels = valid_labels, augment=False, repeat=False, shuffle=False)
        
        sv_best_epoch = tf.keras.callbacks.ModelCheckpoint(
            "fold{}.h5".format(fold+1), monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')
        
        history = model.fit(train_data,epochs = EPOCHS,  class_weight = {0: 1, 1: 1.5},
                            steps_per_epoch= np.ceil(len(train_files)/BATCH_SIZE),
                            verbose= VERBOSE, callbacks=[sv_best_epoch,get_lr_callback(BATCH_SIZE)],validation_data = valid_data)
        
        print("#"*5+" Loading model weights from best epoch "+"#"*5)
        model.load_weights("fold{}.h5".format(fold+1))
        
        TTA = 5
        if valid_set_aug:
            print('#'*5+" With validation set augmentation size {}".format(TTA)+'#'*5)
            valid_data_tta = data_loader_unlabelled(valid_files, augment=True, repeat=True)
            ypred_valid = model.predict(valid_data_tta, steps = np.ceil(TTA*len(valid_files)/BATCH_SIZE), verbose=1)
            ypred_valid = ypred_valid[:len(valid_files)*TTA].reshape((len(valid_files),TTA), order = 'F') # Fortran like indexing, augmentations in columns
            ypred_valid = ypred_valid.mean(axis = 1) # take the average across number of augmentations
        else:
            print('#'*5+" Without validation set augmentation "+'#'*5)
            valid_data_no_tta = data_loader_unlabelled(valid_files, augment=False, repeat=False)
            ypred_valid = model.predict(valid_data_no_tta, verbose =1)
            
        auc_valid = roc_auc_score(valid_labels, ypred_valid)        
        print('AUC of validation fold {} = {}'.format(fold+1, auc_valid))
        
        print('Predicting Test image class')
        test_files = get_filenames(test['image_name'], path = GCS_PATH, train_or_test='test')
        
        if test_set_aug:
            print("Using test set augmentation")
            test_data = data_loader_unlabelled(test_files, augment=True, repeat=True)
            ypred_test = model.predict(test_data, steps = np.ceil(TTA*len(test_files)/BATCH_SIZE), verbose=1)
            ypred_test = ypred_test[:len(test_files)*TTA].reshape((len(test_files),TTA), order = 'F')
            ypred_test = ypred_test.mean(axis = 1)
        else:
            print("Without test set augmentation")
            test_data  = data_loader_unlabelled(test_files, augment=False, repeat=False)
            ypred_test = model.predict(test_data, verbose=1)
        preds += ypred_test/kfold_n_splits
    return preds

#train_model(augment=True, kfold_n_splits = 5)
sub_df = pd.read_csv(DATA_PATH+"sample_submission.csv")
sub_df.loc[:,'target'] = preds
