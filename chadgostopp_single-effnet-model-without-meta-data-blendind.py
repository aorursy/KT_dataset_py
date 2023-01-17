!pip install -q efficientnet >> /dev/null
!pip install tensorflow~=2.2.0 tensorflow_gcs_config~=2.2.0
import pandas as pd, numpy as np
from kaggle_datasets import KaggleDatasets
import tensorflow as tf, re, math
import tensorflow.keras.backend as K
import efficientnet.tfkeras as efn
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
import time,random
DEVICE = "TPU"
# SEED=random.randint(1,100)
SEED = 42
FOLDS = 5 
CROP_SIZE=512
NET_SIZE=384
IMG_SIZES = [768,768,768,768,768]
INC2019 = [0,0,0,0,0]
INC2018 = [1,1,1,1,1]
BATCH_SIZES = [32]*FOLDS
EPOCHS = [13]*FOLDS
EFF_NETS = [6,6,6,6,6]
WGTS = [1/FOLDS]*FOLDS
TTA = 14
MODEL=[5]
VERBOSE = 2
# partition=0
ENSEMBLE=False
# UPSAMPLE MALIGNANT COUNT TIMES
M1 = [0]*FOLDS #2020 malig
M2 = [0]*FOLDS #ISIC malig
M3 = [0]*FOLDS #2019 good malig
M4 = [0]*FOLDS #2018 2017 malig


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
    else:
        DEVICE = "GPU"

if DEVICE != "TPU":
    print("Using default strategy for CPU and single GPU")
    strategy = tf.distribute.get_strategy()

if DEVICE == "GPU":
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    

AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')
GCS_PATH = [None]*FOLDS; GCS_PATH2 = [None]*FOLDS; GCS_PATH3 = [None]*FOLDS
for i,k in enumerate(IMG_SIZES):
    GCS_PATH[i] = KaggleDatasets().get_gcs_path('melanoma-%ix%i'%(k,k))
    GCS_PATH2[i] = KaggleDatasets().get_gcs_path('isic2019-%ix%i'%(k,k))
    GCS_PATH3[i] = KaggleDatasets().get_gcs_path('malignant-v2-%ix%i'%(k,k))
files_train = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/train*.tfrec')))
files_test  = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[0] + '/test*.tfrec')))
ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0
def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.
    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))
def transform(image, DIM=256):    
    XDIM = DIM%2 #fix for size 331
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 
    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 
    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)      
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
    return tf.reshape(d,[DIM, DIM,3])
def read_labeled_tfrecord(example):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
        'diagnosis'                    : tf.io.FixedLenFeature([], tf.int64),
        'target'                       : tf.io.FixedLenFeature([], tf.int64)
    }           
    example = tf.io.parse_single_example(example, tfrec_format)
    
    return {'input_1':[example['sex'],example['age_approx'],example['anatom_site_general_challenge']],
            'input_2':example['image']
            }, example['target']

def read_unlabeled_tfrecord(example, return_image_name):
    tfrec_format = {
        'image'                        : tf.io.FixedLenFeature([], tf.string),
        'image_name'                   : tf.io.FixedLenFeature([], tf.string),
        'patient_id'                   : tf.io.FixedLenFeature([], tf.int64),
        'sex'                          : tf.io.FixedLenFeature([], tf.int64),
        'age_approx'                   : tf.io.FixedLenFeature([], tf.int64),
        'anatom_site_general_challenge': tf.io.FixedLenFeature([], tf.int64),
    } 
    example = tf.io.parse_single_example(example, tfrec_format)
    return {'input_1':[example['sex'],example['age_approx'],example['anatom_site_general_challenge']],
            'input_2':example['image']
            }, example['image_name'] if return_image_name else 0

def prepare_image(img, augment=True, dim=256):    
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = transform(img,DIM=dim)
        img = tf.image.random_crop(img, [CROP_SIZE,CROP_SIZE,3])
        img = tf.image.random_flip_left_right(img)
        #img = tf.image.random_hue(img, 0.01)
        img = tf.image.random_saturation(img, 0.7, 1.3)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        img = tf.image.random_brightness(img, 0.1)
    else:
        img = tf.image.random_crop(img, [CROP_SIZE,CROP_SIZE,3])
    
    img = tf.image.resize(img, [NET_SIZE, NET_SIZE])        
    img = tf.reshape(img, [NET_SIZE,NET_SIZE, 3])
            
    return img

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) 
         for filename in filenames]
    return np.sum(n)

def get_dataset(files, augment = False, shuffle = False, repeat = False, 
                labeled=True, return_image_names=True, batch_size=16, dim=256):
    
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO)
    ds = ds.cache()
    
    if repeat:
        ds = ds.repeat()
    
    if shuffle: 
        ds = ds.shuffle(1024*8,seed=SEED)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)
        
    if labeled: 
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_names), 
                    num_parallel_calls=AUTO)      
    
    ds = ds.map(lambda x,y:({'input_1':x['input_1'],'input_2':prepare_image(x['input_2'], augment=augment, dim=dim)},y),
                num_parallel_calls=AUTO)
    
    ds = ds.batch(batch_size * REPLICAS)
    ds = ds.prefetch(AUTO)
    return ds
EFNS = [efn.EfficientNetB0, efn.EfficientNetB1, efn.EfficientNetB2, efn.EfficientNetB3, 
        efn.EfficientNetB4, efn.EfficientNetB5, efn.EfficientNetB6,efn.EfficientNetB7]

def binary_focal_loss(gamma=2., alpha=.25):
    def binary_focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return binary_focal_loss_fixed

def build_model(dim=384, model=MODEL):
    inpA = tf.keras.layers.Input(shape=(3,),name='input_1')
    y = tf.keras.layers.BatchNormalization()(inpA)
    y = tf.keras.layers.Dense(3,activation='relu',kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dense(3,activation='relu',kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)    
    y = tf.keras.layers.Dense(1,activation='relu',kernel_initializer='he_normal')(y)
    
    inpB = tf.keras.layers.Input(shape=(dim,dim,3),name='input_2')
    for i in model:
        base = EFNS[i](input_shape=(dim,dim,3),weights='imagenet',include_top=False)
        x = base(inpB)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)    
        x = tf.keras.layers.Dense(9,activation='relu',kernel_initializer='he_normal')(x)

        combined=tf.keras.layers.Concatenate()([x, y])
        combined = tf.keras.layers.BatchNormalization()(combined)            
        
    output=tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='he_normal')(combined)        
    model = tf.keras.Model(inputs=[inpA,inpB],outputs=output)

    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
#     loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    loss=binary_focal_loss(gamma = 2.0, alpha = 0.80)
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model

    

def build_ensemble_model(dim=384, model=MODEL):    
    inpA = tf.keras.layers.Input(shape=(3,),name='input_1')
    y = tf.keras.layers.BatchNormalization()(inpA)
    y = tf.keras.layers.Dense(3,activation='relu',kernel_initializer='he_normal')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Dense(1,activation='relu',kernel_initializer='he_normal')(y)    
    
    inpB = tf.keras.layers.Input(shape=(dim,dim,3),name='input_2')
    for i in model:
        base = EFNS[i](input_shape=(dim,dim,3),weights='imagenet',include_top=False)
        x = base(inpB)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.BatchNormalization()(x)    
        x = tf.keras.layers.Dense(1,activation='relu',kernel_initializer='he_normal')(x)
        if i==MODEL[0]:
            combined=x
        else:
            combined=tf.keras.layers.Concatenate()([combined, x])
            
    ensemble=tf.keras.layers.Concatenate()([combined, y])        
    ensemble = tf.keras.layers.BatchNormalization()(ensemble)                    
    output=tf.keras.layers.Dense(1,activation='sigmoid',kernel_initializer='he_normal')(ensemble)        
    model = tf.keras.Model(inputs=[inpA,inpB],outputs=output)

    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    loss=binary_focal_loss(gamma = 2.0, alpha = 0.80)
    model.compile(optimizer=opt,loss=loss,metrics=['AUC'])
    return model

def get_lr_callback(batch_size=8):
    lr_start   = 0.001
    lr_max     = 0.001
    lr_min     = 0.000001
    lr_ramp_ep = 2
    lr_sus_ep  = 0
    lr_decay   = 0.72
   
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
print('SEED:{}'.format(SEED))
skf = KFold(n_splits=FOLDS,shuffle=True,random_state=SEED)
oof_pred = []; oof_tar = []; oof_val = []; oof_names = []; oof_folds = [] 
preds = np.zeros((count_data_items(files_test),1))

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
        
time_callback = TimeHistory()
      
# print('Trainning Partition:{}'.format(partition))
for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15))):
#     if fold==partition:        
    # DISPLAY FOLD INFO
    if DEVICE=='TPU':
        if tpu: tf.tpu.experimental.initialize_tpu_system(tpu)
    print('#'*25); print('#### FOLD',fold+1)
    print('#### Image Size %i with EfficientNet B%i-B%i and batch_size %i'%
          (IMG_SIZES[fold],MODEL[0],MODEL[-1],BATCH_SIZES[fold]*REPLICAS))

    # CREATE TRAIN AND VALIDATION SUBSETS
    files_train = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxT])
    if INC2019[fold]:
        files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+1])
        print('#### Using 2019 external data')
    if INC2018[fold]:
        files_train += tf.io.gfile.glob([GCS_PATH2[fold] + '/train%.2i*.tfrec'%x for x in idxT*2])
        print('#### Using 2018+2017 external data')

    for k in range(M1[fold]):
        files_train += tf.io.gfile.glob([GCS_PATH3[fold] + '/train%.2i*.tfrec'%x for x in idxT])
        print('#### Upsample MALIG-1 data (2020 comp)')
    for k in range(M2[fold]):
        files_train += tf.io.gfile.glob([GCS_PATH3[fold] + '/train%.2i*.tfrec'%x for x in idxT+15])
        print('#### Upsample MALIG-2 data (ISIC website)')
    for k in range(M3[fold]):
        files_train += tf.io.gfile.glob([GCS_PATH3[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+1+30])
        print('#### Upsample MALIG-3 data (2019 comp)')
    for k in range(M4[fold]):
        files_train += tf.io.gfile.glob([GCS_PATH3[fold] + '/train%.2i*.tfrec'%x for x in idxT*2+30])
        print('#### Upsample MALIG-4 data (2018 2017 comp)')
                                
    np.random.shuffle(files_train); print('#'*25)
    files_valid = tf.io.gfile.glob([GCS_PATH[fold] + '/train%.2i*.tfrec'%x for x in idxV])
    files_test = np.sort(np.array(tf.io.gfile.glob(GCS_PATH[fold] + '/test*.tfrec')))

    # BUILD MODEL
    K.clear_session()
    with strategy.scope():
        if ENSEMBLE:
            print('Building Ensemble Model...')
            model = build_ensemble_model(dim=NET_SIZE,model=MODEL)
        else:
            print('Building Single Model...')
            model = build_model(dim=NET_SIZE,model=MODEL)


    # SAVE BEST MODEL EACH FOLD
    sv = tf.keras.callbacks.ModelCheckpoint(
        'fold-%i.h5'%fold, monitor='loss', verbose=0, save_best_only=True,
        save_weights_only=True, mode='min', save_freq='epoch')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_auc', mode = 'max', patience = 5, verbose = 1,
                                                       min_delta = 0.0001)
    cb_lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_auc', 
                                                          factor = 0.5, 
                                                          patience = 2, 
                                                          verbose = 1,
                                                          min_delta = 0.0001, 
                                                          mode = 'max')        
    # TRAIN
    print('Training...')
    history = model.fit(
        get_dataset(files_train, augment=True, shuffle=True, repeat=True,dim=IMG_SIZES[fold], batch_size = BATCH_SIZES[fold]), 
        epochs=EPOCHS[fold], 
        #get_lr_callback(BATCH_SIZES[fold]),cb_lr_schedule
        callbacks = [time_callback,sv,early_stopping,get_lr_callback(BATCH_SIZES[fold])], 
        steps_per_epoch=count_data_items(files_train)/BATCH_SIZES[fold]//REPLICAS,
        validation_data=get_dataset(files_valid,augment=False,shuffle=False,repeat=False,dim=IMG_SIZES[fold]), #class_weight = {0:1,1:2},
        verbose=VERBOSE)

    times = time_callback.times  

    print('Loading best model...')
    model.load_weights('fold-%i.h5'%fold)    

    print('Predicting OOF with TTA...')
    ds_valid = get_dataset(files_valid,labeled=False,return_image_names=False,augment=True,
            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4)
    ct_valid = count_data_items(files_valid);STEPS = TTA * ct_valid/BATCH_SIZES[fold]/4/REPLICAS
    pred = model.predict(ds_valid,steps=STEPS,verbose=VERBOSE)[:TTA*ct_valid,]
    oof_pred.append( np.mean(pred.reshape((ct_valid,TTA),order='F'),axis=1) )

    # GET OOF TARGETS AND NAMES
    ds_valid = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
            labeled=True, return_image_names=True)    
    oof_tar.append( np.array([target.numpy() for X, target in iter(ds_valid.unbatch())]) )
    oof_folds.append( np.ones_like(oof_tar[-1],dtype='int8')*fold )
    ds = get_dataset(files_valid, augment=False, repeat=False, dim=IMG_SIZES[fold],
                labeled=False, return_image_names=True)
    oof_names.append( np.array([img_name.numpy().decode("utf-8") for X, img_name in iter(ds.unbatch())])) 

    # REPORT RESULTS
    auc = roc_auc_score(oof_tar[-1],oof_pred[-1])
    oof_val.append(np.max( history.history['val_auc'] ))
    print('#### FOLD %i OOF AUC without TTA = %.3f, with TTA = %.3f'%(fold+1,oof_val[-1],auc))    

    # PREDICT TEST USING TTA
    print('Predicting Test with TTA...')
    ds_test = get_dataset(files_test,labeled=False,return_image_names=False,augment=True,
            repeat=True,shuffle=False,dim=IMG_SIZES[fold],batch_size=BATCH_SIZES[fold]*4)
    ct_test = count_data_items(files_test); STEPS = TTA * ct_test/BATCH_SIZES[fold]/4/REPLICAS
    pred = model.predict(ds_test,steps=STEPS,verbose=VERBOSE)[:TTA*ct_test,] 
    preds[:,0] += np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) * WGTS[fold]   
# COMPUTE OVERALL OOF AUC
oof = np.concatenate(oof_pred); true = np.concatenate(oof_tar);
names = np.concatenate(oof_names); folds = np.concatenate(oof_folds)
auc = roc_auc_score(true,oof)
print('Overall OOF AUC with TTA = %.3f'%auc)

# SAVE OOF TO DISK
df_oof = pd.DataFrame(dict(
    image_name = names, target=true, pred = oof, fold=folds))
df_oof.to_csv('oof.csv',index=False)
df_oof.head()
ds = get_dataset(files_test, augment=False, repeat=False, dim=IMG_SIZES[fold],
                 labeled=False, return_image_names=True)

image_names = np.array([img_name.numpy().decode("utf-8") 
                        for img, img_name in iter(ds.unbatch())])

submission = pd.DataFrame(dict(image_name=image_names, target=preds[:,0]))
submission = submission.sort_values('image_name') 
submission.to_csv('submission.csv', index=False)
submission.head()