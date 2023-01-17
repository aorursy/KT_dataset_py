!pip install efficientnet
!pip install tensorflow-addons=='0.9.1'
import efficientnet.tfkeras as efn

import pandas as pd
import cv2
import numpy as np
import keras
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Dense , Conv2D , Dropout , Flatten , Activation, MaxPooling2D , GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam , RMSprop 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , EarlyStopping , ModelCheckpoint , LearningRateScheduler

import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets

print(tf.__version__)
print(tf.keras.__version__)

from tensorflow.keras.models import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet

import tensorflow_addons as tfa

#for reproducible results
#import random
#seed_value = 13
#random.seed(seed_value)
#np.random.seed(seed_value)
#tf.random.set_seed(seed_value)
def seed_everything(seed=13):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_KERAS'] = '1'
    random.seed(seed)
    
seed_everything(42)
sample_submission = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")
train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
train.head()
test.head()
img_size = 800
train_images = []
for name in train['image_id'] :
    path = '/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg'
    image = cv2.imread(path)
    image=cv2.resize(image,(img_size,img_size),interpolation=cv2.INTER_AREA)
    train_images.append(image)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
test_images = []
for name in test['image_id'] :
    image = cv2.imread('/kaggle/input/plant-pathology-2020-fgvc7/images/'+name+'.jpg')
    image=cv2.resize(image,(img_size,img_size),interpolation=cv2.INTER_AREA)
    test_images.append(image)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25) :
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i])
y = train.drop('image_id',axis=1)
y = np.array(y)
y.shape
X = np.ndarray(shape=(len(train_images),img_size,img_size,3),dtype=np.float32)


for i,image in enumerate(train_images) :
    X[i] = image
    
    
X = X/255.0    
X.shape    
X_test = np.ndarray(shape=(len(test_images),img_size,img_size,3),dtype=np.float32)

for i,image in enumerate(test_images) :
    X_test[i]=image

X_test = X_test / 255.0  
X_test.shape
def construct_model() :
    
    model = Sequential()
    
    #Block 1
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu',input_shape=(img_size,img_size,3)))
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.3))
    
    #Block 2
    
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(128,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.3))

    #Block 3
    
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(Conv2D(256,kernel_size=(3,3),padding='valid',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.2))
    
    #Block 4 
    
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(512,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.2))    
    
    
    
    
    
    #Block 5
    
    model.add(Conv2D(1024,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(Conv2D(1024,kernel_size=(3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Dropout(0.2))

    
    #Final Block
    
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(256,activation='relu'))
    model.add(Dense(4,activation='softmax'))
        
    #Compile
    
    batch_size = 32
    epochs = 20 
    
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    #model.fit(X,y,validation_split=0.1,batch_size=batch_size,epochs=epochs)
    
    return model  
model = construct_model()
model.summary
data_gen = ImageDataGenerator(rotation_range=45,
                              horizontal_flip=True,
                              vertical_flip=True,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              zoom_range = 0.1,
                              shear_range = 0.1,
                              #brightness_range = [0.5,1.5],
                              fill_mode = 'nearest'
                             )
data_gen.fit(X)
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, random_state=42)
reduce_lr =  ReduceLROnPlateau(monitor = "val_loss", factor = 0.5, patience = 10,
  verbose = 0, mode = "auto", epsilon = 1e-04, cooldown = 0,
  min_lr = 1e-5)
es = EarlyStopping(monitor = "val_loss" , verbose = 1 , mode = 'min' , patience = 50 )
mc = ModelCheckpoint('best_model.h5', monitor = 'loss' , mode = 'min', verbose = 1 , save_best_only = True)
#lrs = LearningRateScheduler(lrfn,verbose=True)
import tensorflow_addons as tfa

checkpoint_path = "best_model.h5"

swa_mc = tfa.callbacks.AverageModelCheckpoint(filepath=checkpoint_path, save_weights_only=False,
                                                   update_weights=True)
import keras.backend as K
import tensorflow as tf

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Implementation of Focal Loss from the paper in multiclass classification
    Formula:
        loss = -alpha*((1-p)^gamma)*log(p)
    Parameters:
        alpha -- the same as wighting factor in balanced cross entropy
        gamma -- focusing parameter for modulating factor (1-p)
    Default value:
        gamma -- 2.0 as mentioned in the paper
        alpha -- 0.25 as mentioned in the paper
    """
    def focal_loss(y_true, y_pred):
        # Define epsilon so that the backpropagation will not result in NaN
        # for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        #y_pred = y_pred + epsilon
        # Clip the prediction value
        y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
        y_pred = tf.cast(y_pred,tf.float32)
        y_true = tf.cast(y_true,tf.float32)
        # Calculate cross entropy
        cross_entropy = -y_true*K.log(y_pred)
        # Calculate weight that consists of  modulating factor and weighting factor
        weight = alpha * y_true * K.pow((1-y_pred), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.sum(loss, axis=1)
        return loss
    
    return focal_loss
from sklearn.metrics import roc_auc_score
def roc_auc(preds, targs, labels=range(4)):
    # One-hot encode targets
    targs = np.eye(4)[targs]
    return np.mean([roc_auc_score(targs[:,i], preds[:,i]) for i in labels])
def healthy_roc_auc(*args):
    return roc_auc(*args, labels=[0])
def multiple_diseases_roc_auc(*args):
    return roc_auc(*args, labels=[1])
def rust_roc_auc(*args):
    return roc_auc(*args, labels=[2])
def scab_roc_auc(*args):
    return roc_auc(*args, labels=[3])
def aucroc_healthy(y_true, y_pred):
    return tf.py_func(healthy_roc_auc, (y_true, y_pred), tf.double)

def aucroc_multiple_diseases(y_true, y_pred):
    return tf.py_func(multiple_diseases_roc_auc, (y_true, y_pred), tf.double)

def aucroc_rust(y_true, y_pred):
    return tf.py_func(rust_roc_auc, (y_true, y_pred), tf.double)
def aucroc_scab(y_true, y_pred):
    return tf.py_func(scab_roc_auc, (y_true, y_pred), tf.double)


history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 150,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
model_effnet.load_weights('best_model.h5')
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model_effnet.predict(test_ds,verbose =1))
import numpy as np
tab = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab[i] = tab[i] + probabilities[j][i]
tab = tab / TTA_NUM              
sub.loc[:, 'healthy':] = tab
sub.to_csv('submissionEffnet+TTA.csv', index=False)
sub.head() 
def plot_performance_transfer_learning(history) :
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    t = f.suptitle('TL Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,121))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, 121, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, 121, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
def plot_performance(history) :
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    t = f.suptitle('Basic CNN Performance', fontsize=12)
    f.subplots_adjust(top=0.85, wspace=0.3)

    epoch_list = list(range(1,151))
    ax1.plot(epoch_list, history.history['accuracy'], label='Train Accuracy')
    ax1.plot(epoch_list, history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xticks(np.arange(0, 151, 5))
    ax1.set_ylabel('Accuracy Value')
    ax1.set_xlabel('Epoch')
    ax1.set_title('Accuracy')
    l1 = ax1.legend(loc="best")

    ax2.plot(epoch_list, history.history['loss'], label='Train Loss')
    ax2.plot(epoch_list, history.history['val_loss'], label='Validation Loss')
    ax2.set_xticks(np.arange(0, 151, 5))
    ax2.set_ylabel('Loss Value')
    ax2.set_xlabel('Epoch')
    ax2.set_title('Loss')
    l2 = ax2.legend(loc="best")
plot_performance(history)
predictions = model.predict(X_test)
df_pred  = pd.DataFrame(predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results_cnn = pd.DataFrame(results)
df_results_cnn.to_csv('submissionCNN.csv',index=False) #0.926
def construct_transfer_learning_model(model_name) :
    if model_name == 'MobileNet' :
        base_model=MobileNet(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))#imports the mobilenet model and discards the last 1000 neuron layer.
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    elif model_name == 'VGG16' : 
        base_model=vgg16.VGG16(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    elif model_name == 'DenseNet' :
        base_model = DenseNet121(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    elif model_name == 'Inception' :
        base_model=InceptionV3(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    elif model_name == 'ResNet' :
        base_model = ResNet50(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    elif model_name == 'EfficientNet' :
        base_model = efn.EfficientNetB3(weights = 'imagenet', include_top=False, input_shape = (img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-3] :
            layer.trainable = True
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
    x=Dropout(0.3)(x)
    x=Dense(256,activation='relu')(x) #dense layer 2
    preds=Dense(4,activation='softmax')(x) #final layer with softmax activation
    model=Model(inputs=base_model.input,outputs=preds)
    model.compile(optimizer='Adam',loss = 'categorical_crossentropy',metrics = ['accuracy'])
    return model
model = construct_transfer_learning_model('MobileNet')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 120,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
plot_performance_transfer_learning(history)
model = construct_transfer_learning_model('VGG16')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 120,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
plot_performance_transfer_learning(history)
model = construct_transfer_learning_model('DenseNet')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 140,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
plot_performance_transfer_learning(history)
predictions = model.predict(X_test)
df_pred  = pd.DataFrame(predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results_densenet = pd.DataFrame(results)
df_results_densenet.to_csv('submissionDenseNet.csv',index=False) #0.88
model = construct_transfer_learning_model('Inception')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 120,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
#plot_performance_transfer_learning(history)
predictions = model.predict(X_test)
df_pred  = pd.DataFrame(predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results = pd.DataFrame(results)
df_results.to_csv('submissionInception.csv',index=False) #0.878
model = construct_transfer_learning_model('ResNet')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 120,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
#plot_performance_transfer_learning(history)
predictions = model.predict(X_test)
df_pred  = pd.DataFrame(predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results = pd.DataFrame(results)
df_results.to_csv('submissionResNet.csv',index=False)
model = construct_transfer_learning_model('EfficientNet')
history = model.fit_generator(data_gen.flow(X_train,Y_train,batch_size=32),
                    steps_per_epoch = X_train.shape[0] // 32,
                    epochs = 120,
                    verbose = True,
                    validation_data= (X_val,Y_val),
                    callbacks = [reduce_lr,es,mc]
                   )
#plot_performance_transfer_learning(history)
predictions = model.predict(X_test)
df_pred  = pd.DataFrame(predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results_efficientnet = pd.DataFrame(results)
df_results_efficientnet.to_csv('submissionEfficientNet.csv',index=False)
df_predictions = 0.25*df_results_densenet + 0.25*df_results_cnn + 0.25 * df_results_mobilenet + 0.25 * df_results_efficientnet

df_pred  = pd.DataFrame(df_predictions, columns = ['healthy','multiple_diseases','rust','scab'])
results = {'image_id' : test.image_id,
            'healthy' : df_pred.healthy,
            'multiple_diseases' : df_pred.multiple_diseases,
            'rust' : df_pred.rust,
            'scab' : df_pred.scab
          }
df_results_densenet = pd.DataFrame(results)
df_results_densenet.to_csv('submissionStackNet.csv',index=False)
AUTO = tf.data.experimental.AUTOTUNE


try :
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU :',tpu.master())
except ValueError : 
    tpu = None
    
if tpu :
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else :
    strategy = tf.distribute.get_strategy()
    
print("Replicas :", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('plant-pathology-2020-fgvc7')  
path='../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv(path + 'sample_submission.csv')


train_paths = train.image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values


train_labels = train.loc[:,'healthy':].values
def LabelSmoothing(encodings , alpha):
    K = encodings.shape[1]
    y_ls = (1 - alpha) * encodings + alpha / K
    return y_ls
nb_classes = 4
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
img_size = 800
EPOCHS = 40
bool_random_brightness = False
bool_random_contrast = False
bool_random_hue = False
bool_random_saturation = False

gridmask_rate =0
cutmix_rate = 0.4
mixup_rate = 0
rotation = False
random_blackout = False
crop_size = 0
def decode_image(filename,label=None, image_size=(img_size,img_size)) :
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits,channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image,image_size)
    if label == None :
        return image
    else :
        return image, label
    
def data_augment(image, label=None, seed=2020) :
    image = tf.image.random_flip_left_right(image,seed=seed)
    image = tf.image.random_flip_up_down(image,seed=seed)
    #image = tf.keras.preprocessing.image.random_rotation(image,45)
    if crop_size :   
        image = tf.image.random_crop(image, size=[crop_size, crop_size, 3], seed=seed)
    if bool_random_brightness:
        image = tf.image.random_brightness(image,0.2)
    if bool_random_contrast:
        image = tf.image.random_contrast(image,0.6,1.4)
    if bool_random_hue:
        image = tf.image.random_hue(image,0.07)
    if bool_random_saturation:
        image = tf.image.random_saturation(image,0.5,1.5)
    if random_blackout :
        image= transform_random_blackout(image)
        
        
    
    if label == None :
        return image
    else :
        return image,label
# batch
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
def transform_cutmix_mix_up(image,label):
    # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
    DIM = img_size
    SWITCH = 0.5
    CUTMIX_PROB = 0.666
    MIXUP_PROB = 0.666
    # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP
    image2, label2 = cutmix(image, label, CUTMIX_PROB)
    image3, label3 = mixup(image, label, MIXUP_PROB)
    imgs = []; labs = []
    for j in range(AUG_BATCH):
        P = tf.cast( tf.random.uniform([],0,1)<=SWITCH, tf.float32)
        imgs.append(P*image2[j,]+(1-P)*image3[j,])
        labs.append(P*label2[j,]+(1-P)*label3[j,])
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
    image4 = tf.reshape(tf.stack(imgs),(AUG_BATCH,DIM,DIM,3))
    label4 = tf.reshape(tf.stack(labs),(AUG_BATCH,nb_classes))
    return image4,label4
def transform_random_blackout(img, sl=0.1, sh=0.2, rl=0.4):

    h, w, c = img_size, img_size, 3
    origin_area = tf.cast(h*w, tf.float32)

    e_size_l = tf.cast(tf.round(tf.sqrt(origin_area * sl * rl)), tf.int32)
    e_size_h = tf.cast(tf.round(tf.sqrt(origin_area * sh / rl)), tf.int32)

    e_height_h = tf.minimum(e_size_h, h)
    e_width_h = tf.minimum(e_size_h, w)

    erase_height = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_height_h, dtype=tf.int32)
    erase_width = tf.random.uniform(shape=[], minval=e_size_l, maxval=e_width_h, dtype=tf.int32)

    erase_area = tf.zeros(shape=[erase_height, erase_width, c])
    erase_area = tf.cast(erase_area, tf.uint8)

    pad_h = h - erase_height
    pad_top = tf.random.uniform(shape=[], minval=0, maxval=pad_h, dtype=tf.int32)
    pad_bottom = pad_h - pad_top

    pad_w = w - erase_width
    pad_left = tf.random.uniform(shape=[], minval=0, maxval=pad_w, dtype=tf.int32)
    pad_right = pad_w - pad_left

    erase_mask = tf.pad([erase_area], [[0,0],[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=1)
    erase_mask = tf.squeeze(erase_mask, axis=0)
    erased_img = tf.multiply(tf.cast(img,tf.float32), tf.cast(erase_mask, tf.float32))

    return tf.cast(erased_img, img.dtype)
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
def transform_rotation(image,label):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image randomly rotated, sheared, zoomed, and shifted
    DIM = img_size
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
    print(d.shape)    
    return tf.reshape(d,[DIM,DIM,3]),label
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

train_dataset=(
    tf.data.Dataset
    .from_tensor_slices((train_paths,train_labels.astype(np.float32)))
    .map(decode_image , num_parallel_calls=AUTO)
    .map(data_augment , num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
   )
#train_dataset = train_dataset.map(cutmix,num_parallel_calls = AUTO)
#train_dataset = train_dataset.map(transform_rotation, num_parallel_calls = AUTO)
#train_dataset = train_dataset.map(gridmask, num_parallel_calls = AUTO)
def create_train_data(train_paths,train_labels) :
    
    
    train_dataset=(tf.data.Dataset
    .from_tensor_slices((train_paths,train_labels.astype(np.float32)))
    .map(decode_image,num_parallel_calls = AUTO)
    .map(data_augment,num_parallel_calls = AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO))
    
    if cutmix_rate :
        train_dataset = train_dataset.map(cutmix,num_parallel_calls = AUTO) 
    if mixup_rate : 
        train_dataset = train_dataset.map(mixup, num_parallel_calls = AUTO)
    if rotation :
        train_dataset = train_dataset.map(transform_rotation, num_parallel_calls = AUTO)
    if blackout :
        train_dataset = train_dataset.map(transform_random_blackout, num_parallel_calls = AUTO)
    if gridmask_rate:
        train_dataset =train_dataset.map(gridmask, num_parallel_calls=AUTO)    
     
    return train_dataset    
def create_validation_data(valid_paths,valid_labels) :
    valid_data = (
        tf.data.Dataset
        .from_tensor_slices((valid_paths,valid_labels))
        .map(decode_image, num_parallel_calls = AUTO)
        .map(data_augment, num_parallel_calls= AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
        
    ) 
    return valid_data
test_dataset=(
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image ,num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)
def create_test_data(ordered=False) :
    test_dataset=(
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image ,num_parallel_calls=AUTO)
    .map(data_augment ,num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
     )
    return test_dataset
lr_start = 0.00001

lr_max = 0.0001 * strategy.num_replicas_in_sync
lr_min = 0.00001 
lr_rampup_epochs = 15
lr_sustain_epochs = 4
lr_exp_decay = .8


def lrfn(epoch) :
    if epoch < lr_rampup_epochs :
        lr = lr_start + (lr_max-lr_min) / lr_rampup_epochs * epoch
    elif epoch < lr_rampup_epochs + lr_sustain_epochs :
        lr = lr_max
    else :
        lr = lr_min + (lr_max - lr_min) * lr_exp_decay**(epoch - lr_sustain_epochs - lr_rampup_epochs)
    return lr

lr_callback = LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]

from matplotlib import pyplot as plt

plt.plot(rng,y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
lr_start = 0.00001

lr_max = 0.0001 * strategy.num_replicas_in_sync
lr_min = 0.00001 
lr_rampup_epochs = 13
lr_sustain_epochs = 2
lr_exp_decay = .8


def lrfn2(epoch) :
    if epoch < lr_rampup_epochs :
        lr = lr_start + (lr_max-lr_min) / lr_rampup_epochs * epoch
    elif epoch < lr_rampup_epochs + lr_sustain_epochs :
        lr = lr_max
    else :
        lr = lr_min + (lr_max - lr_min) * lr_exp_decay**(epoch - lr_sustain_epochs - lr_rampup_epochs)
    return lr

lr_callback = LearningRateScheduler(lrfn, verbose = True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]

from matplotlib import pyplot as plt

plt.plot(rng,y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))
def get_model():
    base_model = efn.EfficientNetB7(weights='imagenet',
                                    include_top = False,
                                    pooling='avg',
                                    input_shape=(img_size,img_size,3)
                                   )
    x = base_model.output
    predictions = Dense(nb_classes,activation='softmax')(x)
    return Model(inputs = base_model.input, outputs=predictions)
base_model = efn.EfficientNetB7(weights='noisy-student',
                                        include_top = False,
                                        #pooling='avg',
                                        input_shape=(img_size,img_size,3)
                                       )
focal_loss = True
label_smoothing = 0
SWA = False
from tensorflow.keras.applications import DenseNet121, DenseNet201
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import MobileNet , MobileNetV2
from tensorflow.keras.applications import InceptionResNetV2
import tensorflow.keras.layers as L


def get_model_generalized(name):
    if name == 'EfficientNet7' :
        base_model = efn.EfficientNetB7(weights='noisy-student',
                                        include_top = False,
                                        #pooling='avg',
                                        input_shape=(img_size,img_size,3)
                                       )
        base_model.trainable = True
        for layer in base_model.layers[:-20] :
            layer.trainable = True
    if name == 'EfficientNet3' :
        base_model = efn.EfficientNetB3(weights='noisy-student',
                                        include_top = False,
                                        #pooling='avg',
                                        input_shape=(img_size,img_size,3)
                                       )
        base_model.trainable = True
        for layer in base_model.layers[:-25] :
            layer.trainable = True        
            
    elif name == 'DenseNet' :
        base_model = DenseNet201(weights='imagenet',include_top=False,input_shape=(img_size,img_size,3))
        base_model.trainable = True
        for layer in base_model.layers[:-25] :
            layer.trainable = True
    elif name == 'MobileNet' :
        base_model = MobileNet(weights = 'imagenet', include_top=False,pooling='avg',input_shape=(img_size,img_size,3))
    elif name == 'Inception' :
        base_model = InceptionV3(weights = 'imagenet',include_top=False,pooling='avg',input_shape=(img_size,img_size,3))
    elif name == 'ResNet' :
        base_model = ResNet50(weights = 'imagenet',include_top=False,pooling='avg',input_shape=(img_size,img_size,3))
    elif name == 'Incepresnet' :
        base_model = InceptionResNetV2(weights = 'imagenet',include_top=False,input_shape=(img_size,img_size,3)) 
        base_model.trainable = True
        for layer in base_model.layers[:-25] :
            layer.trainable = True
    x = base_model.output
    x = L.GlobalAveragePooling2D()(x)
    x = Dense(1024,activation='relu')(x)
    x = Dropout(0.3)(x,training=True)
    predictions = Dense(nb_classes,activation='softmax')(x)
    model = Model(inputs = base_model.input, outputs=predictions) 
    
    
    if focal_loss : 
        loss= tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)
    elif label_smoothing :
        loss=CategoricalCrossentropy(label_smoothing=label_smoothing)
    else :
        loss = 'categorical_cross_entropy'
    if SWA :
        opt = tf.keras.optimizers.Adam(lr=1e-5) # roll back
        opt = tfa.optimizers.SWA(opt)
    else :
        opt = 'adam'
        
    model.compile(optimizer=opt,loss=loss,metrics=['accuracy',tf.keras.metrics.AUC()]) #roc_auc(),healthy_roc_auc(),multiple_diseases_roc_auc(),rust_roc_auc(),scab_roc_auc()
    #model.compile(optimizer='adam',loss=categorical_focal_loss(gamma=2.0, alpha=0.25),metrics=['accuracy',tf.keras.metrics.AUC()])
    return model
with strategy.scope() :
    #model = get_model()
    model_dense = get_model_generalized("DenseNet")
model_dense.fit(
    train2_dataset,
    steps_per_epoch = train2_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS 
)
model_dense.load_weights('best_model.h5')
preds = model_dense.predict(test_dataset)
sub.loc[:, 'healthy':] = preds
sub.to_csv('submissionDenseNet.csv', index=False)
sub.head()
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model_dense.predict(test_ds,verbose =1))
import numpy as np
tab_dense = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab_dense[i] = tab_dense[i] + probabilities[j][i]
tab_dense = tab_dense / TTA_NUM               
sub.loc[:, 'healthy':] = tab_dense
sub.to_csv('densenet+5TTA+pseudo+fl.csv', index=False)
sub.head() 
with strategy.scope() :
    #model = get_model()
    model_Inception = get_model_generalized("Inception")
'''model_Inception.fit(
    train_dataset,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS 
)'''
model_Inception.load_weights('best_model.h5')
preds = model_Inception.predict(test_dataset)
sub.loc[:, 'healthy':] = preds
sub.to_csv('submissionInception.csv', index=False)
sub.head()
with strategy.scope() :
    #model = get_model()
    model_mobilenet = get_model_generalized("MobileNet")
'''model_mobilenet.fit(
    train_dataset,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS 
)'''
model_mobilenet.load_weights('best_model.h5')
preds = model_mobilenet.predict(test_dataset)
sub.loc[:, 'healthy':] = preds
sub.to_csv('submissionMobileNet.csv', index=False)
sub.head()
with strategy.scope() :
    #model = get_model()
    model_resnet = get_model_generalized("ResNet")
''' 
model_resnet.fit(
    train_dataset,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS 
)'''
model_resnet.load_weights('best_model.h5')
preds = model_resnet.predict(test_dataset)
sub.loc[:, 'healthy':] = preds
sub.to_csv('submissionResnet.csv', index=False)
sub.head()
with strategy.scope() :
    #model = get_model()
    model_incepresnet = get_model_generalized("Incepresnet")
model_incepresnet.fit(
    train2_dataset,
    steps_per_epoch = train2_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS 
)
model_incepresnet.summary()
#model_incepresnet.load_weights('best_model.h5')
preds_incep = model_incepresnet.predict(test_dataset)
sub.loc[:, 'healthy':] = preds_incep
sub.to_csv('submissionIncepresnet.csv', index=False)
sub.head()
model_incepresnet.load_weights('best_model.h5')
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model_incepresnet.predict(test_ds,verbose =1))
import numpy as np
tab_incep = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab_incep[i] = tab_incep[i] + probabilities[j][i]
tab_incep = tab_incep / TTA_NUM               
sub.loc[:, 'healthy':] = tab_incep
sub.to_csv('Incepresnet+5TTA+pseudolabeling+focalloss+0.4cutmix.csv', index=False)
sub.head() 
with strategy.scope() :
    #model = get_model()
    model_effnet = get_model_generalized("EfficientNet3")

model_effnet.fit(
    train_dataset,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS )
model_effnet.load_weights('best_model.h5')
preds_eff = model_effnet.predict(test_dataset)
sub.loc[:, 'healthy':] = preds_eff
sub.to_csv('submissionEffnet3.csv', index=False)
sub.head()
model_effnet.load_weights('best_model.h5')
TTA_NUM = 8
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model_effnet.predict(test_ds,verbose =1))
import numpy as np
tab = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab[i] = tab[i] + probabilities[j][i]
tab = tab / TTA_NUM              
sub.loc[:, 'healthy':] = tab
sub.to_csv('submissionEffnet3+TTA.csv', index=False)
sub.head()        
del model_effnet
import gc
gc.collect()   
    
with strategy.scope() :
    model_effnet = get_model_generalized("EfficientNet7")
model_effnet.fit(
    train_dataset,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS )
model_effnet.summary()
train_pseudo = pd.read_csv('../input/heroseo-pseudo-labeling/train_pseudo_label_99999_v2.csv')

train2_paths = train_pseudo.image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values
train2_labels = train_pseudo.loc[:,'healthy':].values
train2_dataset=(
    tf.data.Dataset
    .from_tensor_slices((train2_paths,train2_labels.astype(np.float32)))
    .map(decode_image , num_parallel_calls=AUTO)
    .map(data_augment , num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
   )
#train2_dataset = train2_dataset.map(cutmix,num_parallel_calls = AUTO)   
#model_effnet.load_weights('best_model.h5')

with strategy.scope() :
    model_effnet = get_model_generalized("EfficientNet7")
model_effnet.fit(
    train2_dataset,
    steps_per_epoch = train2_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS )
ls = 0
model_effnet.load_weights('best_model.h5')
preds_eff = model_effnet.predict(test_dataset)
if ls :
    new_pred = LabelSmoothing(preds_eff, alpha=0.01)
    sub.loc[:, 'healthy':] = preds_eff
    sub.to_csv('EffnetB7+0.01LS+FL+Pseudo+swa.csv', index=False)
else :
    sub.loc[:, 'healthy':] = preds_eff
    sub.to_csv('EffnetB7+FL+Pseudo+swa.csv', index=False)
sub.head()
model_effnet.load_weights('best_model.h5')
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model_effnet.predict(test_ds,verbose =1))
tab1 = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab1[i] = tab1[i] + probabilities[j][i]
tab1 = tab1 / TTA_NUM    
if ls : #0.1,0.08 ,0.01
    tab1 = LabelSmoothing(tab1, alpha=ls)
    sub.loc[:, 'healthy':] = tab1           #.round().astype(int) rounding results
    sub.to_csv('EffnetB7+5TTA+FL+LS+Pseudo+swa.csv', index=False)
else :
    sub.loc[:, 'healthy':] = tab1           #.round().astype(int) rounding results
    sub.to_csv('EffnetB7+5TTA+FL+Pseudo+swa.csv', index=False)
sub.head()        
from sklearn.metrics import confusion_matrix
valid_data = pd.read_csv('../input/validationforplantpathology/plants_val120_train1974.csv')
def splitter(df):
    train = df.index[~df.is_valid].tolist()
    valid = df.index[df.is_valid].tolist()
    return train, valid

train,valid = splitter(valid_data)
valid
valid_paths = valid_data.iloc[valid].image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values
valid_labels = valid_data.iloc[valid].loc[:,'healthy':].values
valid_dataset=(
    tf.data.Dataset
    .from_tensor_slices(valid_paths)
    .map(decode_image ,num_parallel_calls=AUTO)
    .map(data_augment ,num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
     )

predictions = model_effnet.predict(valid_dataset,verbose =1)
predictions = predictions.round().astype(int)
y_test_non_category = [ np.argmax(t) for t in valid_labels ]
y_predict_non_category = [ np.argmax(t) for t in predictions ]

from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)


print(conf_mat)
del model_effnet
import gc
gc.collect()
sub.loc[:, 'healthy':] = (tab + tab1) / 2
sub.to_csv('efficientnet3+7.csv', index=False)
sub.head()
sub.loc[:, 'healthy':] = (tab + tab_incep) / 2
sub.to_csv('submissionEnsemblenet1.csv', index=False)
sub.head()
sub.loc[:, 'healthy':] = preds_eff*0.25  + preds_incep*0.75
sub.to_csv('submissionEnsemblenet2.csv', index=False)
sub.head()
sub.loc[:, 'healthy':] = tab*0.75  + tab_incep*0.25
sub.to_csv('submissionEnsemblenet3.csv', index=False)
sub.head()
from sklearn.model_selection import KFold

SEED = 13 
probs = []
histories = []
FOLDS = 5

kfolds = KFold(FOLDS , shuffle=True,random_state= SEED)
for i,(train_indices,valid_indices) in enumerate(kfolds.split(train_paths,train_labels)) :
    print() ; print('#'*25)
    print('Fold' , i+1)
    print('#'*25)
    
    
    trn_paths = train_paths[train_indices]
    trn_labels = train_labels[train_indices]
    
    vld_paths = train_paths[valid_indices]
    vld_labels = train_labels[valid_indices]
    with strategy.scope() :
        model_effnet = get_model_generalized("EfficientNet")
        history = model_effnet.fit(
            create_train_data(trn_paths,trn_labels),
            epochs = EPOCHS,
            steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
            callbacks = [lr_callback,mc],
            validation_data = create_validation_data(vld_paths,vld_labels),
            verbose = 1
        )
                
        model_effnet.load_weights('best_model.h5')
        prob = model_effnet.predict(
            create_test_data(ordered=False),
            verbose = 1
        )
        
        probs.append(prob)
        histories.append(history)
prob_sum = 0
for prob in probs :
    prob_sum = prob_sum + prob
prob_avg = prob_sum / FOLDS

sub.loc[:, 'healthy':] = prob_avg
sub.to_csv('submissionEffnet+5Kfolds.csv', index=False)
sub.head()
import os 

path = '../input/plantvillageapplecolor/Apple___'
labels = ['Apple_scab','Black_rot','Cedar_apple_rust','healthy']
all_image_ids = []
all_labels = []

for label in labels :
    image_ids = os.listdir(path+label)
    for i,image_id in enumerate(image_ids) :
        image_ids[i] =  'Apple___'+label+'/'+image_id
    labels = [label] * len(image_ids)
    
    all_image_ids = all_image_ids + image_ids
    all_labels = all_labels + labels
train_df = pd.DataFrame(
    {
    'id' : all_image_ids,
    'label' : all_labels    
    }
) 
train_df.head()
train_df = pd.concat([train_df, pd.get_dummies(train_df.label)], axis=1)
train_df.drop('label',axis=1,inplace=True)
train_df = train_df.sample(frac=1).reset_index(drop=True)
nb_classes = 4
img_size = 800
EPOCHS = 30
BATCH_SIZE = 64 * strategy.num_replicas_in_sync
train_df.head()
GCS_DS_PATH = KaggleDatasets().get_gcs_path('plantvillageapplecolor')


train_labels = train_df.loc[:,'Apple_scab':].values.astype(np.int64) #if we don't convert to int64 we will get tpu error because int8

#train_labels = train_df[labels].values.astype(np.int64)
train_paths = train_df.id.apply(lambda x: GCS_DS_PATH + '/' + x).values
pretraining_data = create_train_data(train_paths,train_labels)
train_paths
with strategy.scope() :
    #model = get_model()
    model_effnet = get_model_generalized("EfficientNet")

model_effnet.fit(
    pretraining_data,
    steps_per_epoch = train_labels.shape[0] // BATCH_SIZE,
    callbacks = [lr_callback,mc],
    epochs = EPOCHS )
model_effnet.save('effnet_pretrained.h5')
#clear_session()

#del model_effnet
#del train_dataset
#del train_labels
#del strategy
#gc.collect()
from tensorflow.keras.models import load_model

'''def get_pretrained_model():
    base_model = load_model('./effnet_pretrained.h5', compile=False)
    base_model.trainable = True
    for layer in base_model.layers[:-20] :
        layer.trainable = True
    x = base_model.layers[-1].output    
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)'''


with strategy.scope():
    model = get_pretrained_model()
    
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
model_effnet.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback,mc],
    epochs=EPOCHS
)
probs = model.predict(test_dataset)
sub.loc[:, 'healthy':] = probs
sub.to_csv('submissionPretrained.csv', index=False)
sub.head()
TTA_NUM = 5
probabilities = []
for i in range(TTA_NUM):
    #print(f'TTA Number: {i}\n')
    test_ds = create_test_data(ordered=False) 
    probabilities.append(model.predict(test_ds,verbose =1))
import numpy as np
tab = np.zeros((len(probabilities[1]),4))
for i in range(0,len(probabilities[1])) :
    for j in range(0,TTA_NUM) :
        tab[i] = tab[i] + probabilities[j][i]
tab = tab / TTA_NUM              
sub.loc[:, 'healthy':] = tab
sub.to_csv('submissionEffnet+TTA.csv', index=False)
sub.head()  
AUTO = tf.data.experimental.AUTOTUNE


try :
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU :',tpu.master())
except ValueError : 
    tpu = None
    
if tpu :
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else :
    strategy = tf.distribute.get_strategy()
    
print("Replicas :", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('plantpathology-apple-dataset')  
path1='../input/plant-pathology-2020-fgvc7/'
path='../input/plantpathology-apple-dataset/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path1 + 'test.csv')
sub = pd.read_csv(path1 + 'sample_submission.csv')


train_paths = train.image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values
test_paths = test.image_id.apply(lambda x : GCS_DS_PATH + '/images/' + x + '.jpg').values


train_labels = train.loc[:,'healthy':].values
