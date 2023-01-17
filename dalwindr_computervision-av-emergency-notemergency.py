# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#https://www.kaggle.com/nagarajukuruva/computer-vision-imageprocessing
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os
import pickle

import joblib
from skimage.io import imread, imshow
from tqdm import tqdm
from skimage.transform import resize


warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 200)

# import numpy as np
# np.random.seed(1001)
import tensorflow as tf
import random
SEED=26
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
from sklearn.model_selection import train_test_split


#from __future__ import print_function
from keras.models import Model, Sequential, load_model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback

from imageio import imread
from keras import activations
#!pip install keras-vis
#from vis.input_modifiers import Jitter
#from vis.utils import utils
#from vis.visualization import visualize_activation, get_num_filters


## extra imports to set GPU options
import tensorflow as tf
from keras import backend as k

# ###################################
# # TensorFlow wizardry
# config = tf.ConfigProto()
 
# # Don't pre-allocate memory; allocate as-needed
# config.gpu_options.allow_growth = True
 
# # Only allow a total of half the GPU memory to be allocated
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
 
# # Create a session with the above options specified.
# k.tensorflow_backend.set_session(tf.Session(config=config))
# ###################################
def reset_random_seeds():
    import tensorflow as tf
    import random
    import random
    os.environ['PYTHONHASHSEED']=str(26)
    random.seed(26)
    np.random.seed(26)
    tf.random.set_seed(26)
reset_random_seeds()
def smooth_curve(points, factor=0.8):
    smoothed = []
    for point in points:
        if smoothed:
            previous = smoothed[-1]
            smoothed.append(previous * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def plot_compare(history, steps=-1):
    if steps < 0:
        steps = len(history.history['accuracy'])
    acc = smooth_curve(history.history['accuracy'][:steps])
    val_acc = smooth_curve(history.history['val_accuracy'][:steps])
    loss = smooth_curve(history.history['loss'][:steps])
    val_loss = smooth_curve(history.history['val_loss'][:steps])
    
    plt.figure(figsize=(6, 4))
    plt.plot(loss, c='#0c7cba', label='Train Loss')
    plt.plot(val_loss, c='#0f9d58', label='Val Loss')
    plt.xticks(range(0, len(loss), 5))
    plt.xlim(0, len(loss))
    plt.title('Train Loss: %.3f, Val Loss: %.3f' % (loss[-1], val_loss[-1]), fontsize=12)
    plt.legend()
    
    plt.figure(figsize=(6, 4))
    plt.plot(acc, c='#0c7cba', label='Train Acc')
    plt.plot(val_acc, c='#0f9d58', label='Val Acc')
    plt.xticks(range(0, len(acc), 5))
    plt.xlim(0, len(acc))
    plt.title('Train Accuracy: %.3f, Val Accuracy: %.3f' % (acc[-1], val_acc[-1]), fontsize=12)
    plt.legend()
    
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x
 
def save_history(history, fn):
    with open(fn, 'wb') as fw:
        pickle.dump(history.history, fw, protocol=2)

def load_history(fn):
    class Temp():
        pass
    history = Temp()
    with open(fn, 'rb') as fr:
        history.history = pickle.load(fr)
    return history

def jitter(img, amount=32):
    ox, oy = np.random.randint(-amount, amount+1, 2)
    return np.roll(np.roll(img, ox, -1), oy, -2), ox, oy

def reverse_jitter(img, ox, oy):
    return np.roll(np.roll(img, -ox, -1), -oy, -2)

def plot_image(img):
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')

def Image_reader_from_csv(path_of_image,image_colName,_df):
    plt.figure(figsize=(15,20))
    showAxis=430
    for i in _df[image_colName].head(9):
        #print(i)
        ax_pos=1
        showAxis=showAxis+ax_pos
        ax1=plt.subplot(showAxis)
        image = imread(path_of_image+'images/'+i, as_gray=True)
        image = imread(path_of_image+'images/'+i) # as_gray to extend the pixel
        imshow(image,ax=ax1)
        ax1.set_title(i+'-shape'+str(image.shape))
def Steps_1fitModel_2saveModel_3saveHistory(model_Version,train_gen,validation_gen,_model_design,ep_sz=3):
    _model=_model_design
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    STEP_SIZE_TRAIN=train_gen.n//train_gen.batch_size
    STEP_SIZE_VALID=validation_gen.n//validation_gen.batch_size
    _model_history=_model.fit_generator(generator=train_gen,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=validation_gen,
                        validation_steps=STEP_SIZE_VALID,
                        verbose=1,
                        epochs=ep_sz,
                        callbacks=[es]
    )

    #_model.save()
    modelFileName='model'+str(model_Version)+'.fit'
    joblib.dump(_model, modelFileName)
    save_history(_model_history, 'history'+str(model_Version)+'.bin')
    del STEP_SIZE_TRAIN,STEP_SIZE_VALID,_model_history,_model
    #return _model

    
def load_andPlotHistoryForModelVersion(model_Version):
    history = load_history('history'+str(model_Version)+'.bin')
    plot_compare(history) 
    del history

def load_modelForModelVersion(model_Version):
    model = load_model('history'+str(model_Version)+'.fit')
    return model
    
def make_prediction_generate_csv(modelVersion,test_gen,model):
    STEP_SIZE_TEST=test_gen.n//test_gen.batch_size
    test_gen.reset()
    pred=model.predict_generator(test_gen,steps=STEP_SIZE_TEST,verbose=1)

    #pred.shape
    def classLabing(x):
        if x<.5:
            return 0
        else:
            return 1
    predictions=[classLabing(i) for i in pred]
#     predicted_class_indices=np.argmax(pred,axis=1)
#     len(predicted_class_indices)

#     labels = (train_generator.class_indices)
#     labels = dict((v,k) for k,v in labels.items())
#     predictions = [labels[k] for k in predicted_class_indices]
#     len(predictions)

    filenames=test_gen.filenames
    len(filenames)
    results=pd.DataFrame({"image_names":filenames,
                           "emergency_or_not":predictions})
    print(results['emergency_or_not'].value_counts())
    csv_name='results_model'+ str(modelVersion) +'.csv'
    results.to_csv(csv_name,index=False)
    print("\n\n",csv_name," generated \n\n")

import pandas as pd
img_path="/kaggle/input/av-dataset/train_SOaYf6m/images"
sub=pd.read_csv("/kaggle/input/av-dataset/sample_submission_yxjOnvz.csv")
train=pd.read_csv("/kaggle/input/av-dataset/train_SOaYf6m/train.csv")
test=pd.read_csv("/kaggle/input/av-dataset/test_vc2kHdQ.csv")
sub.head()
print("count of 0 and 1 in target columns:\n",train['emergency_or_not'].value_counts(), "\n\nShape:",train.shape,"\n\n")
print(train.head())
train.isna().sum()
trdf=train.copy()
trdf['emergency_or_not']=trdf['emergency_or_not'].astype('str')
trdf.head()
print("\n\nShape:",test.shape,"\n\n")
print(test.head())
test.isna().sum()
def prepareData(validationSplit=1):
    from keras_preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    if validationSplit==1:
        train_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0.25)
        train_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)

        validation_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="validation",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
    elif validationSplit==0:
        train_datagen = ImageDataGenerator(rescale = 1./255)
        train_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
        validation_generator=""
    test_generator=test_datagen.flow_from_dataframe(
                                    dataframe=test, 
                                    directory=img_path,
                                    x_col="image_names",
                                    #subset="testing",
                                    y_col=None,
                                    seed=42,
                                    shuffle=False,
                                    class_mode=None,
                                    target_size=(224,224), 
                                    batch_size=2)
    return train_generator,validation_generator,test_generator
#train_generator,validation_generator,test_generator=prepareData()
#del train_generator,validation_generator,test_generator
#test_generator

def prepareAugmentedData(validation_split=1):
    from keras_preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale = 1./255)
    if validation_split==1:
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                 validation_split=0.25,
                                 shear_range = 0.2,
                                 featurewise_center=False,            ## Set input mean to 0 over the dataset
                                 samplewise_center=False,             ## Set each sample mean to 0
                                 featurewise_std_normalization=False, ## Divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  ## Divide each input by its std
                                 zca_whitening=False,                 ## Apply ZCA whitening
                                 rotation_range=10,                   ## Randomly rotate images in the range (degrees, 0 to 180)
                                 zoom_range = 0.2,                    ## Randomly zoom image 
                                 width_shift_range=0.1,               ## Randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1,              ## Randomly shift images vertically (fraction of total height)
                                 horizontal_flip=True,               ## Randomly flip images horizontally
                                 vertical_flip=True)                 ## Randomly flip images vertically

        train_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
        validation_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="validation",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
    elif validation_split==0:
        #train Not splittling into validation dataset
        train_datagen = ImageDataGenerator(rescale = 1./255,
                                 featurewise_center=False,            ## Set input mean to 0 over the dataset
                                 samplewise_center=False,             ## Set each sample mean to 0
                                 featurewise_std_normalization=False, ## Divide inputs by std of the dataset
                                 samplewise_std_normalization=False,  ## Divide each input by its std
                                 zca_whitening=False,                 ## Apply ZCA whitening
                                 rotation_range=10,                   ## Randomly rotate images in the range (degrees, 0 to 180)
                                 zoom_range = 0.1,                    ## Randomly zoom image 
                                 width_shift_range=0.1,               ## Randomly shift images horizontally (fraction of total width)
                                 height_shift_range=0.1,              ## Randomly shift images vertically (fraction of total height)
                                 horizontal_flip=False,               ## Randomly flip images horizontally
                                 vertical_flip=False)                 ## Randomly flip images vertically
        train_generator=train_datagen.flow_from_dataframe(
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
        validation_generator = ''
    test_generator=test_datagen.flow_from_dataframe(
                                    dataframe=test, 
                                    directory=img_path,
                                    x_col="image_names",
                                    y_col=None,
                                    seed=42,
                                    shuffle=False,
                                    class_mode=None,
                                    target_size=(224,224), 
                                    batch_size=2)
    return train_generator,validation_generator,test_generator
#del 
#train_generator,validation_generator,test_generator=prepareData()
#test_generator
def generate_data(augment=1,validation_split=1):
    if augment == 0:
        return prepareData(validation_split)
    elif augment == 1:
        return prepareAugmentedData(validation_split)
train_generator,validation_generator,test_generator=generate_data(augment=0,validation_split=1)
#del train_generator,validation_generator,test_generator
def numba():
    from numba import cuda
    cuda.select_device(0)
    cuda.close()
#numba()
import gc
gc.collect()
def execute_Model_v16():
    """
    This is different from model_call1 
    Here train_Generater are prequisite
    """
    import gc    
    for lr in [.001]:
        for i in range(15): gc.collect()
        SEED=26
        os.environ['PYTHONHASHSEED']=str(SEED)
        random.seed(SEED)
        np.random.seed(SEED)
        tf.random.set_seed(SEED)


        val_losslog_score=[]
        val_acc_score=[]
        train_losslog_score=[]
        train_acc_score=[]

        runs=1
        print("Runs planned = ",runs)
        for run_iter in range(0,runs):
            print("------- Executing run :",run_iter+1 )
            model = Sequential()
            val_score,train_score=Model_v16(train_generator,validation_generator,test_generator,"3",model,lr,0,1)
            
            val_losslog_score.append(val_score[0])
            val_acc_score.append(val_score[1])
            train_losslog_score.append(train_score[0])
            train_acc_score.append(train_score[1])
            print(val_score,train_score)
            del val_score
        print("Train (loss_log,acc) of ",runs,"runs:",np.mean(train_losslog_score),np.mean(train_acc_score))
        print("Valid (loss_log,acc) of ",runs,"runs:",np.mean(val_losslog_score),np.mean(val_acc_score))

        del model,train_losslog_score,train_acc_score,val_losslog_score,val_acc_score
        for i in range(15): gc.collect()

#kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
def Model_v16(train_generator,validation_generator,test_generator,model_Version_text,model,_lr=.03,augment=0,generate_test_csv=1):
    # Initialising the CNN
    print("----------------------------------------Learning Rate:",_lr )
    for i in range(15): gc.collect()    
    
    model.add(Conv2D(20, (3,3), activation='relu', padding='same', name='conv_1',input_shape=(224, 224, 3)
            , kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.05)))
    
    model.add(Conv2D(50, (3,3), activation='relu', padding='same', name='conv_12',input_shape=(224, 224, 2)
           , kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.05) ))
    model.add(MaxPooling2D((2, 2), name='maxpool_1'))  # DownSampling
    model.add(Conv2D(50, (3,3), activation='relu', padding='same', name='conv_13',input_shape=(224, 224, 2)
       , kernel_regularizer=l2(0.0001),bias_regularizer=l2(0.0005) ))
    #model.add(BatchNormalization(name='batchNormalisation'))
    #,kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)
    #model.add(SeparableConv2D(120,(5,5), activation='relu', padding='same', name='Sepconv_1_1'))
    #model.add(BatchNormalization(name='batchNormalisation1'))
    #model.add(Dropout(0.5))
    
    #model.add(Conv2D(60, (3,3), activation='relu', padding='same', name='conv_13',input_shape=(224, 224, 1)
     #       ))
    #model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    #model.add(Dropout(0.05))
    #model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_1_1',
    #                 kernel_regularizer=l2(0.01),bias_regularizer=l2(0.01)))
    #model.add(Conv2D(16, (3, 3), activation='relu', padding='same', name='conv_1_2'))
    
    #model.add(Dropout(0.15))
    #model.add(MaxPooling2D((2, 2), name='maxpool_2'))  # DownSampling
    model.add(Dropout(0.1))
    #model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1_2'))
    #model.add(Dropout(0.1))
    #model.add(MaxPooling2D((2, 2), name='maxpool_2'))
    model.add(Flatten())
    
    #  drop outs
    model.add(Dropout(0.1))
    
    #model.add(Dense(30, activation='relu', name='dense1'))
    #model.add(Dropout(0.8))
    #model.add(Dense(100, activation='relu', name='dense2'))
    
    #model.add(Dropout(0.40))
    model.add(Dense(1, activation='sigmoid', name='output'))

    
    optimizer=1
    # Model compilation
    def compileModel(Adam_orRMSProp=2):
        if Adam_orRMSProp ==2:
            RMSprp=tf.keras.optimizers.RMSprop(
            learning_rate=0.0001, rho=0.99, momentum=0.4, epsilon=1e-010, centered=True,
            name='RMSprop'
            )
            model.compile(optimizer=RMSprp, loss='binary_crossentropy', metrics=['accuracy'])
        else:
            model.compile(optimizer=Adam(lr=.001), loss='binary_crossentropy', metrics=['accuracy'])
    compileModel(optimizer)
    # model fit and then save model and save history
    Steps_1fitModel_2saveModel_3saveHistory(model_Version_text,train_generator,validation_generator,model,7)
    
    # plot modelfit history
    load_andPlotHistoryForModelVersion(model_Version_text)
    
    # load model from file
    
    loaded_model_fromFile= joblib.load('model' + str(model_Version_text)+ ".fit")
    #if Optimiser is RMSprop, we need to recomple the model after loading from file... but not for Adam
    if optimizer ==2 :
        loaded_model_fromFile.compile(optimizer=RMSprp, loss='binary_crossentropy', metrics=['accuracy'])
    # Evaluate validattion data set from the file loaded from saved file
    # Evaluation matrix logloss, accuracy
    STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
    val_score=loaded_model_fromFile.evaluate_generator(generator=validation_generator,steps=STEP_SIZE_VALID)

    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    train_score=loaded_model_fromFile.evaluate_generator(generator=train_generator,steps=STEP_SIZE_TRAIN)
    
    # making prediction on test data
    if generate_test_csv==1:
        make_prediction_generate_csv(1,test_generator,loaded_model_fromFile)
        
    model.summary()
    gc.collect()
    del model,loaded_model_fromFile,train_generator,validation_generator,test_generator,STEP_SIZE_VALID,model_Version_text
    gc.collect()
    return val_score,train_score

import gc
# from keras.models import Model
# from keras.layers import Dense
from keras.optimizers import RMSprop
# from keras.layers import Flatten,Input
# from keras.layers import Dense, Dropout, Activation
# from tensorflow.keras import activations
from keras.regularizers import l2
# from keras.layers import Conv2D, SeparableConv2D, Dense, Flatten, concatenate, multiply, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input
execute_Model_v16()
img_path
label_col="emergency_or_not"
def import_vgg16_predict_training_data():
    # This will load the whole VGG16 network, including the top Dense layers.
    # Note: by specifying the shape of top layers, input tensor shape is forced
    # to be (224, 224, 3), therefore you can use it only on 224x224 images.
    
    from tensorflow.keras.preprocessing.image import load_img,img_to_array
    from keras.applications.resnet50 import preprocess_input,decode_predictions
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import Flatten
    from keras.layers import Dense, Dropout, Activation
    from tensorflow.keras import activations
    from keras.regularizers import l2
    from keras.layers import Conv2D, SeparableConv2D, Dense, Flatten, concatenate, multiply, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Input
    # Making predicton on training data with class 1 data using VGA model

    #img_Fullpath=img_path+"10.jpg"
    def ImageShow(imageFull_path,showAxis=111):
        ax1=plt.subplot(showAxis)
        ax1.figsize=(17,18)
        image = imread(imageFull_path)
        imshow(image,ax=ax1)
        ax1.set_title('shape'+str(image.shape))
    #ImageShow(img_Fullpath)

    def predictionTrainingClass1Data(_model):
        vgg16_pred=[]
        img_loc=0
        img_axis=220
        for img in train[train[label_col]==1].image_names.head(4):

            # Step 2- Image pre-processing
            #img_path=img_path
            image = load_img(img_path+"/"+img, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)

            # Step 3- prediction
            yhat = _model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0][1]
            img_loc=img_loc+1
            print(img," is  predicted as ",label)
            vgg16_pred.append(label)
            ImageShow(img_path+"/"+img,img_axis+img_loc)

    # example of loading the vgg16 model
    from keras.applications.vgg16 import VGG16
    
    # Step 1- load model
    model = VGG16(weights='imagenet',input_shape=(224, 224, 3),include_top=True)
    # summarize the model
    #model.summary()
    # Let pridict the first 5 images of training data with class1 
    import gc 
    
    predictionTrainingClass1Data(model)
    gc.collect()
    del model
    gc.collect()
    
#del model    
import_vgg16_predict_training_data()
# create an empty python list
X = []

# go through all the image locations one by one
for img_name in train.image_names:
    # read the image from location
    image_path = img_path +"/"+ img_name
    img = plt.imread(image_path)
    # pile it one over the other
    X.append(img)
    
# convert this python list to a single numpy array
#X = np.array(X)

# create an empty python list
X_test = []

# go through all the image locations one by one
for img_name in test.image_names:
    image_path = img_path +"/"+ img_name
    # read the image from location
    img = plt.imread(image_path)
    # pile it one over the other
    X_test.append(img)
    
# convert this python list to a single numpy array
#X = np.array(X)

# convert this python list to a single numpy array
X = np.array(X)
X_test = np.array(X_test)        

#getting the labels for images
y = train.emergency_or_not.values


#show maximum and minimum values for the image array
print("min,max Before Preprocess:",X.min(), X.max())

#preprocess input images accordiing to requirements of VGG16 model ** Important step
X = preprocess_input(X, mode='tf')
X_test = preprocess_input(X_test, mode='tf')

# See the data after preprocessing data
print("min,max After Preprocess:",X.min(), X.max())

# splitting the dataset into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=42)

def import_vgg16_predict_training_data2():
        #     # Making predicton on training data with class 1 data using VGA model
    img_path="/kaggle/input/train_SOaYf6m/images/"
    def makingPrediction(_model):
        vgg16_pred=[]
        for img in train[train[label_col]==1].image_names:

            # Step 2- Image pre-processing
            image = load_img(img_path+img, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)

            # Step 3- makeing prediction
            yhat = _model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0][1]
            print(img," is ",label)
            vgg16_pred.append(label)
            return vgg16_pred
    img_Fullpath=img_path+"10.jpg"
    def ImageShow(imageFull_path,showAxis=111):
        ax1=plt.subplot(showAxis)
        ax1.figsize=(8,8)
        image = imread(imageFull_path)
        imshow(image,ax=ax1)
        ax1.set_title('shape'+str(image.shape))
    #ImageShow(img_Fullpath)
    
    
    
    
    #----------------------------------------------------MAIN FUNCTION PART -------------------------------------#

    # Step 1------------ Set the Base model as VGG16
    # Dont inclue Top and the your personal data input size
    #input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights='imagenet',input_shape=(224, 224, 3),include_top=False)
    print('Model loaded.')

    # Extract the last layer of VGG16
    vgg16_output = base_model.output
    base_model.summary()
    
    img_name
    # step 2 -------------Lets chain into new mode
    flatten = (Dropout(0.5))(vgg16_output)
    flatten = Flatten(name='flatten')(flatten)
    flatten = (Dropout(0.5))(flatten)
    dense = Dense(132, activation='relu', kernel_initializer='he_normal', name='fc1')(flatten)
    dense = Dense(64, activation='relu', kernel_initializer='he_normal', name='fc2')(dense)
    pred = Dense(units=1, activation='sigmoid', name='prediction')(dense)

    # Lets build the new bodel#
    new_model = Model(input=base_model.input, output=pred)
    new_model.summary()

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # To "freeze" a layer means to exclude it from training
    for layer in new_model.layers:
       if layer.name in ['fc1', 'fc2', 'prediction']:
           continue
    layer.trainable = False

    # compile the model with a SGD/momentum optimizer
    #from keras import optimizers
    optimizer=Adam(lr=.0001)
    new_model.compile(optimizer, loss='binary_crossentropy',metrics=['accuracy'])
    # train model using features generated from VGG16 model
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
    new_model.fit(X_train, y_train, epochs=15, validation_data=(X_valid, y_valid),callbacks=[es])
    #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    #,
    #
    #new_model.summary()
    #Steps_1fitModel_2saveModel_3saveHistory(1,new_model,15)
    return new_model
    
new_model= import_vgg16_predict_training_data2()
def prepareData(validationSplit=1):
    from keras_preprocessing.image import ImageDataGenerator
    test_datagen = ImageDataGenerator(rescale = 1./255)
    
    if validationSplit==1:
        train_datagen = ImageDataGenerator(rescale = 1./255,validation_split=0.25)
        train_generator=train_datagen.flow_from_dataframe(
                                        preprocessing_function=preprocess_input,
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)

        validation_generator=train_datagen.flow_from_dataframe(
                                        preprocessing_function=preprocess_input,
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="validation",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
    elif validationSplit==0:
        train_datagen = ImageDataGenerator(rescale = 1./255)
        train_generator=train_datagen.flow_from_dataframe(
                                        preprocessing_function=preprocess_input,
                                        dataframe=trdf, 
                                        directory=img_path, 
                                        x_col="image_names", 
                                        subset="training",
                                        seed=42,
                                        shuffle=False,
                                        y_col="emergency_or_not", 
                                        class_mode="binary", 
                                        target_size=(224,224), 
                                        batch_size=16)
        validation_generator=""
    test_generator=test_datagen.flow_from_dataframe(
                                    preprocessing_function=preprocess_input,
                                    dataframe=test, 
                                    directory=img_path,
                                    x_col="image_names",
                                    #subset="testing",
                                    y_col=None,
                                    seed=42,
                                    shuffle=False,
                                    class_mode=None,
                                    target_size=(224,224), 
                                    batch_size=2)
    return train_generator,validation_generator,test_generator
train_generator,validation_generator,test_generator=prepareData()
#del train_generator,validation_generator,test_generator
#test_generator

def pickFewTopLayers_TransferLearning_VGA16(PoollayerName_toPick="block2_pool",pickLayer=1):
    # download Vgg16 model
    vgg_model = VGG16(weights='imagenet',
                                   include_top=False,
                                   input_shape=(224, 224, 3))
    
    # Creating dictionary that maps layer names to the layers
    layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])
    if pickLayer ==0:

        
        
        # Getting output tensor of the last VGG layer that we want to include
        print(vgg_model.input)
        #print(vgg_model.layers)
        print(vgg_model.summary())
        print(vgg_model.output)
        
        print("Vgg model layers list.......\n")
        print(list(layer_dict))
    elif  pickLayer ==1:
        x = layer_dict[PoollayerName_toPick].output
        # Stacking a new simple convolutional network on top of it    
        #x = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',name="MyConv_layer")(x)
        #x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(1, activation='softmax')(x)

        # Creating new model. Please note that this is NOT a Sequential() model.
        from keras.models import Model
        custom_model = Model(input=vgg_model.input, output=x)
        # Make sure that the pre-trained bottom layers are not trainable
        
        # freezing layers taken from pretrained model ( transfer learning)
            # Finding the Index
        layerList=list(layer_dict)
        for i in range(0,len(layerList)):
            if layerList[i]==PoollayerName_toPick:
                layerIndex=i+1
            # Freeze all layers till the given layer name
        for layer in custom_model.layers[:layerIndex]:
            layer.trainable = False
            
        #custom_model.trainable = True
        #custom_model.summary()
        # Model compilation
        custom_model.compile(optimizer=Adam(lr=.001), loss='binary_crossentropy', metrics=['accuracy'])
        Steps_1fitModel_2saveModel_3saveHistory(1,train_generator,validation_generator,custom_model,15)
        del custom_model,vgg_model
#pickFewTopLayers_TransferLearning_VGA16('block2_pool',pickLayer=1)
def import_vgg16_predict_training_data2_withoutWeight():
        #     # Making predicton on training data with class 1 data using VGA model
    img_path="/kaggle/input/train_SOaYf6m/images/"
    def makingPrediction(_model):
        vgg16_pred=[]
        for img in train[train[label_col]==1].image_names:

            # Step 2- Image pre-processing
            image = load_img(img_path+img, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)

            # Step 3- makeing prediction
            yhat = _model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0][1]
            print(img," is ",label)
            vgg16_pred.append(label)
            return vgg16_pred
    img_Fullpath=img_path+"10.jpg"
    def ImageShow(imageFull_path,showAxis=111):
        ax1=plt.subplot(showAxis)
        ax1.figsize=(8,8)
        image = imread(imageFull_path)
        imshow(image,ax=ax1)
        ax1.set_title('shape'+str(image.shape))
    #ImageShow(img_Fullpath)
    
    
    #----------------------------------------------------MAIN FUNCTION PART -------------------------------------#

    # Step 1------------ Set the Base model as VGG16
    # Dont inclue Top and the your personal data input size
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights=None,input_shape=(224, 224, 3),include_top=False)
    print('Model loaded.')

    # Extract the last layer of VGG16
    vgg16_output = base_model.output

    # step 2 -------------Lets chain into new mode
    flatten = (Dropout(0.5))(vgg16_output)
    flatten = Flatten(name='flatten')(flatten)
    flatten = (Dropout(0.5))(flatten)
    #dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='fc1')(flatten)
    #dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='fc2')(dense)
    pred = Dense(units=1, activation='softmax', kernel_initializer='he_normal', name='prediction')(flatten)

    # Lets build the new bodel
    new_model = Model(input=base_model.input, output=pred)
    #new_model.summary()

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # To "freeze" a layer means to exclude it from training
    for layer in new_model.layers:
        #if layer.name in ['fc1', 'fc2', 'prediction']:
        #    continue
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    from keras import optimizers
    new_model.compile(loss='binary_crossentropy',
    #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    optimizer=optimizers.Adam(lr=.001),
    metrics=['accuracy'])
    #new_model.summary()
    Steps_1fitModel_2saveModel_3saveHistory(1,train_generator,validation_generator,new_model,15)
    
import_vgg16_predict_training_data2_withoutWeight()
def import_vgg16_predict_training_data2_withoutWeight(PoollayerName_toPick="block2_pool",pickLayer=1):
        #     # Making predicton on training data with class 1 data using VGA model
    img_path="/kaggle/input/train_SOaYf6m/images/"
    def makingPrediction(_model):
        vgg16_pred=[]
        for img in train[train[label_col]==1].image_names:

            # Step 2- Image pre-processing
            image = load_img(img_path+img, target_size=(224, 224))
            # convert the image pixels to a numpy array
            image = img_to_array(image)
            # reshape data for the model
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            # prepare the image for the VGG model
            image = preprocess_input(image)

            # Step 3- makeing prediction
            yhat = _model.predict(image)
            # convert the probabilities to class labels
            label = decode_predictions(yhat)
            # retrieve the most likely result, e.g. highest probability
            label = label[0][0][1]
            print(img," is ",label)
            vgg16_pred.append(label)
            return vgg16_pred
    img_Fullpath=img_path+"10.jpg"
    def ImageShow(imageFull_path,showAxis=111):
        ax1=plt.subplot(showAxis)
        ax1.figsize=(8,8)
        image = imread(imageFull_path)
        imshow(image,ax=ax1)
        ax1.set_title('shape'+str(image.shape))
    #ImageShow(img_Fullpath)
    

#         layerList=list(layer_dict)
#         for i in range(0,len(layerList)):
#             if layerList[i]==PoollayerName_toPick:
#                 layerIndex=i+1
#             # Freeze all layers till the given layer name
#         for layer in custom_model.layers[:layerIndex]:
#             layer.trainable = False
            
    
    
    
    #----------------------------------------------------MAIN FUNCTION PART -------------------------------------#

    # Step 1------------ Set the Base model as VGG16
    # Dont inclue Top and the your personal data input size
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(weights=None,input_shape=(224, 224, 3),include_top=False)
    print('Model loaded.')

    # Extract the last layer of VGG16
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    vgg16_output = layer_dict[PoollayerName_toPick].output

    # step 2 -------------Lets chain into new mode
    flatten = (Dropout(0.5))(vgg16_output)
    flatten = Flatten(name='flatten')(flatten)
    flatten = (Dropout(0.5))(flatten)
    #dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='fc1')(flatten)
    #dense = Dense(32, activation='relu', kernel_initializer='he_normal', name='fc2')(dense)
    pred = Dense(units=1, activation='softmax', kernel_initializer='he_normal', name='prediction')(flatten)

    # Lets build the new bodel
    new_model = Model(input=base_model.input, output=pred)
    #new_model.summary()

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    # To "freeze" a layer means to exclude it from training
    for layer in new_model.layers:
        layer.trainable = True

    # compile the model with a SGD/momentum optimizer
    from keras import optimizers
    new_model.compile(loss='binary_crossentropy',
    #optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
    optimizer=optimizers.Adam(lr=.0000000001),
    metrics=['accuracy'])
    #new_model.summary()
    Steps_1fitModel_2saveModel_3saveHistory(1,train_generator,validation_generator,new_model,5)
    
import_vgg16_predict_training_data2_withoutWeight()
from kerastuner import HyperModel
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D
)

class CNNHyperModel(HyperModel):

    def __init__(self, input_shape, num_classes):
        
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self, hp):
        print("\n\n---------------------------------------- Architechting Keras Model ------------------------------\n\n")
        #print("step 1.model initialisation ")
        model = keras.Sequential()
        
        #print("step 2.  Conv2D layer 1 ")
        hp_filters=hp.Choice('num_filters',values=[20,24,28,32],default=32)
        model.add(
            Conv2D(
                filters=hp_filters,
                kernel_size=3,
                activation='relu',
                input_shape=self.input_shape
            )
        )
        model.add(
        Dropout(rate=hp.Float(
            'dropout_1',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
        ))
    )
        model.add(MaxPooling2D((2, 2), name='maxpool_4'))
        model.add(
        Dropout(rate=hp.Float(
            'dropout_2',
            min_value=0.0,
            max_value=0.5,
            default=0.25,
            step=0.05,
        ))
    )
#         model.add(layers.Dense(units=hp.Int('units',
#                                     min_value=32,
#                                     max_value=512,
#                                     step=32),
#                        activation='relu'))
        
#         #print("step 2.  Conv2D layer 2 ")
#         model.add(
#             Conv2D(
#                 filters=16,
#                 activation='relu',
#                 kernel_size=3
#             )
#         )
        
#         #print("step 2.  max Pooling layaer 1 ")
#         model.add(MaxPooling2D(pool_size=2))
#         model.add(
#             Dropout(rate=hp.Float(
#                 'dropout_1',
#                 min_value=0.0,
#                 max_value=0.5,
#                 default=0.25,
#                 step=0.05,
#             ))
#         )
        
#         #print("step 3.  Conv2D layer 1 ")
#         model.add(
#             Conv2D(
#                 filters=32,
#                 kernel_size=3,
#                 activation='relu'
#             )
#         )
        #print("step 3.  Conv2D layer 2 ")
#         model.add(
#             Conv2D(
#                 filters=hp.Choice(
#                     'num_filters',
#                     values=[32, 64],
#                     default=64,
#                 ),
#                 activation='relu',
#                 kernel_size=3
#             )
#         )
#         #print("step 3.  MaxPooling2D 2 ")
#         model.add(MaxPooling2D(pool_size=2))
        
        
#         model.add(
#             Dropout(rate=hp.Float(
#                 'dropout_2',
#                 min_value=0.0,
#                 max_value=0.5,
#                 default=0.25,
#                 step=0.05,
#             ))
#         )
        
        model.add(Flatten())
#         model.add(
#             Dense(
#                 units=hp.Int(
#                     'units',
#                     min_value=32,
#                     max_value=512,
#                     step=32,
#                     default=128
#                 ),
#                 activation=hp.Choice(
#                     'dense_activation',
#                     values=['relu', 'tanh', 'sigmoid'],
#                     default='relu'
#                 )
#             )
#         )
        
#         model.add(
#             Dropout(
#                 rate=hp.Float(
#                     'dropout_3',
#                     min_value=0.0,
#                     max_value=0.5,
#                     default=0.25,
#                     step=0.05
#                 )
#             )
#         )
#         #print("drop out/ Flattening/drop out setting done")
        
        model.add(Dense(self.num_classes, activation='softmax'))

        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    'learning_rate',
                    min_value=.0008,
                    max_value=.001,
                    #sampling='LOG',
                    default=.005
                )
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        print("final dense layer added ")
        return model
NUM_CLASSES = 1  # cifar10 number of classes
import gc
gc.collect()
#del train_generator,validation_generator,test_generator#,best_model,tuner,hypermodel
gc.collect()

INPUT_SHAPE = (224, 224, 3)  # cifar10 images input shape
hypermodel1 = CNNHyperModel(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)

from kerastuner.tuners import RandomSearch

tuner = RandomSearch(
    hypermodel1,
    objective='val_accuracy',
    seed=SEED,
    max_trials=5,
    executions_per_trial=2,
    directory='random_search',
    project_name='cifar10'
)
# from kerastuner.tuners import Hyperband
# tuner = Hyperband(
#     hypermodel,
#     max_epochs=10,
#     objective='val_accuracy',
#     seed=26,
#     executions_per_trial=2,
#     directory='hyperband',
#     project_name='cifar10'
# )

train_generator,validation_generator,test_generator=prepareAugmentedData()
tuner.search(train_generator, steps_per_epoch=1, epochs=5, validation_data=validation_generator)
# Show a summary of the search and evaluating the best model
#best_model = tuner.get_best_models(num_models=1)[0]
#best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]
#print(best_hps.get('learning_rate'))
#STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#print(best_model.evaluate_generator(generator=validation_generator,steps=STEP_SIZE_VALID))
#best_model.summary()
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
