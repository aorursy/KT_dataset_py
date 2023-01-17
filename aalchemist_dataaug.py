# Importing necessary libraries
import numpy as np
import os
import glob
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from keras.utils import to_categorical
from keras.layers import Dense, Input, Conv2D, Flatten, MaxPool2D, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
#Declaring constants
FIG_WIDTH=20 # Width of figure
HEIGHT_PER_ROW=3 # Height of each row when showing a figure which consists of multiple rows
RESIZE_DIM=64 # The images will be resized to 28x28 pixels
data_dir=os.path.join('..','input')
paths_train_a=glob.glob(os.path.join(data_dir,'training-a','*.png'))
paths_train_b=glob.glob(os.path.join(data_dir,'training-b','*.png'))
paths_train_e=glob.glob(os.path.join(data_dir,'training-e','*.png'))
paths_train_c=glob.glob(os.path.join(data_dir,'training-c','*.png'))
paths_train_d=glob.glob(os.path.join(data_dir,'training-d','*.png'))
paths_train_all=paths_train_a+paths_train_b+paths_train_c+paths_train_d+paths_train_e

paths_test_a=glob.glob(os.path.join(data_dir,'testing-a','*.png'))
paths_test_b=glob.glob(os.path.join(data_dir,'testing-b','*.png'))
paths_test_e=glob.glob(os.path.join(data_dir,'testing-e','*.png'))
paths_test_c=glob.glob(os.path.join(data_dir,'testing-c','*.png'))
paths_test_d=glob.glob(os.path.join(data_dir,'testing-d','*.png'))
paths_test_f=glob.glob(os.path.join(data_dir,'testing-f','*.png'))+glob.glob(os.path.join(data_dir,'testing-f','*.JPG'))
paths_test_auga=glob.glob(os.path.join(data_dir,'testing-auga','*.png'))
paths_test_augc=glob.glob(os.path.join(data_dir,'testing-augc','*.png'))
paths_test_all=paths_test_a+paths_test_b+paths_test_c+paths_test_d+paths_test_e+paths_test_f+paths_test_auga+paths_test_augc

path_label_train_a=os.path.join(data_dir,'training-a.csv')
path_label_train_b=os.path.join(data_dir,'training-b.csv')
path_label_train_e=os.path.join(data_dir,'training-e.csv')
path_label_train_c=os.path.join(data_dir,'training-c.csv')
path_label_train_d=os.path.join(data_dir,'training-d.csv')
def get_key(path):
    # seperates the key of an image from the filepath
    key=path.split(sep=os.sep)[-1]
    return key

def get_data(paths_img,path_label=None,resize_dim=None):
    
    X=[] # initialize empty list for resized images
    for i,path in enumerate(paths_img):
        img=cv2.imread(path) # images loaded in color (BGR)
        #(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img=cv2.cvtColor(img,cv2.cv2.COLOR_BGR2GRAY)
        
        #img = cv2.threshold(img, 30, 255, cv2.THRESH_BINARY)[1]
        #img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        if resize_dim is not None:
            img=cv2.resize(img,(resize_dim,resize_dim),interpolation=cv2.INTER_AREA) # resize image to 28x28
#         X.append(np.expand_dims(img,axis=2)) # expand image to 28x28x1 and append to the list.
        img = np.reshape(img,(resize_dim,resize_dim,1))
        X.append(img) # expand image to 28x28x1 and append to the list
        # display progress
        if i==len(paths_img)-1:
            end='\n'
        else: end='\r'
        print('processed {}/{}'.format(i+1,len(paths_img)),end=end)
        
    X=np.array(X) # tranform list to numpy array
    if  path_label is None:
        return X
    else:
        df = pd.read_csv(path_label) # read labels
        df=df.set_index('filename') 
        y_label=[df.loc[get_key(path)]['digit'] for path in  paths_img] # get the labels corresponding to the images
        y=to_categorical(y_label,10) # transfrom integer value to categorical variable
        return X, y
        
def imshow_group(X,y,y_pred=None,n_per_row=10,phase='processed'):
    
    n_sample=len(X)
    X = np.reshape(X,(n_sample,RESIZE_DIM,RESIZE_DIM))
    img_dim=X.shape[1]
    j=np.ceil(n_sample/n_per_row)
    fig=plt.figure(figsize=(FIG_WIDTH,HEIGHT_PER_ROW*j))
    for i,img in enumerate(X):
        plt.subplot(j,n_per_row,i+1)
#         img_sq=np.squeeze(img,axis=2)
#         plt.imshow(img_sq,cmap='gray')
        plt.imshow(img)
        if phase=='processed':
            plt.title(np.argmax(y[i]))
        if phase=='prediction':
            top_n=3 # top 3 predictions with highest probabilities
            ind_sorted=np.argsort(y_pred[i])[::-1]
            h=img_dim+4
            for k in range(top_n):
                string='pred: {} ({:.0f}%)\n'.format(ind_sorted[k],y_pred[i,ind_sorted[k]]*100)
                plt.text(img_dim/2, h, string, horizontalalignment='center',verticalalignment='center')
                h+=4
            if y is not None:
                plt.text(img_dim/2, -4, 'true label: {}'.format(np.argmax(y[i])), 
                         horizontalalignment='center',verticalalignment='center')
        plt.axis('off')
    plt.show()

def create_submission(predictions,keys,path):
    result = pd.DataFrame(
        predictions,
        columns=['label'],
        index=keys
        )
    result.index.name='key'
    result.to_csv(path, index=True)
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util

def random_rotation(image_array: ndarray):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    random_degree = random.uniform(-25, 25)
    return sk.transform.rotate(image_array, random_degree)

def random_noise(image_array: ndarray):
    # add random noise to the image
    return sk.util.random_noise(image_array)
X_train_a,y_train_a=get_data(paths_train_a,path_label_train_a,resize_dim=RESIZE_DIM)
X_train_b,y_train_b=get_data(paths_train_b,path_label_train_b,resize_dim=RESIZE_DIM)
X_train_c,y_train_c=get_data(paths_train_c,path_label_train_c,resize_dim=RESIZE_DIM)
X_train_d,y_train_d=get_data(paths_train_d,path_label_train_d,resize_dim=RESIZE_DIM)
X_train_e,y_train_e=get_data(paths_train_e,path_label_train_e,resize_dim=RESIZE_DIM)
X_train_all=np.concatenate((X_train_a,X_train_b,X_train_c,X_train_d,X_train_e),axis=0)
y_train_all=np.concatenate((y_train_a,y_train_b,y_train_c,y_train_d,y_train_e),axis=0)
X_train_all.shape, y_train_all.shape
X_train_all=X_train_all/255.0

X_test_a=get_data(paths_test_a,resize_dim=RESIZE_DIM)
X_test_b=get_data(paths_test_b,resize_dim=RESIZE_DIM)
X_test_c=get_data(paths_test_c,resize_dim=RESIZE_DIM)
X_test_d=get_data(paths_test_d,resize_dim=RESIZE_DIM)
X_test_e=get_data(paths_test_e,resize_dim=RESIZE_DIM)
X_test_f=get_data(paths_test_f,resize_dim=RESIZE_DIM)
X_test_auga=get_data(paths_test_auga,resize_dim=RESIZE_DIM)
X_test_augc=get_data(paths_test_augc,resize_dim=RESIZE_DIM)
X_test_all=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))
X_test_all1=np.concatenate((X_test_a,X_test_b,X_test_c,X_test_d,X_test_e,X_test_f,X_test_auga,X_test_augc))
X_test_all=X_test_all/255.0
indices=list(range(len(X_train_all)))
np.random.seed(23)
np.random.shuffle(indices)

ind=int(len(indices)*0.80)
# train data
X_train=X_train_all[indices[:ind]] 
y_train=y_train_all[indices[:ind]]
# validation data
X_val=X_train_all[indices[-(len(indices)-ind):]] 
y_val=y_train_all[indices[-(len(indices)-ind):]]

from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.layers import  GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Concatenate,LeakyReLU
def get_model():
    
    image_input = Input( shape = (64, 64, 1), name = 'images' )
    #angle_input = Input( shape = [1], name = 'angle' )
    activation = 'relu'
    bn_momentum = 0.99
    
    img_1 = BatchNormalization(momentum=bn_momentum)( image_input )
    img_1 =   Conv2D( 32, kernel_size = (3, 3), activation = activation, padding = 'same' )(img_1)

    img_1 = MaxPooling2D( (2,2)) (img_1 )
    
    img_1 = Conv2D( 64, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_1) )
    img_1 = MaxPooling2D( (2,2), name='skip1' ) ( img_1 )
    
     # Residual block
    img_2 =  Conv2D( 128, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_1))
    img_2 = Conv2D( 64, name='img2', kernel_size = (3, 3), 
                    activation = activation, padding = 'same' ) ( (BatchNormalization(momentum=bn_momentum)) (img_2) )
    
    img_2 = add( [img_1, img_2] )
    img_2 = MaxPooling2D( (2,2), name='skip2' ) ( img_2 )
    
    # Residual block
    img_3 = Conv2D( 128, kernel_size = (3, 3), activation = activation, 
                    padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_2))
    img_3 =  Conv2D( 64, name='img3', kernel_size = (3, 3), 
                    activation = activation, padding = 'same' ) ((BatchNormalization(momentum=bn_momentum)) (img_3))
    
    img_res = add( [img_2, img_3] )

    # Filter residual output
    img_res = Conv2D( 128, kernel_size = (3, 3), 
                      activation = activation ) ((BatchNormalization(momentum=bn_momentum)) (img_res))
    
    # Can you guess why we do this? Hint: Where did Flatten go??
    img_res = GlobalMaxPooling2D(name='global_pooling') ( img_res )
    
    # What is this? Hint: We have 2 inputs. An image and a number.
    #cnn_out = Concatenate(name='What_happens_here')( img_res )

    dense_layer = Dropout( 0.5 ) (Dense(128, activation = activation) (img_res)) 
    dense_layer = Dropout( 0.5 )  (Dense(64, activation = activation) (dense_layer)) 
    output = Dense( 10, activation = 'softmax' ) ( dense_layer )
    
    model = Model( image_input, output )

    opt = Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

    model.compile( loss = 'binary_crossentropy', 
                   optimizer = opt, 
                   metrics = ['accuracy'] )

    model.summary()

    return model
model=get_model()
model.summary()
from keras.preprocessing.image import ImageDataGenerator

#K.set_value(model.optimizer.lr,1e-3) # set the learning rate
# fit the model
X_train = np.array(X_train)
X_train = X_train.reshape( (-1,64,64,1))
X_train = X_train.astype('float32')
print("SHAPE",X_train_all.shape[0] )
image_gen = ImageDataGenerator(
    #samplewise_center=True,
    #featurewise_std_normalization=True,
    #featurewise_center=True,
    #rotation_range=30,
    #zca_epsilon=0.7,
    #zca_whitening=True,
    width_shift_range=.15,
    height_shift_range=.15,
    shear_range=0.4,
    zoom_range=0.5
    )
'''test_datagen = ImageDataGenerator(  
    #samplewise_center=True,
    #featurewise_std_normalization=True,
    #featurewise_center=True,
    zca_epsilon=0.7,
     zca_whitening=True
     
    )
test_datagen.fit(X_val)
for i in range(len(X_val)):
    X_val[i] = test_datagen.standardize(X_val[i])
test_datagen2 = ImageDataGenerator(  
    #samplewise_center=True,
    #featurewise_std_normalization=True,
    #featurewise_center=True,
    zca_epsilon=0.7,
     zca_whitening=True
    )
test_datagen2.fit(X_test_all)
for i in range(len(X_test_all)):
    # this is what you are looking for
    X_test_all[i] = test_datagen2.standardize(X_test_all[i])'''
path_model='model_simple_keras_starter.h5' # save model at this location after each epoch
#K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=get_model() # create the model
checkpointer = ModelCheckpoint('model_simple_keras_starter.h5', verbose=1, save_best_only=True)
image_gen.fit(X_train, augment=True)
h=model.fit_generator(image_gen.flow(X_train, y_train, batch_size=128),
          steps_per_epoch =  X_train.shape[0]//100, 
            epochs=30, 
            verbose=1, 
            validation_data=(X_val,y_val),
            shuffle=True,
            callbacks=[
                checkpointer,
            ]
            )
model.load_weights('model_simple_keras_starter.h5')
score = model.evaluate(X_val,y_val, verbose=0)
print(len(X_val))
print('Test accuracy:', score[1])
predictions_prob=model.predict(X_test_all) # get predictions for all the testing data
n_sample=200
np.random.seed(42)
ind=np.random.randint(0,len(X_test_all), size=n_sample)
imshow_group(X=X_test_all[ind],y=None,y_pred=predictions_prob[ind], phase='prediction')
labels=[np.argmax(pred) for pred in predictions_prob]
keys=[get_key(path) for path in paths_test_all ]
create_submission(predictions=labels,keys=keys,path='submission_simple_keras_starter_kindofgood2aug.csv')