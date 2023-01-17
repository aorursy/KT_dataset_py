import numpy as np

import os

import cv2

from sklearn.model_selection import train_test_split

import tensorflow as tf

from tensorflow import keras

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Dense, Activation,Dropout,Conv2D, MaxPooling2D,BatchNormalization, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import categorical_crossentropy

from tensorflow.keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.applications import imagenet_utils

from tensorflow.keras.preprocessing import image

from tensorflow.keras.models import Model, load_model, Sequential

from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt

import time

import os
# set ansi color values

Cblu ='\33[34m'

Cend='\33[0m'   # sets color back to default 

Cred='\033[91m'

Cblk='\33[39m'

Cgreen='\33[32m'

Cyellow='\33[33m'
d=r'/kaggle/input'

d_list=os.listdir(d)

d_path= os.path.join(d, d_list[0])

d_list=os.listdir(d_path)

next_path=os.path.join(d_path, d_list[0])

next_list=os.listdir(next_path)

deeper_path=os.path.join(next_path, next_list[0])

deeper_list=os.listdir(deeper_path)

input_path=os.path.join(deeper_path, 'images')

input_list=os.listdir(input_path)

size=len(input_list)

print ('length of images is ', size)
data_list=[]

labels=[]

for i,f in enumerate(input_list):    

    f_path=os.path.join(input_path, f)

    data_list.append(cv2.imread (f_path))

    labels.append (1 if f[0]=='P' else 0) 

labels = tf.keras.utils.to_categorical(labels, num_classes = 2)

Y=np.array(labels)

X=np.array(data_list)

print ('X shape is ', X.shape, ' Y shape is ', Y.shape)

# just check to make sure labels match with input file names 

for i in range (10):

    f_path=os.path.join(input_path, input_list[i])

    f=os.path.basename(f_path)

    print('label = ',labels[i], '    file names = ', f)
xTrain, x, yTrain, y =train_test_split(X, Y, test_size=.2)

xTest, xValid, yTest,yValid=train_test_split(x,y, test_size=.4)


train_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                horizontal_flip=True,

                samplewise_center=True,

                width_shift_range=.2,

                height_shift_range=.2,                

                samplewise_std_normalization=True).flow(xTrain, yTrain, )

val_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                samplewise_center=True,                

                samplewise_std_normalization=True).flow(xValid, yValid, shuffle=False)        

test_gen=ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,

                samplewise_center=True,                

                samplewise_std_normalization=True).flow(xTest, yTest, shuffle=False)



        
Top=False

weights=None

layer_cut=-6

lr_rate=.002

rand_seed=128

epochs=20

mobile = tf.keras.applications.mobilenet.MobileNet( include_top=True,

                                                           input_shape=(224,224,3),

                                                           pooling='avg', weights='imagenet',

                                                           alpha=1, depth_multiplier=1)  

                 

x=mobile.layers[layer_cut].output

x=Dense(128, kernel_regularizer = regularizers.l2(l = 0.015), activation='relu')(x)

x=Dropout(rate=.5, seed=rand_seed)(x)

predictions=Dense (2, activation='softmax')(x)

model = Model(inputs=mobile.input, outputs=predictions)

        

for layer in model.layers:

    layer.trainable=True

model.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])
class tr(tf.keras.callbacks.Callback):

    best_weights=model.get_weights()

    best_acc=0

    patience=10

    p_count=0

    focus='acc'

    def __init__(self):

        super(tr, self).__init__()

        self.best_acc = 0

        self.patience=10

        self.p_count=0

    def on_batch_end(self, batch, logs=None): 

        epoch=logs.get('epoch')

        acc=logs.get('accuracy')

        if tr.best_acc>.9:

            if tr.focus=='acc':

                msg='{0}\n with training accuracy= {1:7.4f} will now start adjusting learning rate based on validation loss\n{2}'

                print(msg.format(Cblu, tr.best_acc, Cend))

                tr.focus='val'

        else:

            if tr.best_acc<acc:

                #accuracy at batch end is better then highest accuracy thus far

                #msg='\non batch {0} accuracy improved from {1:7.4f}  to {2:7.4f} \n'

               # print(msg.format(batch + 1, tr.best_acc, acc ))

                tr.best_acc=acc

                tr.p_count=0

                tr.best_weights=model.get_weights()

           

            else:

                #accuracy on current batch was below highest accuracy thus far

                tr.p_count=tr.p_count + 1

                #msg='\n for batch {0} current accuracy {1:7.4f}  was below highest accuracy of {2:7.4f} for {3} batches'

                #print(msg.format(batch + 1, acc, tr.best_acc,tr.p_count))

                if tr.p_count >= tr.patience:

                    tr.p_count=0

                    lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                    new_lr=lr*.95

                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr) 

                    print('\n adjusted learning rate for batch {0} to {1}\n'.format(batch + 1, new_lr))

                    

class val(tf.keras.callbacks.Callback):

    best_loss=np.inf

    best_weights=tr.best_weights

    lr=float(tf.keras.backend.get_value(model.optimizer.lr))

    def __init__(self):

        super(val, self).__init__()

        self.best_loss=np.inf

        self.best_weights=tr.best_weights

        self.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

    def on_epoch_end(self, epoch, logs=None):            

        v_loss=logs.get('val_loss')

        v_acc=logs.get('val_accuracy')

        

        if v_loss<val.best_loss:

            msg='{0}\nfor epoch {1} validation loss improved,saving weights with validation loss= {2:7.4f}\n{3}'

            print(msg.format(Cgreen,epoch + 1, v_loss, Cend))

            val.best_loss=v_loss

            val.best_weights=model.get_weights()

        else:

            if tr.focus=='val':

                    #validation loss did not improve at end of current epoch

                    lr=float(tf.keras.backend.get_value(self.model.optimizer.lr))

                    new_lr=lr * .7

                    tf.keras.backend.set_value(model.optimizer.lr, new_lr)

                    msg='{0}\n for epoch {1} current loss {2:7.4f} exceeds best boss of {3:7.4f} reducing lr to {4:11.9f}{5}'

                    print(msg.format(Cyellow,epoch + 1, v_loss, val.best_loss, new_lr,Cend))

        val.lr=float(tf.keras.backend.get_value(model.optimizer.lr))

callbacks=[tr(), val()]
start_epoch=0

start=time.time()

results = model.fit_generator(generator = train_gen, validation_data= val_gen, epochs=epochs, initial_epoch=start_epoch,

                       callbacks = callbacks, verbose=1)

stop=time.time()

duration = stop-start

hrs=int(duration/3600)

mins=int((duration-hrs*3600)/60)

secs= duration-hrs*3600-mins*60

msg='{0}Training took\n {1} hours {2} minutes and {3:6.2f} seconds {4}'

print(msg.format(Cblu,hrs, mins,secs,Cend))

tacc=results.history['accuracy']

tloss=results.history['loss']

vacc=results.history['val_accuracy']

vloss=results.history['val_loss']

        
Epoch_count=len(tloss)

Epochs=[]

for i in range (0,Epoch_count):

    Epochs.append(i+1)

index_loss=np.argmin(vloss)#  this is the epoch with the lowest validation loss

val_lowest=vloss[index_loss]

index_acc=np.argmax(vacc)

val_highest=vacc[index_acc]

plt.style.use('fivethirtyeight')

sc_label='best epoch= '+ str(index_loss+1)

vc_label='best epoch= '+ str(index_acc + 1)

fig,axes=plt.subplots(nrows=1, ncols=2, figsize=(15,5))

axes[0].plot(Epochs,tloss, 'r', label='Training loss')

axes[0].plot(Epochs,vloss,'g',label='Validation loss' )

axes[0].scatter(index_loss+1,val_lowest, s=150, c= 'blue', label=sc_label)

axes[0].set_title('Training and Validation Loss')

axes[0].set_xlabel('Epochs')

axes[0].set_ylabel('Loss')

axes[0].legend()

axes[1].plot (Epochs,tacc,'r',label= 'Training Accuracy')

axes[1].plot (Epochs,vacc,'g',label= 'Validation Accuracy')

axes[1].scatter(index_acc+1,val_highest, s=150, c= 'blue', label=vc_label)

axes[1].set_title('Training and Validation Accuracy')

axes[1].set_xlabel('Epochs')

axes[1].set_ylabel('Accuracy')

axes[1].legend()

plt.tight_layout

#plt.style.use('fivethirtyeight')

plt.show()

lr_rate=val.lr 

weights=val.best_weights #( weights with lowest validation loss)

config = model.get_config()

pmodel = Model.from_config(config)  # copy of the model

pmodel.set_weights(weights) #load saved weights with lowest validation loss

pmodel.compile(Adam(lr=lr_rate), loss='categorical_crossentropy', metrics=['accuracy'])    

print('Training has completed. Now loading test set to see how accurate the model is')

results=pmodel.evaluate(test_gen, verbose=0)

print('{0}Model accuracy on Test Set is {1:7.2f} % {2}'.format(Cblu,results[1]* 100, Cend))

    