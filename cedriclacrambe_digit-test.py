# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
import functools
import keras
from keras import backend as K
import PIL
import IPython
import math
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Input
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
import pandas as pd
from keras.losses import mse, binary_crossentropy,sparse_categorical_crossentropy
from keras.utils import plot_model
import  logging
logging.basicConfig(filename='out.log',level=logging.INFO)
logger=logging.getLogger()
import math


import  datetime
date_depart=datetime.datetime.now()
print(os.listdir("../input"))
duree_max=datetime.timedelta(hours=5,minutes=45)
date_limite= date_depart+duree_max
epochs=int(1e8)
val_split=0.1
load_weights=False
frac_gen_epoch=4
# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("../input/digit-recognizer/train.csv",dtype="int16")
train_df=train_df.sample(frac=1)
val_size=int(np.ceil( len(train_df)*val_split) )
Xtrain=train_df.values[:-val_size,1:]
Xtrain=Xtrain.reshape(-1,28,28,1)
Ytrain=train_df.label.values[:-val_size]
train_df.info()
train_df.head()

n_fig=10

sample=np.random.choice(Xtrain.shape[0],n_fig)


fig=plt.figure(figsize=(15,15), dpi=100)
lignes=1
cols=n_fig//lignes
if cols==0:
    cols=n_fig
elif n_fig%lignes!=0:
    cols+=1
for i,idx in enumerate(sample):
    
    
    plt.subplot(lignes,cols,i+1)
    plt.imshow(Xtrain[idx,...,0],cmap="Greys", interpolation="nearest",vmin=-5, vmax=260)
    plt.title(Ytrain[idx])

    
plt.show()
Xval=train_df.values[-val_size:,1:]
Xval=Xval.reshape(-1,28,28,1)
Yval=train_df.label.values[-val_size:]

test_df=pd.read_csv("../input/digit-recognizer/test.csv",dtype="int16")
Xtest=test_df.values
Xtest=Xtest.reshape(-1,28,28,1)
test_df.info()
test_df.head()

class termination_date(keras.callbacks.Callback ):
    def __init__(self,end_date):
        self.end_date=end_date
    def on_epoch_end(self, batch, logs=None):
        if datetime.datetime.now()>self.end_date:
            self.model.stop_training = True
            logging.info("end date")


def covariance(x):
    n=K.get_variable_shape(x)[0]
    xm=x-K.mean(x,axis=0)
    K.get_variable_shape(x)
    return K.dot(K.transpose(xm),xm)/n
def corellation(x):
    return covariance(x)/K.var(x,axis=0)

def corellation_mid(x):
    n=K.get_variable_shape(x)[1]    
    return corellation(x)-np.identity(n)

def corellation_regul(*args,**kvargs):    
    return lambda x: keras.regularizers.l1_l2(*args,**kvargs)(corellation_mid(x))
def regul_combi(l1corr=0.01,l2corr=0.01,*args,**kvargs):
    reg=keras.regularizers.l1_l2(*args,**kvargs)
    return lambda x: reg(x)+corellation_regul(l1corr,l2corr)(x)

    
    

basel1=1e-5
basel2=2e-3
reguls=keras.regularizers.l1_l2(basel1,2e-3)
regulsfc=keras.regularizers.l1_l2(basel1,2e-3)
regulsfc1=regul_combi(1e-3,2e-1,basel1,basel2*20)
inp=Input(shape=(28,28,1), name="image", dtype="float32")

inp1=keras.layers.ZeroPadding2D(padding=2)(inp)
inp2=keras.layers.GaussianNoise(10)(inp1)

inp3=keras.layers.Lambda( lambda x:x/256
                        )(inp2)
c1pre=Conv2D(12,3,
          strides=1,
          padding='same',
          kernel_regularizer=None, 
          activation="selu",
          name="c1pre"
         )(inp3)

c1pre1=keras.layers.concatenate([c1pre,inp3])
c1pre1=keras.layers.SpatialDropout2D(0.2)(c1pre1)
c1=Conv2D(80,7,
          strides=1,
          padding='same',
          kernel_regularizer=None, 
          activation="selu",
          name="c1"
         )(c1pre1)

c2=Conv2D(100,1,
          strides=1,
          kernel_regularizer=reguls,            
          activation="selu",
           padding='same',
          name="c2"
         )(c1)
c2c=keras.layers.concatenate([c2,c1pre1])
c2c=keras.layers.BatchNormalization()(c2c)
c2c=keras.layers.SpatialDropout2D(0.32)(c2c)

c3=Conv2D(80,5,
          strides=2,
          kernel_regularizer=reguls, 
          activation="selu",
           padding='same',
          name="c3"
         )(c2c)
c3a=Conv2D(60,1,
          strides=1,
          kernel_regularizer=reguls, 
          activation="selu",
           padding='same',
          name="c3a"
         )(c3)
c4=Conv2D(80,5,
          strides=2,
          kernel_regularizer=reguls, 
          activation="selu",
           padding='same',
          name="c4",
          
         )(c3a)
c4a=Conv2D(80,1,
          strides=1,
          kernel_regularizer=reguls, 
          activation="selu",
           padding='same',
          name="c4a",
          
         )(c4)
c4a1=keras.layers.concatenate([c4a,c4])
c4a2=keras.layers.SpatialDropout2D(0.3)(c4a1)
c5=Conv2D(200,5,
          strides=2,
          kernel_regularizer=reguls, 
          activation="selu",
           padding='same',
          name="c5",
          
         )(c4a2)


pool_c2=keras.layers.GlobalMaxPooling2D(name="pooling_c2")(c2)
pool=keras.layers.GlobalMaxPooling2D(name="pooling_c5")(c5)
pool_c=keras.layers.concatenate([pool_c2,pool])
pool_c=keras.layers.BatchNormalization()(pool_c)

pool_c=keras.layers.Dropout(0.3)(pool_c)

hidd=Dense(200,
        activation="selu", 
        kernel_regularizer=regulsfc,
        name="hidd"
       )(pool_c)

o=Dense(10,
        activation="softmax", 
        kernel_regularizer=regulsfc1,
        name="softmax"
       )(hidd)
model=Model(inputs=inp,outputs=o)

optimizer=keras.optimizers.Adam(clipnorm=1. , clipvalue=1.)
top2acc=functools.partial(keras.metrics.sparse_top_k_categorical_accuracy,k=2)
top2acc.__name__="sparse_top_2_categorical_accuracy"


model.compile(optimizer,
              loss="sparse_categorical_crossentropy",
              metrics=["sparse_categorical_accuracy","sparse_categorical_crossentropy",top2acc]
             )


if load_weights:
    model.load_weights("../input/digit-test/digits.h5", by_name=True, skip_mismatch=True, reshape=False)
    if os.path.exists("digits.h5"):
        model.load_weights("digits.h5", by_name=True, skip_mismatch=True, reshape=False)
model.summary()


plot_model(model,show_shapes=True, show_layer_names=True)
IPython.display.Image(filename='model.png')
fichier_modele="digits.h5"
batch_size=512

val_batch_size=batch_size


steps_per_epoch=Xtrain.shape[0]/batch_size*2.5/frac_gen_epoch
validation_steps=Xval.shape[0]/val_batch_size
steps_per_epoch=int(steps_per_epoch)
validation_steps=int(validation_steps)+2

def noise_degrade(im, scale=40):
    im+=np.random.normal(loc=0.0, scale=scale, size=im.shape)
    im+=np.random.uniform(-scale*2,scale*2, size=im.shape)
    return im

noise_degrade_r=functools.partial(noise_degrade,scale=5)
img_gen_base=keras.preprocessing.image.ImageDataGenerator()
img_gen_aug=keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.95,1.2],
                                                     rotation_range=10,
                                                    brightness_range=[0.8,1.2],
                                                         rescale=0,
                                                     shear_range=0.02,
                                                     width_shift_range=0.02,
                                                     height_shift_range=0.02,
                                                            preprocessing_function=noise_degrade,
                                                     validation_split=0.1)

img_gen_aug_fine1=keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.99,1.05],
                                                     rotation_range=1,
                                                    brightness_range=[0.95,1.05],
                                                         rescale=0,
                                                     shear_range=0.005,
                                                     width_shift_range=0.005,
                                                     height_shift_range=0.005,
                                                              preprocessing_function=noise_degrade_r,
                                                     validation_split=0.1)
img_gen_aug_fine=keras.preprocessing.image.ImageDataGenerator(zoom_range=[0.99,1.05],
                                                     rotation_range=1,
                                                    brightness_range=[0.95,1.05],
                                                         rescale=0,
                                                     shear_range=0.005,
                                                     width_shift_range=0.005,
                                                     height_shift_range=0.005,
                                                              
                                                     validation_split=0.1)





callbacks=[
        keras.callbacks.ReduceLROnPlateau(monitor='val_sparse_categorical_accuracy',
                                          patience=10,
                                          min_delta=0.0005,
                                          #factor=0.2,
                                          min_lr=1e-6,
                                          verbose=1,
                                          cooldown=20

                                          ),
        keras.callbacks.ModelCheckpoint(monitor='val_sparse_categorical_accuracy',
                                        filepath=fichier_modele,
                                        verbose=1,
                                        save_best_only=True,
                                        period=20),
        keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                                      patience=100,
                                         
                                          verbose=1,
                                          restore_best_weights=True

                                          ),
         keras.callbacks.CSVLogger("train.csv", separator=',', append=True),
         termination_date(date_limite)

        ]
def resc_gen(genbase):
    for X,Y in genbase:
      
        yield (X-128,Y)


train_gen=img_gen_aug.flow(Xtrain, Ytrain, batch_size=batch_size, shuffle=True)
train_gen_fine=img_gen_aug_fine.flow(Xtrain, Ytrain, batch_size=batch_size, shuffle=True)
train_gen_fine1=img_gen_aug_fine.flow(Xtrain, Ytrain, batch_size=batch_size, shuffle=True)
val_gen=img_gen_base.flow(Xval, Yval, batch_size=val_batch_size, shuffle=True)

#train_gen=resc_gen(train_gen)
#train_gen_fine=resc_gen(train_gen_fine)
#train_gen_fine1=resc_gen(train_gen_fine1)


#val_gen=resc_gen(val_gen)


fig=plt.figure(figsize=(15,7), dpi=100)
ti=next(train_gen)[0]
n_fig=12
for i in range(n_fig):
    plt.subplot(2,n_fig,i+1)
    plt.imshow(ti[i,...,0],cmap="Greys", interpolation="nearest",vmin=-5, vmax=260)
    plt.subplot(2,n_fig,i+n_fig+1)
    plt.imshow(Xtrain[i,...,0],cmap="Greys", interpolation="nearest",vmin=-5, vmax=260)
    
plt.show()


hist_pre=model.fit(Xtrain, Ytrain,
          epochs=epochs,
          verbose=2,
          batch_size=batch_size,
          callbacks=callbacks+[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                          patience=20,
                              
                              verbose=1,
                              restore_best_weights=True

                              )],
          validation_data=(Xval,Yval),
)




hist=model.fit_generator(train_gen,
          epochs=epochs,
          initial_epoch=hist_pre.epoch[-1]+1,
           steps_per_epoch= steps_per_epoch,
          callbacks=callbacks,
          verbose=2,
        validation_data=(Xval,Yval),
        #validation_steps=validation_steps
                        )


hist_fine1=model.fit_generator(train_gen_fine1,
          epochs=epochs,
          initial_epoch=hist.epoch[-1]+1,
           steps_per_epoch= steps_per_epoch,
          callbacks=callbacks,
          verbose=2,
        validation_data=(Xval,Yval),
        
                        )

hist_fine=model.fit_generator(train_gen_fine,
          epochs=epochs,
          initial_epoch=hist_fine1.epoch[-1]+1,
           steps_per_epoch= steps_per_epoch,
          callbacks=callbacks,
          verbose=2,
        validation_data=(Xval,Yval),
        #validation_steps=validation_steps
                        )



hist_post=model.fit(Xtrain, Ytrain,
          epochs=epochs,
          verbose=2,
          batch_size=batch_size,
          initial_epoch=hist_fine.epoch[-1]+1,
           callbacks=callbacks+[keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy',
                          patience=20,
                              
                              verbose=1,
                              restore_best_weights=True

                              )],
          validation_data=(Xval,Yval)
)


model.save(fichier_modele)


for m,e in zip(model.metrics_names,
               model.evaluate(Xval, Yval,batch_size=batch_size) ):
               print (m,e)

fig=plt.figure(figsize=(15,7), dpi=100)
plt.subplot("211")
train_history=hist.history
for k in train_history.keys():
    train_history[k]=hist_pre.history[k]+ train_history[k]+hist_fine.history[k]+hist_post.history[k]
    
    
for k in train_history.keys():
    if "acc" in k:
        plt.plot(train_history[k],label=k)
plt.ylim(0.7,1)
plt.legend()
plt.subplot("212")
plt.yscale("log")
for k in train_history.keys():
    if "loss" in k:
        plt.plot(train_history[k],label=k)
plt.legend()

plt.ylim(top=0.8)
fig.savefig("graph.png",dpi=200,transparent=False)
#IPython.display.Image(filename="graph.png")
preds=model.predict(Xtest,batch_size=256, verbose=1,)
baseindex=pd.RangeIndex(start=1, stop=preds.shape[0]+1, step=1, name='ImageId')
sub=pd.DataFrame( data=preds.argmax(axis=-1),
                 index=baseindex, 
                 columns=["Label"], dtype="int32", copy=True)



n_fig=8*4
sample=sub.sample(n=n_fig)
sample.Label

fig=plt.figure(figsize=(15,15), dpi=100)
ti=next(train_gen)[0]
lignes=4
cols=n_fig//lignes
if n_fig%lignes!=0:
    cols+=1
for i in range(n_fig):
    label=sample.Label.iloc[i]
    idx=sample.index.values[i]
    plt.subplot(lignes,cols,i+1)
    plt.imshow(Xtest[idx,...,0],cmap="Greys", interpolation="nearest",vmin=-127, vmax=128)
    plt.title(label)

    
plt.show()
sub.to_csv("submission.csv",index_label="ImageId")
sub.info()
sub
