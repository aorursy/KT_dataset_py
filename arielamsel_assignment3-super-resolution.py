!pip install opencv-python
!pip install efficientnet
from keras.models import load_model
image_dir = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages'
amount_glob = 100
max_amount = 6000
"set"
import cv2
import os

# def load_all_images(amount = max_amount):
#     orig_imgs.clear()
#     all_im = os.listdir(image_dir)
#     for idx,img_path in enumerate(all_im):
#         if (idx == amount):
#             break    
#         img = cv2.imread(image_dir+"/"+img_path)
#         orig_imgs.append(img)
#         if (idx % 100 == 0):
#             print (idx)
#     return orig_imgs


# load_all_images()
# "finished loading ", len (orig_imgs)

def load_images_iter(amount = 64,start_pos = 0, end_pos = 5011):
  ans = []
  all_im = os.listdir(image_dir)
  all_im.sort()
  print (len(all_im))
  while (True):
    for idx,img_path in enumerate(all_im[start_pos:end_pos]):
      if (len(ans) !=0 and len(ans)%amount == 0):
        ret = ans
        ans = []
        yield ret
      ans.append(cv2.imread(image_dir+"/"+img_path))
import numpy as np

def train_generator_1(batch=64, tar_size=144, train_size=72):
    collect_train = []
    collect_target = []
    
    while True:
        file_gen = load_images_iter(amount=batch, start_pos = 1000)#first 1000 images are for validation
        imgs = []
        while (True):
          try:
            imgs = next(file_gen)
          except:
            break
          for idx,img in enumerate(imgs): 
              if (len(collect_train)!=0 and len(collect_train)%batch == 0):
                  ans_train = np.asarray(collect_train,dtype=np.float)
                  ans_target = np.asarray(collect_target,dtype=np.float)
                  collect_train = []
                  collect_target = []
                  yield (ans_train, ans_target)
              collect_train.append(cv2.resize(img,(train_size,train_size))/255.0)
              collect_target.append(cv2.resize(img,(tar_size,tar_size))/255.0)

def train_generator_2(batch=64, tar1_size=144, tar2_size=288 , train_size=72):
    collect_train = []
    collect_target_1 = []
    collect_target_2 = []

    while True:
        file_gen = load_images_iter(amount=batch, start_pos = 1000)#first 1000 images are for validation
        imgs = []
        while (True):
          try:
            imgs = next(file_gen)
          except:
            break
          for idx,img in enumerate(imgs): 
              if (len(collect_train)!=0 and len(collect_train)%batch == 0):
                  ans_train = np.asarray(collect_train,dtype=np.float)
                  ans_target_1 = np.asarray(collect_target_1,dtype=np.float)
                  ans_target_2 = np.asarray(collect_target_2,dtype=np.float)
                  collect_train = []
                  collect_target_1 = []
                  collect_target_2 = []
                  yield (ans_train, (ans_target_1,ans_target_2))
              collect_train.append(cv2.resize(img,(train_size,train_size))/255.0)
              collect_target_1.append(cv2.resize(img,(tar1_size,tar1_size))/255.0)
              collect_target_2.append(cv2.resize(img,(tar2_size,tar2_size))/255.0)
            

def val_generator_1(batch=64, tar_size= 144, train_size=72):
    collect_train = []
    collect_target = []
    
    while True:
        file_gen = load_images_iter(amount=batch, end_pos = 1000)#first 1000 images are for validation
        imgs = []
        while (True):
          try:
            imgs = next(file_gen)
          except:
            break
          for idx,img in enumerate(imgs): 
              if (len(collect_train)!=0 and len(collect_train)%batch == 0):
                  ans_train = np.asarray(collect_train,dtype=np.float)
                  ans_target = np.asarray(collect_target,dtype=np.float)
                  collect_train = []
                  collect_target = []
                  yield (ans_train, ans_target)
              collect_train.append(cv2.resize(img,(train_size,train_size))/255.0)
              collect_target.append(cv2.resize(img,(tar_size,tar_size))/255.0)
            
def val_generator_2(batch=64, tar1_size=144, tar2_size=288 , train_size=72):
    collect_train = []
    collect_target_1 = []
    collect_target_2 = []

    while True:
        file_gen = load_images_iter(amount=batch, end_pos = 1000)#first 1000 images are for validation
        imgs = []
        while (True):
          try:
            imgs = next(file_gen)
          except:
            break
          for idx,img in enumerate(imgs): 
              if (len(collect_train)!=0 and len(collect_train)%batch == 0):
                  ans_train = np.asarray(collect_train,dtype=np.float)
                  ans_target_1 = np.asarray(collect_target_1,dtype=np.float)
                  ans_target_2 = np.asarray(collect_target_2,dtype=np.float)
                  collect_train = []
                  collect_target_1 = []
                  collect_target_2 = []
                  yield (ans_train, (ans_target_1,ans_target_2))
              collect_train.append(cv2.resize(img,(train_size,train_size))/255.0)
              collect_target_1.append(cv2.resize(img,(tar1_size,tar1_size))/255.0)
              collect_target_2.append(cv2.resize(img,(tar2_size,tar2_size))/255.0)


import matplotlib.pyplot as plt
%matplotlib inline

def plot_imgs(rows,cols,img_list,titles = None,fig_size=(20,9)):
  fig,ax = plt.subplots(rows,cols,figsize=fig_size)
  if (rows == 1):
    for j in range(cols):
      ax[j].imshow(img_list[j][0])
      if (titles):
         ax[j].set_title(titles[j])
  else:
    for i in range(rows):
      for j in range(cols):
        ax[i,j].imshow(img_list[j][i])
    if (titles):
      for idx,title in enumerate(titles):
        ax[0,idx].set_title(title)
  plt.show()


"generators created"
import matplotlib.pyplot as plt
%matplotlib inline

def plot_imgs(rows,cols,img_list,titles = None,fig_size=(20,9)):
  fig,ax = plt.subplots(rows,cols,figsize=fig_size)
  if (rows == 1):
    for j in range(cols):
      ax[j].imshow(img_list[j][0])
      if (titles):
         ax[j].set_title(titles[j])
  else:
    for i in range(rows):
      for j in range(cols):
        ax[i,j].imshow(img_list[j][i])
    if (titles):
      for idx,title in enumerate(titles):
        ax[0,idx].set_title(title)
  plt.show()


train_imgs = next(train_generator_2())
plot_imgs(3,3,[train_imgs[0],train_imgs[1][0],train_imgs[1][1]],fig_size=(20,9), titles = ["size = 72","size = 144","size = 288"])
val_imgs = next(val_generator_2())
plot_imgs(3,3,[val_imgs[0],val_imgs[1][0],val_imgs[1][1]],fig_size=(20,9), titles = ["size = 72","size = 144","size = 288"])
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras import backend as K
import math

import numpy as np

def set_callbacks(description,patience=5,tb_base_logdir='./logs/'):
#     cp = ModelCheckpoint(name +'.h5',save_best_only=True)
#     es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=patience,monitor='val_out1_PSNR')
    cb = [rlop]
    return cb


def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * (1.0 / math.log(10)) * K.log((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def plot_histories(history, model_name = "Model"):
    plt.plot(history.history['out1_loss'], label = "out1_loss")
    plt.plot(history.history['val_out1_loss'], label = "val_out1_loss")
    plt.plot(history.history['out2_loss'], label = "out2_loss")
    plt.plot(history.history['val_out2_loss'], label = "val_out2_loss")
    plt.title(model_name + ' Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.plot(history.history['out1_PSNR'], label = "out1_PSNR")
    plt.plot(history.history['out2_PSNR'], label = "out1_PSNR")
    plt.plot(history.history['val_out1_PSNR'], label = "val_out1_PSNR")
    plt.plot(history.history['val_out2_PSNR'], label = "val_out2_PSNR")
    
    plt.title(model_name + ' PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.show()

"set"
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input
from keras.utils import plot_model

inp = Input((72,72,3))
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)
x = UpSampling2D(size=(2,2))(x)
x = Conv2D(3,(1,1), activation = 'relu', padding = 'same')(x)

model1 = Model(inputs = inp ,outputs = x)
model1.summary()
plot_model(model1,show_shapes=True)

batch_size = 64
model1.compile(optimizer = 'adam', loss='mse',metrics=[PSNR])
history1 = model1.fit(train_generator_1(), epochs=12, batch_size=batch_size,steps_per_epoch= 4011//batch_size,
                     validation_data = val_generator_1(),validation_steps=1000//batch_size )

from keras.models import load_model
model1.save('model1.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plt.plot(history1.history['loss'], label = "loss")
plt.plot(history1.history['val_loss'], label = "val_loss")
plt.title('Model Loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()

plt.plot(history1.history['PSNR'], label = "PSNR")
plt.plot(history1.history['val_PSNR'], label = "PSNR")
plt.title('Model PSNR')
plt.ylabel('PSNR')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
gen = val_generator_1()
imgs = next(gen)
predictions = model1.predict(imgs[0][:10])
plot_imgs(5,3,[imgs[0][:10],imgs[1][:10], predictions], titles=["source-72","val-144","prediction-144"])

imgs = next(gen)
predictions = model1.predict(imgs[0][:10])
plot_imgs(5,3,[imgs[0][:10],imgs[1][:10], predictions], titles=["source-72","val-144","prediction-144"])
import keras 
model1 = None
keras.backend.clear_session()
"clear"
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input
from keras.utils import plot_model

inp = Input((None,None,3))
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)
up1 = UpSampling2D(size=(2,2))(x)
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = UpSampling2D(size=(2,2))(up1)
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)



model2 = Model(inputs = inp ,outputs=[out1,out2])
model2.summary()
plot_model(model2,show_shapes=True)

batch_size = 64
model2.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR])
history2 =model2.fit(train_generator_2(), epochs=12, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model2.save('model2.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history2, "Model 2")
# model2 = load_model("model2.h5")

gen = val_generator_2()
imgs = next(gen)

pred2 = model2.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred2[0],pred2[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred2 = model2.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred2[0],pred2[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
import keras 
model2 = None
keras.backend.clear_session()
"clear"
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU
from keras.utils import plot_model


def get_residual_block(in_layer, filters = 32):
  x = Conv2D(filters,(3,3), activation = LeakyReLU(0.2), padding = 'same')(in_layer)
  x = Conv2D(filters,(3,3), activation = LeakyReLU(0.2), padding = 'same')(x)
  # x = averagepooling or somethign
  x = Add()([in_layer,x])
  x = Activation(LeakyReLU(0.2))(x)
  return x

inp = Input((None,None,3))
x = Conv2D(32,(1,1),activation = LeakyReLU(0.2), padding = 'same')(inp)
res = get_residual_block(x)
res = get_residual_block(res)
up1 = UpSampling2D(size=(2,2))(res)
out1 = Conv2D(3,(1,1), activation = LeakyReLU(0.2), padding = 'same', name="out1")(up1)
res2 = get_residual_block(up1)
up2 = UpSampling2D(size=(2,2))(res2)
out2 = Conv2D(3,(1,1), activation = LeakyReLU(0.2), padding = 'same', name="out2")(up2)



model3 = Model(inputs = inp ,outputs=[out1,out2])
model3.summary()
plot_model(model3,show_shapes=True)

batch_size = 64
model3.compile(optimizer = 'adam', loss =["mse","mse"], metrics = [PSNR], loss_weights=[0.3,0.7])

history3 = model3.fit(train_generator_2(), epochs=12, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model3.save('model3.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history3, "Model 3")
gen = val_generator_2()
imgs = next(gen)
pred3 = model3.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred3[0],pred3[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)
pred3 = model3.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred3[0],pred3[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
import keras 
model3 = None
keras.backend.clear_session()
"clear"
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate
from keras.utils import plot_model


def get_atrous_block(in_layer, filters = 32):
  x = Conv2D(filters,(3,3), activation = LeakyReLU(0.2), padding = 'same')(in_layer)
  x = Conv2D(filters,(3,3),dilation_rate=(2,2), activation = LeakyReLU(0.2), padding = 'same')(x)
  x = Conv2D(filters,(3,3),dilation_rate=(4,4), activation = LeakyReLU(0.2), padding = 'same')(x)
  # x = averagepooling or somethign
  x = Concatenate()([in_layer,x])
  x = Activation(LeakyReLU(0.2))(x)
  x = Conv2D(filters,(3,3), activation = LeakyReLU(0.2), padding = 'same')(x)
  return x

inp = Input((72,72,3))
x = Conv2D(32,(1,1),activation = LeakyReLU(0.2), padding = 'same')(inp)
res = get_atrous_block(x)
res = get_atrous_block(res)
up1 = UpSampling2D(size=(2,2))(res)
out1 = Conv2D(3,(1,1), activation = LeakyReLU(0.2), padding = 'same', name="out1")(up1)
res2 = get_atrous_block(up1)
up2 = UpSampling2D(size=(2,2))(res2)
out2 = Conv2D(3,(1,1), activation = LeakyReLU(0.2), padding = 'same', name="out2")(up2)



model4 = Model(inputs = inp ,outputs=[out1,out2])
model4.summary()
plot_model(model4,show_shapes=True)

batch_size = 64
model4.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR], loss_weights=[0.3,0.7])

history4 = model4.fit(train_generator_2(), epochs=12, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model4.save('model4.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history4,"Model 4")

gen = val_generator_2()
imgs = next(gen)

pred4 = model4.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred4[0],pred4[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred4 = model4.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred4[0],pred4[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
import keras 
model4 = None
keras.backend.clear_session()
"clear"
import efficientnet.keras as efn 


efn = efn.EfficientNetB4(include_top = False, input_shape = (None,None,3))

efn.trainable = False
for layer in efn.layers:
  layer.trainable = False

efn.summary()
from keras.models import Sequential, Model

my_efn = Model(inputs = efn.input, outputs = efn.get_layer("block1a_activation").output)
my_efn.trainable = False
for layer in efn.layers:
  my_efn.trainable = False

my_efn.summary()
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate
from keras.utils import plot_model


inp = Input((None,None,3))
ext = my_efn(inp)
ext = UpSampling2D(size=(2,2))(ext)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)
x = Concatenate()([x,ext])
x = Conv2D(112,(1,1), activation = 'relu', padding = 'same')(x)
up1 = UpSampling2D(size=(2,2))(x)
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = UpSampling2D(size=(2,2))(out1)
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)



model5 = Model(inputs = inp ,outputs=[out1,out2])
model5.summary()
plot_model(model5,show_shapes=True)

batch_size = 64
model5.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR], loss_weights=[0.3,0.7])
history5 = model5.fit(train_generator_2(), epochs=20, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model5.save('model5.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history5, "Model 5")

gen = val_generator_2()
imgs = next(gen)

pred5 = model5.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5[0],pred5[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred5 = model5.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5[0],pred5[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
import keras 
model5 = None
keras.backend.clear_session()
"clear"
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate
from keras.utils import plot_model


inp = Input((None,None,3))
ext = my_efn(inp)
ext = UpSampling2D(size=(2,2))(ext)
ext = Conv2D(48,(1,1),activation = 'relu', padding = 'same')(ext) ## Added convolutional layer

x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)

x = Concatenate()([x,ext])
x = Conv2D(112,(1,1), activation = 'relu', padding = 'same')(x)
up1 = UpSampling2D(size=(2,2))(x)
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = UpSampling2D(size=(2,2))(out1)
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)



model5_2 = Model(inputs = inp ,outputs=[out1,out2])
model5_2.summary()
plot_model(model5_2,show_shapes=True)

batch_size = 64
model5_2.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR])
history5_2 = model5_2.fit(train_generator_2(), epochs=12, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model5_2.save('model5_2.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history5_2, "Model 5_2")

gen = val_generator_2()
imgs = next(gen)

pred5_2 = model5_2.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_2[0],pred5_2[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred5_2 = model5_2.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_2[0],pred5_2[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate
from keras.utils import plot_model


inp = Input((None,None,3))
ext = my_efn(inp)
ext = UpSampling2D(size=(2,2))(ext)
ext = Conv2D(96,(1,1),activation = 'relu', padding = 'same')(ext) ## Doubled Number of Filters

x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)

x = Concatenate()([x,ext])
x = Conv2D(80,(1,1), activation = 'relu', padding = 'same')(x) 
up1 = UpSampling2D(size=(2,2))(x)
up1 = Conv2D(80,(1,1), activation = 'relu', padding = 'same')(up1) ## Added convolutinal layer
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = UpSampling2D(size=(2,2))(out1)
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)



model5_3 = Model(inputs = inp ,outputs=[out1,out2])
model5_3.summary()
plot_model(model5_3,show_shapes=True)

batch_size = 64
model5_3.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR], loss_weights= [0.65,0.35])
history5_3 = model5_3.fit(train_generator_2(), epochs=25, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model5_3.save('model5_3.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history5_3, "Model 5_3")

gen = val_generator_2()
imgs = next(gen)

pred5_3 = model5_3.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_3[0],pred5_3[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred5_3 = model5_3.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_3[0],pred5_3[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM,Dropout, BatchNormalization, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate
from keras.utils import plot_model


inp = Input((None,None,3))
ext = my_efn(inp)
ext = UpSampling2D(size=(2,2))(ext)
ext = Conv2D(96,(1,1),activation = 'relu', padding = 'same')(ext)

x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)

x = Concatenate()([x,ext])
x = Conv2D(80,(1,1), activation = 'relu', padding = 'same')(x) 
up1 = UpSampling2D(size=(2,2))(x)
up1 = Conv2D(80,(1,1), activation = 'relu', padding = 'same')(up1)
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = Conv2D(32,(1,1), activation = 'relu', padding = 'same')(out1) ## Added convolutional layer
up2 = UpSampling2D(size=(2,2))(out1)
up2 = Conv2D(64,(1,1), activation = 'relu', padding = 'same')(up2) ## Added convolutional layer
up2 = Conv2D(64,(1,1),dilation_rate = (2,2), activation = 'relu', padding = 'same')(up2) ## Added convolutional layer
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)



model5_4 = Model(inputs = inp ,outputs=[out1,out2])
model5_4.summary()
plot_model(model5_4,show_shapes=True)

batch_size = 64
model5_4.compile(optimizer = 'adam', loss =['mse','mse'], metrics =[PSNR], loss_weights= [0.65,0.35])
history5_4 = model5_4.fit(train_generator_2(), epochs=30, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model5_4.save('model5_4.h5')

import matplotlib.pyplot as plt
%matplotlib inline


plot_histories(history5_4)
gen = val_generator_2()
imgs = next(gen)

pred5_4 = model5_4.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_4[0],pred5_4[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred5_4 = model5_4.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred5_4[0],pred5_4[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
# !pip install efficientnet

import efficientnet.keras as efn 


efn = efn.EfficientNetB4(include_top = False, input_shape = (None,None,3))

efn.trainable = False
for layer in efn.layers:
  layer.trainable = False

efn.summary()
from keras.models import Sequential, Model
my_efn = Model(inputs = efn.input, outputs = efn.get_layer("block1a_activation").output)
my_efn.trainable = False
for layer in efn.layers:
  my_efn.trainable = False

my_efn.summary()
from keras.models import Sequential, Model
from keras.layers import Dense, UpSampling2D, Conv2D, Input, Add, Activation, LeakyReLU, Concatenate, Lambda
from keras.utils import plot_model
from tensorflow.nn import depth_to_space


inp = Input((None,None,3))

ext = my_efn(inp)
ext = UpSampling2D(size=(2,2))(ext)
ext = Conv2D(48,(1,1),activation = 'relu', padding = 'same')(ext)

x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(inp)
x = Conv2D(64,(3,3), activation = 'relu', padding = 'same')(x)

conc = Concatenate()([x,ext])
conc = Conv2D(112,(1,1), activation = 'relu', padding = 'same')(conc)
up1 =  Lambda(lambda x:depth_to_space(x,2))(conc)
up1 = Conv2D(64,(1,1), activation = 'relu', padding = 'same')(up1)
out1 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out1")(up1)
up2 = Lambda(lambda x:depth_to_space(x,2))(Conv2D(64,(1,1), activation = 'relu', padding = 'same')(out1))
out2 = Conv2D(3,(1,1), activation = 'relu', padding = 'same', name="out2")(up2)


model6 = Model(inputs=inp,outputs=[out1, out2])
model6.summary()
plot_model(model6,show_shapes=True)
batch_size=64
model6.compile(optimizer = 'adam',loss =['mse','mse'], metrics =[PSNR],loss_weights= [0.7,0.3])
history6 = model6.fit(train_generator_2(), epochs=20, batch_size=batch_size, steps_per_epoch= (4011//batch_size),
                      validation_data = val_generator_2(), validation_steps= (1000//batch_size) )


from keras.models import load_model
model6.save('model6.h5')

import matplotlib.pyplot as plt
%matplotlib inline

plot_histories(history6,"Model 6")

gen = val_generator_2()
imgs = next(gen)

pred6 = model6.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred6[0],pred6[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])

imgs = next(gen)

pred6 = model6.predict(imgs[0][:10])
plot_imgs(5,5,[imgs[0],imgs[1][0],imgs[1][1],pred6[0],pred6[1]] ,
          titles=["source-72","val-144","val-288","prediction-144","prediction-288"])
