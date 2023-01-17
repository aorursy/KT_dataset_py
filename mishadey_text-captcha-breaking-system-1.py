!pip install captcha
# import module we'll need to import our custom module

from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/data4/CNN_Model.py", dst = "../working/CNN_Model.py")

copyfile(src = "../input/data3/Captcha_Sequence_Generator.py", dst = "../working/Captcha_Sequence_Generator.py")

copyfile(src = "../input/data3/Utils_funX.py", dst = "../working/Utils_funX.py")

import string

import pandas as pd

from keras import *

import numpy as np

import tensorflow as tf 

import tensorflow.compat.v1.keras.backend as Keras_backend

from keras.utils import Sequence

from captcha.image import ImageCaptcha

#configure the session to prevent tensorflow from occupying all the video memmory

config = tf.compat.v1.ConfigProto() 

config.gpu_options.allow_growth=True

session= tf.compat.v1.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)
from Captcha_Sequence_Generator import Captcha_Sequence_Generator

characters = string.digits + string.ascii_uppercase + string.ascii_lowercase

#Demostration

Data = Captcha_Sequence_Generator(characters,batch_size=1,steps=1)
X_val,Y_val=Data[0]
print(X_val.shape)

X_val
Y_val
from Utils_funX import Decode_Y_Val

y_decoded,char_classes=Decode_Y_Val(Y_val,characters)

char_classes,y_decoded
import matplotlib.pyplot as plt

plt.style.use('dark_background')

plt.imshow(X_val[0])

plt.title(y_decoded)
from CNN_Model import *
height=64 ;

width =128 ;

n_classes = len(characters) # 26+26+10 = 52 characters

n_len = 5;
model=CNN_Model_Initialize(height,width,n_classes,n_len)
model.summary()
CNN_model_visualize(model)
Train_data = Captcha_Sequence_Generator(characters,batch_size=128,steps=1000)

Test_data = Captcha_Sequence_Generator(characters,batch_size=128,steps=100)
history=CNN_model_Compile_and_Train(model,Train_data,Test_data,train_num=1)
#train the model again after loading the weights from the saved model --> with reduced Learning Rate for better Convergence --> increase the over all Accuracy of Our Model

model.load_weights("CNN_Model.h5")

history=CNN_model_Compile_and_Train(model,Train_data,Test_data,train_num=2)
model.save('CNN_Model.h5', include_optimizer=False)
df = pd.read_csv('cnn.csv')

df[['loss', 'val_loss']].plot()
df[['loss', 'val_loss']].plot(logy=True)
model=models.load_model("CNN_Model.h5")
y_val,y_pred=CNN_Model_Test(model,Test_data,characters)
y_val=np.array(y_val)

y_pred=np.array(y_pred)
y_val.shape,y_pred.shape
y_val[:10,:]
y_pred[:10,:]
from Utils_funX import evaluate_metrics
model.metrics_names
Model_metrics=pd.read_csv("CNN_Model_Epochs.csv")
Model_metrics
print("The overall Accuracy is : ",evaluate_metrics(model,characters))