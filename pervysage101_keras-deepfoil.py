# !pip install gdown

# !gdown https://drive.google.com/uc?id=1YPQifK-q3-ix3bA_yrN8Q9g7SqL8WI1V
!ls "../input"
import tensorflow as tf

import keras

# import tensorflow.keras as keras

from keras import *

from keras.layers import *

from keras.models import *

import keras.backend as K

import numpy as np

import time

import pandas as pd

import os

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from keras.callbacks import *
def create_model(num_layers,num_width):

#     model = Sequential()

    

    ## inputs

    t = Input(shape=(1,))

    x = Input(shape=(1,))

    y = Input(shape=(1,))

    

    X = concatenate([t, x, y], 1)

#     X = 2.0*(tf.concat([t,x,y], 1) - X_min)/(X_max - X_min) - 1.0

#     t_tmp = 2.0*(t - self.X_min[0])/(self.X_max[0] - self.X_min[0]) - 1

    

    ## eta_model

    new_layer_eta = Dense(num_width,activation='relu')(t)

    for l in range(int((num_layers-2)/2)):

        new_layer_eta = BatchNormalization()(new_layer_eta)

        new_layer_eta = Dense(num_width,activation='relu')(new_layer_eta)

        new_layer_eta = Dense(num_width,activation='relu')(new_layer_eta)

#     new_layer = Dense(num_width,activation='relu')(first_layer)    

    out_eta = Dense(1,activation='tanh',name='out_eta')(new_layer_eta)

    

    ## uvp model

    first_layer_uvp = Dense(num_width*3,activation='relu')(X)

    new_layer_uvp = Dense(num_width*3,activation='relu')(first_layer_uvp)

    for l in range(int((num_layers-2)/2)-1):

        new_layer_uvp = BatchNormalization()(new_layer_uvp)

        new_layer_uvp = Dense(num_width*3,activation='relu')(new_layer_uvp)

        new_layer_uvp = Dense(num_width*3,activation='relu')(new_layer_uvp)

    

    out_uvp = Dense(3,activation='tanh',name='out_uvp')(new_layer_uvp)

    

    final_out = concatenate([out_uvp,out_eta])

    

#     model = Model(inputs=[t,x,y] , outputs=[out_eta ,out_uvp])

    

    model = Model(inputs=[t,x,y], outputs=final_out)

    

    return model
def custom_loss_wrapper(Re):

#     input_tensor = concatenate([t, x, y], 1)

    def gradient_calc(Re):

        

        uvp = model.output

        u = uvp[:,0:1]

        v = uvp[:,1:2]

        p = uvp[:,2:3]

        eta = uvp[:,3:4]

#         print(u)

        

        u_t,u_x,u_y = K.gradients(u,model.input)

        v_t,v_x,v_y = K.gradients(v,model.input)

        p_t,p_x,p_y = K.gradients(p,model.input)

        eta_t,eta_x,eta_y = K.gradients(eta,model.input)

        

        u_xx = K.gradients(u_x,model.input[1])[0]

        u_yy = K.gradients(u_y,model.input[2])[0]

        v_xx = K.gradients(v_x,model.input[1])[0]

        v_yy = K.gradients(v_y,model.input[2])[0]

        eta_tt = K.gradients(eta_t,model.input[0])[0]

        

#         print((u_xx)+(u_yy))

        

        eq1 = u_t + (u*u_x + v*u_y) + p_x - (1.0/Re)*(u_xx + u_yy)

        eq2 = v_t + (u*v_x + v*v_y) + p_y - (1.0/Re)*(v_xx + v_yy) + eta_tt

        eq3 = u_x + v_y

        

        loss = K.mean(tf.square(eq1)) + K.mean(tf.square(eq2)) + K.mean(tf.square(eq3))

        

#         print((u_xx))

        return loss



    def custom_loss(y_true, y_pred):

        navier_loss = gradient_calc(Re=Re)

#         navier_loss = net_VIV(input_tensor,y_pred,Re=1000)

        return tf.reduce_mean(tf.square(y_true - y_pred)) + navier_loss

    return custom_loss



# model.compile(loss={'out_eta': 'mean_squared_error', 'out_uvp': custom_loss_wrapper(model.layers[3])}, optimizer='adam', metrics=['accuracy','mean_squared_error'])

# model.compile(loss=custom_loss_wrapper(Re=1000), optimizer='adam', metrics=['mse'])
model = create_model(18,32)

model.compile(loss=custom_loss_wrapper(Re=1000), optimizer='adam', metrics=['mse'])

model.summary()
def load_data(filename = "../input/prevdata/final_translated_data_1_6.csv"):

    time_1 = time.time()

    data = pd.read_csv(filename)

    print("[INFO] Time taken = "+str(time.time()-time_1)+" seconds.")

    return data
def create_tensors(data):

    

    t_star = data['t'] # T x 1

    eta_star = data['eta'] # T x 1

    

    T = t_star.shape[0]

        

    X_star = data['x']

    Y_star = data['y']        

    U_star = data['u']

    V_star = data['v']

    P_star = data['p']



    t = t_star.to_numpy()

    eta = eta_star.to_numpy()

    

    # T = t_star.shape[0]

        

    x = X_star.to_numpy()

    y = Y_star.to_numpy()        

    u = U_star.to_numpy()

    v = V_star.to_numpy()

    p = P_star.to_numpy()

    

    t = t.reshape((t.shape[0],1))

    x = x.reshape((x.shape[0],1))

    y = y.reshape((y.shape[0],1))

    u = u.reshape((u.shape[0],1))

    v = v.reshape((v.shape[0],1))

    p = p.reshape((p.shape[0],1))

    eta = eta.reshape((eta.shape[0],1))

    

    return t,x,y,u,v,p,eta
def plot_time_data(time_snap):

  time_data = data[data['t']==time_snap]

  x_snap = time_data["x"]

  y_snap = time_data["y"]

  plt.plot(x_snap, y_snap, 'bo',markersize=0.1,marker='o',scalex=3,scaley=3)

  plt.title("at time: "+str(time_snap)+" second")

  plt.show()
data = load_data(filename="../input/prevdata/final_translated_data_1_6.csv")

t,x,y,u,v,p,eta = create_tensors(data)

t.shape,x.shape,y.shape,u.shape,v.shape,p.shape,eta.shape
## Scaling the input

scale_vals = np.concatenate([t,x,y,u,v,p,eta],1)

scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(scale_vals)

scale_vals = scaler.transform(scale_vals)

scale_vals.shape

scaled_list = np.split(scale_vals,7,axis=1)



t = scaled_list[0]

x = scaled_list[1]

y = scaled_list[2]

u = scaled_list[3]

v = scaled_list[4]

p = scaled_list[5]

eta = scaled_list[6]



t.shape,x.shape,y.shape,u.shape,v.shape,p.shape,eta.shape
N_train = 9650000 #3800000

idx = np.random.choice(t.shape[0], N_train, replace=False)

t_train = t[idx]

x_train = x[idx]

y_train = y[idx]

u_train = u[idx]

v_train = v[idx]

p_train = p[idx]

eta_train = eta[idx]

t_train.shape,x_train.shape,y_train.shape,u_train.shape,v_train.shape,p_train.shape,eta_train.shape
# for time in np.arange(start=1,step=0.5,stop=3.02):

#   plot_time_data(time)
# model.load_weights("../input/keras-deepfoil/model_230epoch.h5")
# Uncomment to train from scratch

# init_epoch = 230

n_epoch = 60

batch_size = 10000

log_dir = "logs/18x32_logs/"

tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoints=keras.callbacks.callbacks.ModelCheckpoint("model_1.h5", monitor='mse', verbose=0, save_best_only=True, save_weights_only=False, mode='min', period=1)

uvpEta_train = np.concatenate([u_train,v_train,p_train,eta_train],1)

h = model.fit([t_train,x_train,y_train],[uvpEta_train], epochs=n_epoch, batch_size=batch_size, shuffle=True, callbacks = [checkpoints])
model.save("18X32_100epoch.h5")
import keras

from matplotlib import pyplot as plt

history = h

plt.plot(history.history['mse'])

# plt.plot(history.history['loss'])

plt.title('model mse')

plt.ylabel('MSE')

plt.xlabel('epoch')

# plt.legend(['train', 'val'], loc='upper left')

plt.show()