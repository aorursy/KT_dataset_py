import keras
from keras.models import Model;
from keras.regularizers import  l2,l1,l1_l2;
from keras.layers import Input,Dense,merge,Dropout,BatchNormalization,\
    Activation,Conv2D,MaxPooling1D,MaxPooling2D,AveragePooling2D,Reshape,Flatten,UpSampling2D,Conv2DTranspose;
from keras.layers.advanced_activations import PReLU,LeakyReLU;
import time;import os;import pickle;import random;
from keras.models import Sequential,load_model;import numpy as np;import json;
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau;
from keras.utils.vis_utils import plot_model;from tqdm import tqdm;
from PIL import Image;from keras.activations import elu;
from keras.datasets import mnist,cifar10;
from keras import optimizers
DIR=os.getcwd();
reg_l2=l2(l=1e-7);
def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * elu(x, alpha)
def generator(noise_input):
    g=Dense(192,name="generator_1")(noise_input);
    #g=BatchNormalization(name="generator_2")(g);
    #g=Activation(selu,name="generator_3")(g);
    g=Reshape((8,8,3),name="generator_2")(g);
    g=Conv2DTranspose(filters=128,kernel_size=8,padding="same",name="generator_3")(g);
    #g=BatchNormalization(name="generator_4")(g)
    g=Activation(selu,name="generator_5")(g);
    g=UpSampling2D(size=(2,2),name="generator_11")(g);
    g=Conv2DTranspose(filters=128,kernel_size=5,padding="same",name="generator_12")(g);
    #g=BatchNormalization(name="generator_13")(g);
    g=Activation(selu,name="generator_14")(g);
    g = UpSampling2D(size=(2, 2), name="generator_15")(g);
    g = Conv2DTranspose(filters=64, kernel_size=5, padding="same", name="generator_16")(g);
    # g=BatchNormalization(name="generator_13")(g);
    g = Activation(selu, name="generator_18")(g);
    g=Conv2DTranspose(filters=3,kernel_size=5,padding="same",name="generator_23")(g);
    g=Activation("tanh",name="generator_24")(g);
    return  Model(inputs=noise_input,outputs=g);
def discriminator(d_input):
    d=Conv2D(filters=32,kernel_size=5,padding="same",name="discriminator_1")(d_input);
    d=LeakyReLU(alpha=0.2,name="discriminator_2")(d);
    d=AveragePooling2D(strides=(2,2),name="discriminator_3")(d);
    d=Conv2D(filters=64,kernel_size=5,padding="same",name="discriminator_5")(d);
    d=LeakyReLU(alpha=0.2,name="discriminator_6")(d);
    d=AveragePooling2D(strides=(2,2),name="discriminator_7")(d);
    d = Conv2D(filters=128, kernel_size=5, padding="same", name="discriminator_8")(d);
    d = LeakyReLU(alpha=0.2, name="discriminator_9")(d);
    d = AveragePooling2D(strides=(2, 2), name="discriminator_10")(d);
    d=Flatten(name="discriminator_15")(d)
    d=Dense(256,name="discriminator_16")(d)
    d=LeakyReLU(alpha=0.2,name="discriminator_17")(d);
    #d=Dropout(0.2,name="discriminator_16")(d);
    d=Dense(1,name="discriminator_19")(d);
    d=Activation("sigmoid",name="discriminator_20")(d);
    return Model(inputs=d_input,outputs=d);

def set_trainable(model, key_word, value=True):
    layers_list = [layer for layer in model.layers if key_word in layer.name]
    for layer in layers_list:
        layer.trainable = value
def g_d_together(g_model,d_model,noise_input):
    g_output=g_model(noise_input);
    output=d_model(g_output);
    return Model(inputs=noise_input,outputs=output);

noise_input=Input(shape=(100,));
g_model=generator(noise_input);
d_input=Input(shape=(32,32,3));
d_model=discriminator(d_input);
g_d_model=g_d_together(g_model,d_model,noise_input);
d_optim=keras.optimizers.Adam(learning_rate=0.001)
# d_optim=adam(lr=0.001);
g_optim=keras.optimizers.Adam(learning_rate=0.001)
# g_optim=adam(lr=0.0001);
d_model.compile(loss="binary_crossentropy",optimizer=d_optim);
g_d_model.compile(loss="binary_crossentropy",optimizer=g_optim);
# preparing dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],3));
X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],3))
import numpy as np
#g_model.load_weights(DIR+"/g_weights");
#d_model.load_weights(DIR+"/d_weights");
total_loss={};
batch_size=200;
for epoch in range(8000):
    total_loss["d_loss"] = [];
    total_loss["g_loss"] = [];
    all_image_labels=np.arange(0,len(X_train));
    np.random.shuffle(all_image_labels);
    batch_labels=np.array_split(all_image_labels,int(len(X_train)/batch_size));
    for j in tqdm(range(len(batch_labels))):
        batch_image_matrix = np.zeros((batch_size, 32, 32, 3), dtype=np.float32);
        for i,ele in enumerate(batch_labels[j]):
            batch_image_matrix[i]=(X_train[ele]-127.5)*127.5**-1;
        batch_noise=np.random.normal(size=(batch_size,100));
        generated_images = g_model.predict(batch_noise);
        X = np.concatenate((batch_image_matrix, generated_images));
        y = [0.9]*batch_size +\
            [0]*batch_size;
        set_trainable(d_model, "discriminator", True);
        d_loss = d_model.train_on_batch(X, y);
        total_loss["d_loss"].append(d_loss);
        set_trainable(d_model, "discriminator", False);
        g_loss = g_d_model.train_on_batch(batch_noise,
                                          [1]*batch_size)
        total_loss["g_loss"].append(g_loss);
    d_loss_average=np.mean(total_loss["d_loss"]);
    g_loss_average=np.mean(total_loss["g_loss"]);
    with open(DIR+"/log","a") as f:
        f.write(str(epoch)+"\t"+str(d_loss_average)+"\t"+str(g_loss_average)+"\n");
    g_model.save_weights(DIR+"/g_weights");
    d_model.save_weights(DIR+"/d_weights");
    if epoch%10==0:
        noise_sample = np.random.normal(size=(1, 100));
        generated = g_model.predict(noise_sample);
        generated = generated.reshape((32, 32, 3));
        generated = generated * 127.5 + 127.5;
        generated = np.asarray(generated, dtype=np.uint8);
        img = Image.fromarray(generated);
        img.save("images/generated"+str(epoch)+".png");
