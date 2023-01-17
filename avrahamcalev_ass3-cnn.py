import numpy as np
import imageio
import os
import cv2 
import matplotlib.pyplot as plt

def load_img(name):
    img = imageio.imread(name)
    return img

def load_images():
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)
    imgs = list()
    for i,p in enumerate(arr):
        img = load_img(img_directory+p)
        imgs.append(np.array(img))
    return np.array(imgs)

def resize_imgs(imgs, nrow=244,ncol=244):
    output = []
    for i, img in enumerate(imgs):
        tmp = cv2.resize(img,dsize=(nrow,ncol))
        output.append(tmp)
    return np.array(output)

def plot(img_a , img_b, img_c):
    fig = plt.figure(figsize=(20,20))
    a = fig.add_subplot(1, 3, 1)
    imgplot = plt.imshow(img_a)
    a.set_title('72*72')
    a = fig.add_subplot(1, 3, 2)
    imgplot = plt.imshow(img_b)
    a.set_title('144*144')
    a = fig.add_subplot(1, 3, 3)
    imgplot = plt.imshow(img_c)
    a.set_title('288*288')

def better_plot(cols, imgs, titles, figsize=(20,20)):
    fig = plt.figure(figsize=figsize)
    for j in range(cols):
        ax = fig.add_subplot(1, cols, j+1)
        ax.set_title(titles[j])
        plt.imshow(imgs[j])
            

imgs = load_images()
print(len(imgs),',',len(imgs[0]),',',len(imgs[0][0]) ,',',len(imgs[0][0][0]))
resize_a = resize_imgs(imgs, 72, 72)
resize_b = resize_imgs(imgs, 144, 144)
resize_c = resize_imgs(imgs, 288, 288)

titles = ['72*72','144*144','288*288']
lst = [resize_a[50],resize_b[50], resize_c[50]]
better_plot(len(lst), lst, titles)
def next_X(s,e):
    while True:
        for i, img in enumerate(resize_a[s:e]):
            img = img / 255
            yield img

def next_y_mid(s,e):
    while True:
        for i, img in enumerate(resize_b[s:e]):
            img = img / 255
            yield img

        
def next_y_large(s,e):
    while True:
        for i, img in enumerate(resize_c[s:e]):
            img = img / 255
            yield img

def next_X_y_mid(s,e):
    while True:
        for i, (img_x, img_y) in enumerate(zip(resize_a[s:e],resize_b[s:e])):
            img_x = img_x / 255
            img_y = img_y / 255
            yield np.expand_dims(img_x, axis=0), np.expand_dims(img_y, axis=0) 
        
def next_X_y_mid_large(s,e):
    while True:
        for i, (img_x, img_y_mid, img_y_large) in enumerate(zip(resize_a[s:e],resize_b[s:e],resize_c[s:e])):
            img_x = img_x / 255
            img_y_mid = img_y_mid / 255
            img_y_large = img_y_large / 255
            yield np.expand_dims(img_x, axis=0) , [np.expand_dims(img_y_mid, axis=0), np.expand_dims(img_y_large, axis=0)]

def next_y(s,e):
    img_directory = '../input/pascal-voc-2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007/JPEGImages/'
    arr = os.listdir(img_directory)
    while True:
        for i, p in enumerate(arr[s:e]):
            img = load_img(img_directory+p)
            img = np.array(img)
            yield img

def get_i_gen(gen, i):
    while i > 0:
        res = next(gen)
        i -= 1
    return res        

def new_gens(num_train=3600, num_val=900, num_max=5011):
    X_train_2 = next_X_y_mid(0, num_train + num_val)
    X_val_2 = next_X_y_mid(num_train, num_max)
    X_test_2 = next_X_y_mid(num_train + num_val, num_max) #511 test sampels
    X_train_3 = next_X_y_mid_large(0, num_train + num_val)
    X_val_3 = next_X_y_mid_large(num_train, num_max)
    X_test_3 = next_X_y_mid_large(num_train + num_val, num_max)
    return X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3

num_train = 3600
num_val = 900
num_max = 5011  
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()
import imageio
from PIL import Image
import keras
import matplotlib.animation as animation

def plot_gif(cols, imgs, titles, figsize=(20,20), rows=2):
    fig = plt.figure(figsize=figsize)
    for j in range(cols):
        ax = fig.add_subplot(rows, cols, j+1)
        ax.set_title(titles[j])
        plt.imshow(imgs[j])
    plt.show()


class Gif(keras.callbacks.Callback):
    
    def __init__(self, img, model, name,  mode='double'):
        super().__init__()
        self.img = np.expand_dims(img, axis=0)
        self.preds = []
        self.model = model
        self.name = name
        self.mode = mode 
    
    def on_epoch_end(self, batch, logs={}):
        pred = model.predict(self.img)
        self.preds.append(pred)
        
    def on_train_end(self, logs=None):
        times = 12
        size = (288, 288)
        plot_imgs = []
        imgs = []
        imgs2 = []
        for p in self.preds:
            if self.mode == 'solo':
                p = (p[0] * 255).astype(int)
                plot_imgs.append(p)
                for i in range(0,times):
                    imgs.append(p)
            elif self.mode =='double':
                p0 = (p[0][0] * 255).astype(int)
                p1 = (p[1][0] * 255).astype(int)
                plot_imgs.append(p1)
                for i in range(0,times):
                    imgs.append(p0)
                    imgs2.append(p1)
        if self.mode == 'solo':       
            imageio.mimsave('./'+self.name+'.gif', imgs)
        elif self.mode =='double':
            imageio.mimsave('./'+self.name+'_1.gif', imgs)
            imageio.mimsave('./'+self.name+'_2.gif', imgs2)
        self.show_plot(plot_imgs)
            
    def show_plot(self, imgs):
        titles = []
        for i in range(0, len(imgs)):
            titles.append('epoc '+ str(i))
        print(titles)
        plot_gif(len(imgs), imgs, titles, figsize=(40,40))
        
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.layers import Input,Add, Activation, LeakyReLU, Concatenate, Dense, Conv2D, UpSampling2D, Lambda
import keras.backend as K 
def PSNR(real, pred):
    max_val = 1.0
    alpha = 10.0
    div = 2.303
    p = 2
    inner = (max_val ** p) / K.mean(K.square(pred - real), axis=-1)
    return (alpha * K.log(inner)) / div
                                    
def get_metrics():
    return [PSNR]
                                    
def get_callbacks(model, name, mode='double'):
    img = resize_a[50] / 255
    early_stopping_monitor = EarlyStopping(patience=2) 
    gifer = Gif(img, model, name, mode=mode)
    return [early_stopping_monitor, gifer]
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()

def create_model():
    model = Sequential()
    model.add( Conv2D(64, kernel_size=(3,3), padding='same', activation='relu',input_shape =(72,72,3)))
    model.add( Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
    model.add( UpSampling2D(size=(2,2)))
    model.add( Conv2D(3, kernel_size=(1,1), activation='relu'))
    model.compile(loss='mse', optimizer='adam', metrics=get_metrics())
    model.summary()
    return model

model = create_model()

callbacks = get_callbacks(model ,'model1', 'solo')
history = model.fit_generator(X_train_2, steps_per_epoch=64, epochs=10, validation_data=X_val_2, validation_steps = 64,  callbacks=callbacks)
def quick_plot_history(history):
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
quick_plot_history(history)
    
def quick_plot_PSNR(history):
    # Plot training & validation loss values
    plt.plot(history.history['PSNR'])
    plt.plot(history.history['val_PSNR'])
    plt.title('Model loss')
    plt.title('Model PSNR')
    plt.ylabel('PSNR')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    
quick_plot_PSNR(history)  
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()
X_print = next_X(num_train + num_val, num_max)
# preds = model.predict(X_train[80:])
preds = model.predict_generator(X_test_2, 511)
lst = [(next(X_print) * 255).astype(int), (preds[0]* 255) .astype(int)]
titles = ['72*72','144*144']
better_plot(len(lst), lst, titles)
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()

def create_model_2():
    inp = Input(shape=(None,None,3), name='input')
    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inp)
    x = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = UpSampling2D(size=(2,2))(x)
    y = UpSampling2D(size=(2,2))(x)
    y = Conv2D(3, kernel_size=(1,1), activation='relu', name='output1')(y)
    x = Conv2D(3, kernel_size=(1,1), activation='relu', name='output2')(x)
    
    model = Model(inputs=inp,outputs=[x,y])
    model.compile(loss='mse', optimizer='adam', metrics=get_metrics())
    model.summary()
    return model

model = create_model_2()
callbacks = get_callbacks(model, 'model2')

history = model.fit_generator(X_train_3, steps_per_epoch=64, epochs=10, validation_data=X_val_3, validation_steps = 64,  callbacks=callbacks)
def quick_plot_history_3(history, name_conv1, name_conv2):
    plt.plot(history.history['loss'])
    plt.plot(history.history[name_conv1])
    plt.plot(history.history[name_conv2])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_'+name_conv1])
    plt.plot(history.history['val_'+name_conv2])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Train_144x144','Train_288x288', 'Val','Val_144x144', 'Val_288x288'], loc='upper left')
    plt.show()
    
quick_plot_history_3(history,'output1_loss', 'output2_loss')
def quick_plot_PSNR_3(history):
    # Plot training & validation loss values
    plt.plot(history.history['output2_PSNR'])
    plt.plot(history.history['output1_PSNR'])
    plt.plot(history.history['val_output2_PSNR'])
    plt.plot(history.history['val_output1_PSNR'])
    plt.title('Model PSNR')
    plt.ylabel('PSNR')
    plt.xlabel('Epoch')
    plt.legend(['Train 2 PSNR','Train 1 PSNR' ,'Val 2 PSNR','Val 1 PSNR'], loc='upper left')
    plt.show()

quick_plot_PSNR_3(history)
def visual_model(model, index = 5):
    X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()
    X_print = next_X(num_train + num_val, num_max)
    y_mid = next_y_mid(num_train + num_val, num_max)
    y_large = next_y_large(num_train + num_val, num_max)
    preds = model.predict_generator(X_test_3, 511)
    titles = ['input 72*72','original 144*144','pred 144*144','original 288*288','pred 288*288']
    lst = [(get_i_gen(X_print, index) * 255).astype(int), (get_i_gen(y_mid, index) * 255).astype(int) , (preds[0][index-1] * 255).astype(int) ,
            (get_i_gen(y_large, index) * 255).astype(int), (preds[1][index-1] * 255).astype(int)]
    better_plot(len(lst), lst, titles)

visual_model(model)
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()

def create_residual(h,w,z=32):
    inp = Input(shape=(h,w,z), name='input')
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(inp)
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    x = Add()([inp,x])
    x = Activation('relu')(x)
    return Model(inputs=inp,outputs=[x])

def create_model_3(h=None,w=None):
    inp = Input(shape=(h,w,3), name='input')
    x = Conv2D(32, kernel_size=(3,3),padding='same', activation='relu')(inp)
    x = create_residual(h,w)(x)
    x = create_residual(h,w)(x)
    x = UpSampling2D(size=(2,2))(x)
    y = create_residual(h,w)(x)
    y = UpSampling2D(size=(2,2))(y)
    y = Conv2D(3, kernel_size=(1,1), activation='relu', name='output1')(y)
    x = Conv2D(3, kernel_size=(1,1), activation='relu', name='output2')(x)
    model = Model(inputs=inp,outputs=[x,y])
    model.compile(loss='mse', optimizer='adam',metrics=get_metrics())
    model.summary()
    return model

model = create_model_3()

callbacks = get_callbacks(model, 'model3')

history = model.fit_generator(X_train_3, steps_per_epoch=64, epochs=10, validation_data=X_val_3, validation_steps = 64,  callbacks=callbacks)
quick_plot_history_3(history,'output1_loss', 'output2_loss')
quick_plot_PSNR_3(history)
visual_model(model)
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()

def create_delayed(h,w,z=32):
    inp = Input(shape=(h,w,z), name='input')
    x1 = Conv2D(32, kernel_size=(3,3), padding='same', dilation_rate = (1,1), activation=LeakyReLU(0.2))(inp)
    x2 = Conv2D(32, kernel_size=(3,3), padding='same', dilation_rate = (2,2), activation=LeakyReLU(0.2))(inp)
    x3 = Conv2D(32, kernel_size=(3,3), padding='same', dilation_rate = (4,4), activation=LeakyReLU(0.2))(inp)
    x = Concatenate()([x1,x2,x3])
    x = Activation('relu')(x)
    x = Conv2D(32, kernel_size=(3,3), padding='same', activation='relu')(x)
    return Model(inputs=inp,outputs=[x])

def create_model_4(h=None,w=None):
    inp = Input(shape=(h,w,3), name='input')
    x = Conv2D(32, kernel_size=(3,3),padding='same', activation=LeakyReLU(0.2))(inp)
    x = create_delayed(h,w)(x)
    x = create_delayed(h,w)(x)
    x = UpSampling2D(size=(2,2))(x)
    y = create_delayed(h,w)(x)
    y = UpSampling2D(size=(2,2))(y)
    y = Conv2D(3, kernel_size=(1,1), activation='relu', name='output1')(y)
    x = Conv2D(3, kernel_size=(1,1), activation='relu', name='output2')(x)
    model = Model(inputs=inp,outputs=[x,y])
    model.compile(loss='mse', optimizer='adam',metrics=get_metrics())
    model.summary()
    return model

model = create_model_4()

callbacks = get_callbacks(model, 'model4')
history = model.fit_generator(X_train_3, steps_per_epoch=64, epochs=10, validation_data=X_val_3, validation_steps = 64,  callbacks=callbacks)
quick_plot_history_3(history,'output1_loss', 'output2_loss')
quick_plot_PSNR_3(history)
visual_model(model)
from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input,Add, Activation, LeakyReLU, Concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Conv2D, UpSampling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

vgg = VGG16(weights='imagenet', include_top=False, input_shape = (None,None, 3))
vgg = Model(inputs=vgg.input,outputs=vgg.get_layer("block1_conv2").output)
X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()
early_stopping_monitor = EarlyStopping(patience=2) 

def create_model_5(h=None,w=None):
    inp = Input(shape=(h,w,3), name='input')
    x1 = Conv2D(64, kernel_size=(3,3),padding='same', activation=LeakyReLU(0.8))(inp)
    x1 = Conv2D(64, kernel_size=(3,3),padding='same', activation=LeakyReLU(0.8))(x1)
    x2 = vgg(inp)
    x = Concatenate()([x1, x2])
    x = UpSampling2D(size=(2,2))(x)
    x = Conv2D(3, kernel_size=(1,1), activation=LeakyReLU(0.2),name='output1')(x)
    y = UpSampling2D(size=(2,2))(x)
    y = Conv2D(3, kernel_size=(1,1), activation=LeakyReLU(0.2), name='output2')(y)
    model = Model(inputs=inp,outputs=[x,y])
    model.compile(loss='mse', optimizer='adam',metrics=get_metrics())
    model.summary()
    return model

model = create_model_5()

callbacks = get_callbacks(model, 'model5')

history = model.fit_generator(X_train_3, steps_per_epoch=64, epochs=10, validation_data=X_val_3, validation_steps = 64,  callbacks=callbacks)
quick_plot_history_3(history,'output1_loss', 'output2_loss')
quick_plot_PSNR_3(history)
visual_model(model)
import tensorflow as tf

X_train_2, X_val_2, X_test_2, X_train_3, X_val_3, X_test_3 = new_gens()

def create_model_6(h=None,w=None):
    inp = Input(shape=(h,w,3), name='input')
    x1 = Conv2D(64, kernel_size=(3,3),padding='same',  activation=LeakyReLU(0.3))(inp)
    x1 = Conv2D(64, kernel_size=(3,3),padding='same',  activation=LeakyReLU(0.3))(x1)
    x2 = vgg(inp)
    x = Concatenate()([x1, x2])
    x = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(x)
    x = Conv2D(3, kernel_size=(1,1), activation=LeakyReLU(0.2), name='output1')(x)
    y = Conv2D(12, kernel_size=(1,1), activation=LeakyReLU(0.2))(x)
    y = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2))(y)
    y = Conv2D(3, kernel_size=(1,1), activation=LeakyReLU(0.05), name='output2')(y) # 
    model = Model(inputs=inp,outputs=[x,y])
    model.compile(loss='mse', optimizer='adam', metrics=get_metrics())
    model.summary()
    return model

model = create_model_6()

metrics=get_metrics()
callbacks = get_callbacks(model, 'model6')

history = model.fit_generator(X_train_3, steps_per_epoch=64, epochs=10, validation_data=X_val_3, validation_steps = 64,  callbacks=callbacks)
quick_plot_history_3(history,'output1_loss', 'output2_loss')
quick_plot_PSNR_3(history)
visual_model(model)