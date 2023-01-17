import numpy as np

import matplotlib.pyplot as plt

import scipy.io as scio

import keras

from IPython.display import clear_output

import keras.utils as ult

from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization,Conv3D, MaxPooling3D, Dense, Add, Activation

from keras import regularizers

from keras.models import Model

from keras.optimizers import Adam, SGD, Adagrad, RMSprop

import time



from numpy.random import seed

seed(1)

from tensorflow import set_random_seed

set_random_seed(2)
class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        self.losses2 = []

        self.val_losses2 = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('loss'))

        self.val_losses.append(logs.get('val_loss'))

        self.losses2.append(logs.get('categorical_accuracy'))

        self.val_losses2.append(logs.get('val_categorical_accuracy'))



        self.i += 1

        

        clear_output(wait=True)

        plt.subplot(1,2,1)

        plt.plot(self.x, self.losses2, label="Training accuracy",linestyle='-')

        plt.plot(self.x, self.val_losses2, label="Testing accuracy",linestyle='--')

        plt.ylim(0,1)

        plt.legend()

        plt.xlabel('Epoch')

        plt.ylabel('Accuracy')

        

        plt.subplot(1,2,2)

        plt.plot(self.x, self.losses, label="Training loss",linestyle='-')

        plt.plot(self.x, self.val_losses, label="Testing loss",linestyle='--')



        plt.legend()

        plt.xlabel('Epoch')

        plt.ylabel('Loss')

        

        plt.tight_layout()

        

        plt.show();

        

plot_losses = PlotLosses()
def show_images(images,galaxy_labels):

    fig = plt.figure()

    plt.subplot(1,3,1)

    plt.title(label_trans(galaxy_labels[0]))

    plt.imshow(images[0,:,:,0], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,2)

    plt.imshow(images[0,:,:,1], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,3)

    plt.imshow(images[0,:,:,2], vmax=255)

    plt.axis('off')



    fig = plt.figure()

    plt.subplot(1,3,1)

    plt.title(label_trans(galaxy_labels[1]))

    plt.imshow(images[1,:,:,0], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,2)

    plt.imshow(images[1,:,:,1], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,3)

    plt.imshow(images[1,:,:,2], vmax=255)

    plt.axis('off')



    fig = plt.figure()

    plt.subplot(1,3,1)

    plt.title(label_trans(galaxy_labels[5]))

    plt.imshow(images[5,:,:,0], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,2)

    plt.imshow(images[5,:,:,1], vmax=255)

    plt.axis('off')

    plt.subplot(1,3,3)

    plt.imshow(images[5,:,:,2], vmax=255)

    plt.axis('off')
def label_trans(label_id):

    if label_id==0: return "star"

    if label_id==1: return "spiral galaxy"

    if label_id==2: return "elliptical galaxy"

    else: return "unknown"  
images=np.load("../input/galaxy_cubes.npy")

galaxy_labels=np.load("../input/labels.npy")

[images.shape, galaxy_labels.shape]



show_images(images,galaxy_labels) 
images= images+np.random.randn(10000,41,41,3)*20

images= np.clip(images, 0, 255)



show_images(images,galaxy_labels) 
num_train_examples =   2000

num_test_examples  =    500
images_train=images[0:num_train_examples,:,:,:]

images_test=images[2500:2500+num_test_examples,:,:,:]



Train_data=images_train.reshape(num_train_examples, images.shape[1],images.shape[2],images.shape[3],1)

Test_data=images_test.reshape(num_test_examples, images.shape[1],images.shape[2],images.shape[3],1)



train_labels=galaxy_labels[0:num_train_examples]

test_labels=galaxy_labels[2500:2500+num_test_examples]



train_labels_cat=ult.to_categorical(train_labels,num_classes=3)

test_labels_cat=ult.to_categorical(test_labels,num_classes=3)
inputs = Input((images.shape[1], images.shape[2], images.shape[3], 1),name='main_input')



conv00  = Conv3D(16, (3, 3, 2), strides=(1, 1, 1), padding='same', name='conv00')(inputs)

#bn00 = BatchNormalization()(conv00)

act00 = Activation('relu')(conv00)

pool00  = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(act00)





conv10  = Conv3D(16, (3, 3, 2), strides=(1, 1, 1), padding='same', name='conv10')(pool00)

#bn10 = BatchNormalization()(conv10)

act10 = Activation('relu')(conv10)

#add00 = Add()([pool00,act10])

pool10  = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(act10)





conv20  = Conv3D(16, (3, 3, 2), strides=(1, 1, 1), padding='same', name='conv20')(pool10)

#bn20 = BatchNormalization()(conv20)

act20 = Activation('relu')(conv20)

#add20 = Add()([pool10,act20])

pool20  = MaxPooling3D(pool_size=(3, 3, 1), strides=(2, 2, 1), padding='same')(act20)



fl0 = Flatten(name='fl0')(pool20)

#Do0 = Dropout(rate=0.5)(fl0)

fc0 = Dense(32,activation='linear')(fl0)

#Do1 = Dropout(rate=0.5)(fc0)

fc1 = Dense(8,activation='linear')(fc0)

#Do2 = Dropout(rate=0.5)(fc1)



Dn0 = Dense(3,activation='softmax', name='Dn0' )(fc1)



model_1 = Model(inputs=[inputs], outputs=[Dn0])
optzr =  Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, decay=0.0)

model_1.compile(loss='categorical_crossentropy',optimizer=optzr, metrics =['categorical_accuracy'])

model_1.summary()
start_time = time.time()

history=model_1.fit(Train_data,train_labels_cat, batch_size=20, epochs=50,validation_data=[Test_data,test_labels_cat],callbacks=[plot_losses],shuffle=True)

elapsed_time = time.time() - start_time

time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
ls,acc=model_1.evaluate(Test_data,test_labels_cat)

print("Loss value: %.2f" % (ls))  

print("Accuracy: %.1f" % (acc*100))   
preds=model_1.predict(Test_data[50:51,:,:,:,:])

print(preds)

print(test_labels[50])
lr1=model_1.layers[1].output  

lr2=model_1.layers[4].output

lr3=model_1.layers[7].output



activation_model_lr1 = Model(inputs=[inputs], outputs=lr1)

activation_model_lr2 = Model(inputs=[inputs], outputs=lr2)

activation_model_lr3 = Model(inputs=[inputs], outputs=lr3)
s = np.random.randint(0,100)

print(s)



plt.imshow(Test_data[s,:,:,0,0])

plt.title('Input image')

plt.show()



activations_lr1 = activation_model_lr1.predict(Test_data[s:s+1,:,:,:,:]) 

activations_lr2 = activation_model_lr2.predict(Test_data[s:s+1,:,:,:,:]) 

activations_lr3 = activation_model_lr2.predict(Test_data[s:s+1,:,:,:,:]) 



for i in range(16):

    img=activations_lr3[0,:,:,0,i]

    plt.imshow(img)

    plt.title('Number ' + str(i))

    plt.show()

plt.show()

def build_finetune_model(base_model, dropout, fc_layers, num_classes):

    for layer in base_model.layers:

        layer.trainable = False



    x = base_model.output

    x = Flatten()(x)

    for fc in fc_layers:

        # New FC layer, random init

        x = Dense(fc, activation='relu')(x) 

        x = Dropout(dropout)(x)



    # New softmax layer

    predictions = Dense(num_classes, activation='softmax')(x) 

    

    finetune_model = Model(inputs=base_model.input, outputs=predictions)



    return finetune_model
from keras.applications.resnet50 import ResNet50



base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(205, 205, 3))



FC_LAYERS = [256]

dropout = 0.5



finetune_model = build_finetune_model(base_model, dropout=dropout, fc_layers=FC_LAYERS, num_classes=3)



finetune_model.summary()

from PIL import Image

images2=np.empty([3000,205,205,3])

for i in range(3000):

    for c in range(3):

        tmp=images[i,:,:,c]

        img = Image.fromarray(tmp)

        img2 = img.resize((205,205),Image.BICUBIC)

        images2[i,:,:,c]=img2    



#show_figures(images2)
Train_data=images2[0:num_train_examples,:,:,:]

Test_data=images2[2500:2500+num_test_examples,:,:,:]



print(Train_data.shape)

print(Test_data.shape)



train_labels=galaxy_labels[0:num_train_examples]

test_labels=galaxy_labels[2500:2500+num_test_examples]



train_labels_cat=ult.to_categorical(train_labels,num_classes=3)

test_labels_cat=ult.to_categorical(test_labels,num_classes=3)



print(train_labels_cat.shape)

print(test_labels_cat.shape)
finetune_model.compile(optzr, loss='categorical_crossentropy', metrics=['categorical_accuracy'])



history = finetune_model.fit(Train_data,train_labels_cat, batch_size=20, epochs=50,validation_data=[Test_data,test_labels_cat],callbacks=[plot_losses],shuffle=True)

ls,acc=finetune_model.evaluate(Test_data,test_labels_cat)

print("Loss value: %.2f" % (ls))  

print("Accuracy: %.1f" % (acc*100))  