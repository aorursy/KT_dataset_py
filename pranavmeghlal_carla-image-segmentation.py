import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from tqdm import notebook
from multiprocessing import Pool

from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint
import keras.backend as K
import tensorflow as tf
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator 
src_dir = '../input/lyft-udacity-challenge/dataA/dataA'
input_imgs_names = os.listdir(os.path.join(src_dir,'CameraRGB'))
mask_names = os.listdir(os.path.join(src_dir,'CameraSeg'))
index = random.randint(0,len(input_imgs_names)-1)

img = cv2.imread(os.path.join(src_dir,'CameraRGB',input_imgs_names[index]))
mask = cv2.imread(os.path.join(src_dir,'CameraSeg',mask_names[index]))

plt.figure()
plt.imshow(img)

plt.figure()
plt.imshow(mask[:,:,2])

def split_img(mask_img):    
    new_mask_img = np.zeros((mask_img.shape[0],mask_img.shape[1],13))
    
    for j in range(13):
        for k in range(mask_img.shape[0]):
            for l in range(mask_img.shape[1]):
                if mask_img[k,l,2] == j:
                    new_mask_img[k,l,j] = j
    return new_mask_img
dest_dir = '/kaggle/working/'

os.makedirs(os.path.join(dest_dir,'Inputs'), exist_ok = True)
os.makedirs(os.path.join(dest_dir,'Outputs'), exist_ok = True)


input_images = []
mask_images = []

for i in notebook.tqdm(range(1000)):
    input_img = cv2.imread(os.path.join(src_dir,'CameraRGB',input_imgs_names[i]))
    input_img = cv2.resize(input_img, (150,200))
    input_images.append(input_img)
#     inp_file = 'inp_'+str(i)+'.png'
#     cv2.imwrite(os.path.join(dest_dir,'Inputs',inp_file), input_img)

    mask_img = cv2.imread(os.path.join(src_dir,'CameraSeg',mask_names[i]))
    mask_img = cv2.resize(mask_img,(144,192))
    
    newimg = mask_img[:,:,2]
    
#     newimg = np.zeros((mask_img.shape[0],mask_img.shape[1]))
    
#     newimg[np.where(mask_img==7)[0],np.where(mask_img==7)[1]] = 1
#     op_img = mask_img[:,:,2]
    
#     op_file = 'op_'+str(i)+'.png'
#     cv2.imwrite(os.path.join(dest_dir,'Outputs',op_file), op_img)

    mask_images.append(newimg)

input_images = np.array(input_images)
input_images = input_images/255.
mask_images = np.array(mask_images)
mask_images = mask_images.reshape((mask_images.shape[0], mask_images.shape[1], mask_images.shape[2], 1))

print(input_images.shape, mask_images.shape)
index = random.randint(0,1000)

# input_file = os.listdir(os.path.join(dest_dir, 'Inputs'))[index]
# output_file = os.listdir(os.path.join(dest_dir, 'Outputs'))[index]

# input_img = cv2.imread(os.path.join(dest_dir, 'Inputs', input_file))
# output_img = cv2.imread(os.path.join(dest_dir, 'Outputs', output_file), 0)

# print(input_img.shape, output_img.shape)

input_img = input_images[index]
output_img = mask_images[index]

img_ht = input_img.shape[0]
img_wd = input_img.shape[1]
img_ly = input_img.shape[2]

print(img_ht, img_wd, img_ly)

plt.figure()
plt.imshow(input_img)

plt.figure()
plt.imshow(output_img.reshape(output_img.shape[0], output_img.shape[1]))

# print(output_img)


datagen = ImageDataGenerator(validation_split = 0.1)

training_generator = datagen.flow(input_images, mask_images, batch_size = 32, subset = 'training')
validation_generator = datagen.flow(input_images, mask_images, batch_size = 32, subset = 'validation')

main_input = Input(shape=(img_ht,img_wd,img_ly), name = 'img_input')

''' ~~~~~~~~~~~~~~~~~~~ ENCODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

c1 = Conv2D(32, kernel_size=(3,3), padding = 'same')(main_input)
c1 = LeakyReLU(0.2)(c1)
c1 = BatchNormalization()(c1)
c1 = Conv2D(32, kernel_size=(3,3), padding = 'same')(c1)
c1 = LeakyReLU(0.2)(c1)
c1 = BatchNormalization()(c1)

p1 = MaxPooling2D((2,2))(c1)

c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(p1)
c2 = LeakyReLU(0.2)(c2)
c2 = BatchNormalization()(c2)
c2 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c2)
c2 = LeakyReLU(0.2)(c2)
c2 = BatchNormalization()(c2)

p2 = MaxPooling2D((2,2))(c2)

c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(p2)
c3 = LeakyReLU(0.2)(c3)
c3 = BatchNormalization()(c3)
c3 = Conv2D(32*2, kernel_size=(1,1), padding = 'same')(c3)
c3 = LeakyReLU(0.2)(c3)
c3 = BatchNormalization()(c3)
c3 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c3)
c3 = LeakyReLU(0.2)(c3)
c3 = BatchNormalization()(c3)

p3 = MaxPooling2D((2,2))(c3)

c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p3)
c4 = LeakyReLU(0.2)(c4)
c4 = BatchNormalization()(c4)
c4 = Conv2D(32*4, kernel_size=(1,1), padding = 'same')(c4)
c4 = LeakyReLU(0.2)(c4)
c4 = BatchNormalization()(c4)
c4 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(c4)
c4 = LeakyReLU(0.2)(c4)
c4 = BatchNormalization()(c4)

p4 = MaxPooling2D((2,2))(c4)

c5 = Conv2D(32*6, kernel_size=(3,3), padding = 'same')(p4)
c5 = LeakyReLU(0.2)(c5)
c5 = BatchNormalization()(c5)


''' ~~~~~~~~~~~~~~~~~~~ DECODING LAYERS ~~~~~~~~~~~~~~~~~~~ '''

u1 = UpSampling2D((2,2))(c5)
concat1 = concatenate([u1, Cropping2D(((0,1), (0,0)))(c4)])

c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(concat1)
c6 = LeakyReLU(0.2)(c6)
c6 = BatchNormalization()(c6)
c6 = Conv2D(32*4, kernel_size=(3,3), padding = 'same')(c6)
c6 = LeakyReLU(0.2)(c6)
c6 = BatchNormalization()(c6)


u2 = UpSampling2D((2,2))(c6)
concat2 = concatenate([Cropping2D(((1,1), (0,1)))(c3), u2])

c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(concat2)
c7 = LeakyReLU(0.2)(c7)
c7 = BatchNormalization()(c7)
c7 = Conv2D(32*2, kernel_size=(3,3), padding = 'same')(c7)
c7 = LeakyReLU(0.2)(c7)
c7 = BatchNormalization()(c7)

u3 = UpSampling2D((2,2))(c7)
concat3 = concatenate([Cropping2D(((2,2), (1,2)))(c2), u3])

c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(concat3)
c8 = LeakyReLU(0.2)(c8)
c8 = BatchNormalization()(c8)
c8 = Conv2D(32, kernel_size=(3,3), padding = 'same')(c8)
c8 = LeakyReLU(0.2)(c8)
c8 = BatchNormalization()(c8)

u4 = UpSampling2D((2,2))(c8)
concat4 = concatenate([Cropping2D(((4,4),(3,3)))(c1), u4])

c9 = Conv2D(16, kernel_size = (1,1), padding = 'same')(concat4)
c9 = LeakyReLU(0.2)(c9)
c9 = BatchNormalization()(c9)

mask_out = Conv2D(13, (1,1), padding = 'same', activation = 'sigmoid', name = 'mask_out')(c9)

model = Model(inputs = [main_input], outputs = [mask_out])

# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
# model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
history = model.fit(input_images, mask_images, batch_size=32, epochs = 100, validation_split = 0.1)
# history = model.fit_generator(training_generator,steps_per_epoch=(len(input_images)*0.9)//32, epochs=10, validation_data=validation_generator, validation_steps=(len(input_images)*0.1)//32)
plt.figure()
plt.plot(history.history['accuracy'], label = 'Training')
plt.plot(history.history['val_accuracy'], label = 'Accuracy')
plt.title('Accuracy')
plt.legend()

plt.figure()
plt.plot(history.history['loss'], label = 'Training')
plt.plot(history.history['val_loss'], label = 'Accuracy')
plt.title('loss')
plt.legend()
index = random.randint(0,1000)

input_img = input_images[index]
output_img = mask_images[index]

img_ht = input_img.shape[0]
img_wd = input_img.shape[1]
img_ly = input_img.shape[2]

# print(img_ht, img_wd, img_ly)

plt.figure()
plt.imshow(input_img)

plt.figure()
plt.imshow(output_img.reshape(output_img.shape[0], output_img.shape[1]))

pred = model.predict(input_img.reshape((1, input_img.shape[0], input_img.shape[1], input_img.shape[2] )))

print(pred.shape)
                     
newimg = np.zeros((pred.shape[1], pred.shape[2]))

for i in range(pred.shape[3]):
    for j in range(pred.shape[1]):
        for k in range(pred.shape[2]):
            if pred[0,j,k,i] > 0.3:
                newimg[j,k] = i

plt.figure()
plt.imshow(newimg)
