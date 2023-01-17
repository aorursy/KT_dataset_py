%cd /kaggle/working
! wget http://images.cocodataset.org/zips/val2017.zip
!unzip val2017.zip
import os

import tensorflow as tf

import numpy as np

import PIL

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import tensorflow

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import add,Dense

from tensorflow.keras.layers import UpSampling2D

from tensorflow.keras.layers import LeakyReLU, PReLU

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import Activation,Flatten

from tensorflow.keras.applications.vgg19 import VGG19

from tensorflow.keras.optimizers import Adam

import tensorflow.keras.backend as K

from IPython.display import display

from skimage.transform import rescale, resize

hr_shape=(384,384,3)

lr_shape=(hr_shape[0]/4,hr_shape[1]/4,hr_shape[2])
## Loading Data ##



def load_data(directory):

  images = []

  count = 0

  for img in os.listdir(directory):

    

    if(count==700):

      break



    img = PIL.Image.open(os.path.join(directory,img)).convert('RGB')

    img_resized=np.array(img.resize((hr_shape[0],hr_shape[1]),PIL.Image.BICUBIC))

    images.append(img_resized)

    count+=1

  

  return images



directory = "/kaggle/working/val2017"

data = load_data(directory)

                                                # data is a list containing path of 5k images

X_train = data[0:500]                           # used 1000 images for training

X_test = data[500:700]                        # used 280 images for validation or testing



print(X_train[270].shape)


# Takes list of images and provide HR images in form of numpy array

def high_res_images(images):

    HR_images = np.array(images)

    return HR_images                              # Shape (np_of_images, H, W, no_of_channels)



# Takes list of images and provide LR images in form of numpy array

# i.e., Downsampling function

# use downscale = 4

    

def low_res_images(images_real , downscale):



    images = []

    for img in  range(len(images_real)):

        img1 = PIL.Image.fromarray(images_real[img])

        img2 = img1.resize((images_real[img].shape[0]//downscale,images_real[img].shape[1]//downscale),PIL.Image.BICUBIC)



        images.append(np.array(img2))



    images_lr = np.array(images)

    return images_lr



def normalize_img(input_data):

  return (input_data.astype(np.float32) - 127.5)/127.5 



def denormalize(input_data):

    input_data = (input_data + 1) * 127.5

    return input_data.astype(np.uint8) 



print(len(X_train))
print(len(X_test))
X_train_hr = high_res_images(X_train)

X_train_hr = normalize_img(X_train_hr)



X_train_lr = low_res_images(X_train, 4)

X_train_lr = normalize_img(X_train_lr)             ### Need some work to denormalize and converting it back to unit8 before imshow()





X_test_hr = high_res_images(X_test)

X_test_hr = normalize_img(X_test_hr)



X_test_lr = low_res_images(X_test, 4)

X_test_lr = normalize_img(X_test_lr)
###Upsample function(changing from H * W * C to H * W * 4C then to 2H * 2W * C using pixelshuffling)

def upsample(model,filter_size,no_of_channels,strides):

  #scaling factor=2

  scale=2

  no_of_filters=no_of_channels *(scale ** 2)

  model=Conv2D(filters=no_of_filters,kernel_size=filter_size,strides=strides,padding='same')(model)

  model=UpSampling2D(size=scale)(model)

  model=PReLU()(model)

  return model


def residual_block(model, kernel_size, no_of_filters, strides):

    

    gen = model

    model = Conv2D(filters = no_of_filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)

    model = BatchNormalization(momentum = 0.5)(model)

    

    # Using Parametric ReLU

    model = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1,2])(model)

    model = Conv2D(filters = no_of_filters, kernel_size = kernel_size, strides = strides, padding = "same")(model)

    model = BatchNormalization(momentum = 0.5)(model)

        

    model = add([gen, model])

    

    return model

# Using Functional API of Keras



def generator_network(gen_input):



  gen_input = Input(shape = gen_input)

     

  model = Conv2D(filters = 64, kernel_size = 9, strides = 1, padding = "same")(gen_input)

  model = PReLU(shared_axes=[1,2])(model)               # each filter only has one set of parameters

  model_part1 = model

        

  # Using 16 Residual Blocks

  for i in range(16):

      model = residual_block(model, 3, 64, 1)

      # 16 residual blocks with skip connections 



  model = Conv2D(filters = 64, kernel_size = 3, strides = 1, padding = "same")(model)

  model = BatchNormalization(momentum = 0.5)(model)

  model = add([model_part1, model])                      

  #  Element wise of model_part1 and model after 16 residual blocks

     



  for i in range(2):

      model = upsample(model, 3, 64, 1)  #no of channels=64  

     

  model = Conv2D(filters = 3, kernel_size = 9, strides = 1, padding = "same")(model)

  

  model = Activation('tanh')(model)                  # tanh activation in last layer

    

  gen_model = Model(inputs = gen_input, outputs = model)     # specifying the input and output to the model

  

  return gen_model
def conv_disc_block(model,filters,filter_size,strides):

  model=Conv2D(filters=filters,kernel_size=filter_size,strides=strides,padding='same')(model)

  model = BatchNormalization(momentum = 0.5)(model)

  model=LeakyReLU(alpha=0.1)(model)

  return model
def discriminator_network(image_shape):

  disc_input=Input(shape = image_shape)

  #convolution layer(k3n64s1)

  model=Conv2D(filters = 64,kernel_size = 3,strides=1,padding='same' )(disc_input)

  #Activation-leaky relu

  model=LeakyReLU(alpha=0.1)(model)

  

  #discriminator block (k3n64s2)

  model=conv_disc_block(model,64,3,2)

  #discriminator block (k3n128s1)

  model=conv_disc_block(model,128,3,1)

  #discriminator block (k3n128s2)

  model=conv_disc_block(model,128,3,2)

  #discriminator block (k3n256s1)

  model=conv_disc_block(model,256,3,1)

  #discriminator block (k3n256s2)

  model=conv_disc_block(model,256,3,2)

  #discriminator block (k3n512s1)

  model=conv_disc_block(model,512,3,1)

  #discriminator block (k3n512s2)

  model=conv_disc_block(model,512,3,2)



  #for dense layer input should be column vector/flatten

  model=Flatten()(model)

  #dense layer with 1024 nodes

  model=Dense(1024)(model)

  #Activation-leaky relu

  model=LeakyReLU(alpha=0.1)(model)



  #dense layer with 1 nodes

  model=Dense(1)(model)

  #Activation-sigmoid

  model=Activation('sigmoid')(model)

  disc_model=Model(inputs=disc_input,outputs=model)

  return disc_model
# To use mean, square we need to use keras.backend

# For content loss, compare the results which the vgg19 provides which feeding the y_true and y_pred



def content_loss(image_shape):

  def loss( y_true, y_pred):

      vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=image_shape)

      vgg19.trainable = False

      for layer in vgg19.layers:

          layer.trainable = False

      model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)

      model.trainable = False

      return K.mean(K.square(model(y_true) - model(y_pred)))

  return loss  
def gan_network(generator_model,discriminator_model,shape):

  discriminator_model.trainable = False

  gan_input=Input(shape=shape)

  print("input")

  SR=generator_model(gan_input)

  print("SR")

  gan_output=discriminator_model(SR)

  print("gan_output")

  model=Model(inputs=gan_input,outputs=[SR,gan_output])

  return model
import os



os.makedirs('/kaggle/working/Super-Resolve', exist_ok = True)




def plot_generated_images(epoch,generator, examples=3 , dim=(1, 3), figsize=(15, 5)):  

    

    rand_nums = np.random.randint(0, X_test_hr.shape[0], size=examples)

    image_batch_hr = denormalize(X_test_hr[rand_nums])

    image_batch_lr = X_test_lr[rand_nums]

    gen_img = generator.predict(image_batch_lr)

    generated_image = denormalize(gen_img)

    image_batch_lr = denormalize(image_batch_lr)

    

    

    plt.figure(figsize=figsize)

    

    plt.subplot(dim[0], dim[1], 1)

    plt.imshow(image_batch_lr[1], interpolation='nearest')

    plt.axis('off')

        

    plt.subplot(dim[0], dim[1], 2)

    plt.imshow(generated_image[1], interpolation='nearest')

    plt.axis('off')

    

    plt.subplot(dim[0], dim[1], 3)

    plt.imshow(image_batch_hr[1], interpolation='nearest')

    plt.axis('off')

    

    plt.tight_layout()

    plt.savefig('/kaggle/working/Super-Resolve/gan_generated_image_epoch_%d.png' % epoch)

 
np.random.seed(10)
hr_shape
from tqdm import tqdm



def train_model(batch_size,epochs):

  no_of_batches=X_train_hr.shape[0]//batch_size

  adam = Adam(lr=0.0001 ,beta_1=0.9 ,beta_2=0.999, epsilon=1e-08 )

  discriminator_model = discriminator_network(hr_shape)

  generator_model = generator_network(lr_shape)

  generator_model.compile(loss=content_loss(hr_shape), optimizer=adam)

  discriminator_model.compile(loss='binary_crossentropy',optimizer=adam)

  

  gan_model=gan_network(generator_model,discriminator_model,lr_shape)

  discriminator_model.trainable = False

  gan_model.compile(loss=[content_loss(hr_shape),'binary_crossentropy'],loss_weights=[1.0,1e-3],optimizer=adam)





  for i in range(0,epochs):

    print("\nEpoch    : "+ str(i))



    for j in range(no_of_batches):



      print("Batch    : "+str(j))



      rand_nums = np.random.randint(0, X_train_hr.shape[0], size=batch_size)

      

      image_batch_hr = X_train_hr[rand_nums]

      image_batch_lr = X_train_lr[rand_nums]



      batch_gen_sr = generator_model.predict(image_batch_lr)

      real_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2      ## Here we use concept of label smoothing

      fake_data_Y = np.random.random_sample(batch_size)*0.2



      discriminator_model.trainable = True

      d_loss_real = discriminator_model.train_on_batch(image_batch_hr, real_data_Y)

      d_loss_fake = discriminator_model.train_on_batch(batch_gen_sr, fake_data_Y)

      d_loss = 0.5*np.add(d_loss_fake, d_loss_real)



      discriminator_model.trainable = False     

      gan_data_Y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2

      loss_gan = gan_model.train_on_batch(image_batch_lr, [image_batch_hr,gan_data_Y])





    print("discriminator_loss : %f" % d_loss)

    print("gan_loss :", loss_gan)

    loss_gan = str(loss_gan)

    

    loss_file = open( '/kaggle/working/losses.txt' , 'a')

    loss_file.write('epoch%d : gan_loss = %s ; discriminator_loss = %f\n' %(i, loss_gan, d_loss) )

    loss_file.close()



    plot_generated_images(i, generator_model)



    generator_model.save('/kaggle/working/Super-Resolve/gen_model.h5')

    discriminator_model.save('/kaggle/working/Super-Resolve/dis_model.h5')
tf.config.experimental_run_functions_eagerly(True)
lr_shape = tuple(map(int, lr_shape))
# Not Enough RAM to train on kaggle, but I have successfully trained it on colab

# To train with batch size 4 and epochs 50



# train_model(4, 50)