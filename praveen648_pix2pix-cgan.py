os.makedirs('..output/kaggle/working/Outputs')
import os

from os import listdir

from numpy import asarray

from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.preprocessing.image import load_img

from numpy import savez_compressed



from numpy import load

from matplotlib import pyplot

import numpy as np



from random import randint



from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Conv2D

from tensorflow.keras.layers import Conv2DTranspose

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Concatenate

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras.layers import LeakyReLU

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.utils import plot_model

def load_images(path,size=(256,512)):

    src_list,tar_list=list(),list()

    for filename in listdir(path):

        try:

            pixels=load_img(path+filename,target_size=size)

            pixels=img_to_array(pixels)

            sat_img,map_img=pixels[:,:256],pixels[:,256:]

            src_list.append(sat_img)

            tar_list.append(map_img)

        except OSError:

            continue

    return [asarray(src_list),asarray(tar_list)]



path='../input/pix2pix-dataset/facades/facades/train/'

#load data

[src_images,tar_images]=load_images(path=path)

print('Shape of Source and Target Images:',src_images.shape,tar_images.shape)
#save data

filename='facades.npz'

savez_compressed(filename,src_images,tar_images)

print('dataset saved')
#load dataset



data=load(filename)

src_images,tar_images=data['arr_0'],data['arr_1']
#plot images



n_samples=3



for i in range(n_samples):

    pyplot.subplot(2,n_samples,i+1)

    pyplot.axis('off')

    pyplot.imshow(src_images[i].astype('uint8'))



for i in range(n_samples):

    pyplot.subplot(2,n_samples,i+1+n_samples)

    pyplot.axis('off')

    pyplot.imshow(tar_images[i].astype('uint8'))



pyplot.show()
#Discriminator which takes Patches of both images and predicts which if both comes

#same pair or not. 



def define_discriminator(image_shape):



	#Weight initialisation

	init=RandomNormal(stddev=0.02)



	#source image input

	in_src_image=Input(shape=image_shape)

	in_target_image=Input(shape=image_shape)



	merged=Concatenate()([in_src_image,in_target_image])



	#64

	d=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(merged)

	d=LeakyReLU(alpha=0.2)(d)

	#128

	d=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#256

	d=Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#512

	d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#one more 512 layer convolution

	d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#Patch Output 

	d=Conv2D(1,(4,4),padding='same',kernel_initializer=init)(d)

	patch_out=Activation('sigmoid')(d)



	#define the model



	model=Model([in_src_image, in_target_image],patch_out)

	opt=Adam(lr=0.002,beta_1=0.5)

	model.compile(loss='binary_crossentropy',optimizer=opt,loss_weights=[0.5])

	return model
#Discriminator which takes Patches of both images and predicts which if both comes

#same pair or not. 



def define_discriminator(image_shape):



	#Weight initialisation

	init=RandomNormal(stddev=0.02)



	#source image input

	in_src_image=Input(shape=image_shape)

	in_target_image=Input(shape=image_shape)



	merged=Concatenate()([in_src_image,in_target_image])



	#64

	d=Conv2D(64,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(merged)

	d=LeakyReLU(alpha=0.2)(d)

	#128

	d=Conv2D(128,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#256

	d=Conv2D(256,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#512

	d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#one more 512 layer convolution

	d=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d)

	d=BatchNormalization()(d)

	d=LeakyReLU(alpha=0.2)(d)

	#Patch Output 

	d=Conv2D(1,(4,4),padding='same',kernel_initializer=init)(d)

	patch_out=Activation('sigmoid')(d)



	#define the model



	model=Model([in_src_image, in_target_image],patch_out)

	opt=Adam(lr=0.002,beta_1=0.5)

	model.compile(loss='binary_crossentropy',optimizer=opt,loss_weights=[0.5])

	return model



def define_encoder_block(layer_in,n_filters,batchnorm=True):

		

	#weight initialisation

	init=RandomNormal(stddev=0.02)

	g=Conv2D(n_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)

	if batchnorm:

		g=BatchNormalization()(g,training=True)

	g=LeakyReLU(alpha=0.2)(g)

	return g







def decoder_block(layer_in,skip_in,n_filters,dropout=True):





	#weight initialisation

	init=RandomNormal(stddev=0.02)

	g=Conv2DTranspose(n_filters,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(layer_in)

	g=BatchNormalization()(g,training=True)

	if dropout:

		g=Dropout(0.5)(g,training=True)

	g=Concatenate()([g,skip_in])

	g=Activation('relu')(g)

	return g



#define standalone generator model



def define_generator(image_shape=(256,256,3)):

	init=RandomNormal(stddev=0.2)



	in_image=Input(shape=image_shape)

	#image input

	e1=define_encoder_block(in_image,64,batchnorm=False)

	e2=define_encoder_block(e1,128)

	e3 = define_encoder_block(e2, 256)

	e4 = define_encoder_block(e3, 512)

	e5 = define_encoder_block(e4, 512)

	e6 = define_encoder_block(e5, 512)

	e7 = define_encoder_block(e6, 512)



	b=Conv2D(512,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(e7)

	b=Activation('relu')(b)



	#Decoder Block

	d1=decoder_block(b,e7,512)

	d2=decoder_block(d1,e6,512)

	d3 = decoder_block(d2, e5, 512)

	d4 = decoder_block(d3, e4, 512, dropout=False)

	d5 = decoder_block(d4, e3, 256, dropout=False)

	d6 = decoder_block(d5, e2, 128, dropout=False)

	d7 = decoder_block(d6, e1, 64, dropout=False)



	#output

	g=Conv2DTranspose(3,(4,4),strides=(2,2),padding='same',kernel_initializer=init)(d7)

	out_image=Activation('tanh')(g)



	#Define Model

	model=Model(in_image,out_image)

	return model
	

#Define Combined discriminator and Generator



def define_gan(g_model,d_model,image_shape):



	d_model.trainable=False



	in_src=Input(shape=image_shape)



	#Connect source to generator inpout

	gen_out=g_model(in_src)

	#Connect input image and generator output to discriminator input

	dis_out=d_model([in_src,gen_out])

	# src image as input, generated image and classification output

	model=Model(in_src,[dis_out,gen_out])



	#model compile

	opt=Adam(lr=0.002,beta_1=0.5)

	model.compile(loss=['binary_crossentropy','mae'],optimizer=opt,loss_weights=[1,100])

	return model


image_shape=(256,256,3)



d_model=define_discriminator(image_shape)

g_model=define_generator(image_shape)



gan_model=define_gan(g_model,d_model,image_shape)



gan_model.summary()


#Select Real samples from dataset

def load_real_samples(filename):

	data=load(filename)

	X1,X2=data['arr_0'],data['arr_1']

	X1=(X1-127.5)/255

	X2=(X2-127.5)/255

	return [X1,X2]



def generate_real_samples(dataset,n_samples,patch_shape):

	trainA,trainB=dataset

	#choose random instances from dataset

	ix=np.random.randint(trainA.shape[0],size=n_samples)

	X1,X2=trainA[ix],trainB[ix]

	#generate ones of patch size

	y=np.ones((n_samples,patch_shape,patch_shape,1))

	return [X1,X2],y



def generate_fake_samples(g_model,samples,patch_shape):

	#forward pass

	X = g_model.predict(samples)

	y=np.zeros((len(X),patch_shape,patch_shape,1))

	return X,y


#generate samples save as plot and save the model in h5 format

def summarise_performance(step,g_model,d_model,dataset,n_samples=3):

    #generate source images for evaluation

    [X_realA,X_realB],_= generate_real_samples(dataset,n_samples,1)

    X_fakeB,_=generate_fake_samples(g_model,X_realA,1)

    #SCALES real and fake images from [-1,1] to [0,1]

    X_realA=(X_realA+1)/2.0

    X_realB=(X_realB+1)/2.0

    X_fakeB=(X_fakeB+1)/2.0



#plot source image

    for i in range(n_samples):

        pyplot.subplot(3,n_samples,1+i)

        pyplot.axis('off')

        pyplot.imshow(X_realA[i])

    #plot fake target image

    for i in range(n_samples):

        pyplot.subplot(3,n_samples,1+i+n_samples)

        pyplot.axis('off')

        pyplot.imshow(X_fakeA[i])

    #plot real target

    for i in range(n_samples):

        pyplot.subplot(3,n_samples,1+i+n_samples*2)

        pyplot.axis('off')

        pyplot.imshow(X_realB[i])

    pyplot.show()

    

    #save plot to file

    filename1='..output/kaggle/working/Outputs/plot_%06d.png'%(step+1)

    pyplot.savefig(filename1)

    pyplot.close()



    #save the generator model

    filename2='..output/kaggle/working/Outputs/model_%06d.h5'%(step+1)

    g_model.save(filename2)

    print('Saved %s and %s',(filename1,filename2))
def train(d_model,g_model,gan_model,dataset,n_epochs=100,n_batch=32):

    n_patch=d_model.output_shape[1]

    trainA,trainB=dataset

    bat_per_epo = int(len(trainA)/n_batch)

    n_steps = bat_per_epo*n_epochs



    for i in range(n_steps):

        [X_realA,X_realB],y_real = generate_real_samples(dataset,n_batch,n_patch)

        X_fakeA,y_fake=generate_fake_samples(g_model,X_realA,n_patch)



        d_loss1 = d_model.train_on_batch([X_realA,X_realB],y_real)



        d_loss2 = d_model.train_on_batch([X_realA,X_fakeA],y_fake)



        g_loss,_,_ = gan_model.train_on_batch(X_realA,[y_real,X_realB])



        #summarise_performance

        print('>%d,d1.[%.3f] d2.[%.3f], g[%.3f]',(i+1, d_loss1, d_loss2, g_loss))



        if (i+1) % (bat_per_epo*10)==0:

            summarise_performance(i,g_model,d_model,dataset,3)
dataset=load_real_samples('facades.npz')

print('loaded', dataset[0].shape, dataset[1].shape)
image_shape = dataset[0].shape[1:]



d_model=define_discriminator(image_shape)

g_model=define_generator(image_shape)



gan_model = define_gan(g_model,d_model,image_shape)



train(d_model,g_model,gan_model,dataset)

summarise_performance(step=i,g_model=g_model,d_model=d_model,dataset=dataset,n_samples=3)




from tensorflow.keras.models import load_model

from numpy import load

from matplotlib import pyplot

from numpy.random import randint





def plot_images(src_img,gen_img,tar_img):

    images = np.vstack((src_img,gen_img,tar_img))



    images = (images + 1)/2

    for i in range(len(images)):

        pyplot.subplot(1,3,1+i)

        pyplot.axis('off')

        pyplot.imshow(images[i])

    pyplot.show()



[src_img,tar_img],_ = generate_real_samples(dataset=dataset,n_samples=1,patch_shape=d_model.output_shape[1])





model = load_model('..output/kaggle/working/Outputs/model_000960.h5')

gen_img = model.predict(src_img)

plot_images(src_img,gen_img,tar_img)