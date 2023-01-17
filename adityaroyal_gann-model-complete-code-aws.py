# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        pass#print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

!pip install mtcnn

!pip install tqdm

!pip install smart_open

from __future__ import print_function, division



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from keras.datasets import mnist

from keras.layers import Input, Dense, Reshape, Flatten, Dropout

from keras.layers import LSTM

from keras.layers import TimeDistributed

from keras.layers import BatchNormalization, Activation, ZeroPadding2D

from keras.layers import LeakyReLU

from keras.models import Sequential, Model

from keras.optimizers import Adam

from mtcnn import MTCNN

from smart_open import smart_open

import matplotlib.pyplot as plt

import cv2

from io import BytesIO

import numpy as np

import boto3

import pandas as pd

from sagemaker import get_execution_role

from boto.s3.connection import S3Connection

from matplotlib import pyplot as plt

from PIL import Image

from tqdm.notebook import tqdm

import time



import sys



import numpy as np

print('second')

class GAN():

    def __init__(self):

        self.img_rows = 16

        self.img_cols = 16

        self.channels = 3

        self.total=299

        self.img_shape = (self.total,self.img_rows, self.img_cols, self.channels)

        self.fakeimg_dim=(299,16,16,3)

        self.latent_dim = self.fakeimg_dim

        



        # Build and compile the discriminator

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy',optimizer='Adam'

            ,metrics=['accuracy'])



        # Build the generator

        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs

        z = Input(shape=(self.fakeimg_dim))

        img = self.generator(z)



        print('cf v')

        

        print(img.shape)

        # For the combined model we will only train the generator

        self.discriminator.trainable = False

        self.discriminator.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])







        # The discriminator takes generated images as input and determines validity

        validity = self.discriminator(img)



        # The combined model  (stacked generator and discriminator)

        # Trains the generator to fool the discriminator

        self.combined = Model(z, validity)

        self.combined.compile(loss='binary_crossentropy',optimizer='Adam')







    def build_generator(self):

        model = Sequential()



        model.add(Dense(12, input_shape=self.fakeimg_dim))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(24))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(3))

        model.add(LeakyReLU(alpha=0.2))

        model.add(BatchNormalization(momentum=0.8))

        #model.add(Dense(np.prod(self.img_shape), activation='tanh'))

        #model.add(Reshape(self.img_shape))



        model.summary()



        noise = Input(shape=(self.fakeimg_dim))

        img = model(noise)

        

        

        return Model(noise, img)



    def build_discriminator(self):



        model = Sequential()

        # define CNN model

        # define LSTM model



        model.add(TimeDistributed(Dense(36),input_shape=self.img_shape))

        model.add(TimeDistributed(Flatten()))



        model.add(LeakyReLU(alpha=0.2))



        model.add(LSTM(24))

        model.add(Dense(24))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(12))

        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        img = Input(shape=(self.img_shape))

        

        

        opt = Adam()

        return Model(img, model(img))



    def train(self, epochs, batch_size=128, sample_interval=50):

        def plot_faces(images, figsize=(10.8/2, 19.2/2)):

                    shape = images[0].shape

                    try:

                        print('lenimages')

                        print(len(images))

                        print(shape)

                    except:

                        pass

                    images = images[np.linspace(0, len(images)-1, 16).astype(int)]

                    im_plot = []

                    for i in range(0, 16, 4):

                        im_plot.append(np.concatenate(images[i:i+4], axis=0))

                    im_plot = np.concatenate(im_plot, axis=1)



                    fig, ax = plt.subplots(1, 1, figsize=figsize)

                    try:

                        print('implotshape')

                        print(im_plot.shape)

                    except:

                        pass

                    ax.imshow(im_plot)

                    ax.xaxis.set_visible(False)

                    ax.yaxis.set_visible(False)



                    ax.grid(False)

                    fig.tight_layout()

        def timer(detector, detect_fn, images, *args):

                    start = time.time()

                    faces = detect_fn(detector, images, *args)

                    elapsed = time.time() - start

                    return faces, elapsed  

        def sample_images(self, epoch):

                r, c = 5, 5

                noise = np.random.normal(0, 1, (r * c, self.latent_dim))

                gen_imgs = self.generator.predict(noise)



                # Rescale images 0 - 1

                gen_imgs = 0.5 * gen_imgs + 0.5



                fig, axs = plt.subplots(r, c)

                cnt = 0

                for i in range(r):

                    for j in range(c):

                        axs[i,j].imshow(gen_imgs[cnt, :,:,0],cmap='gray')

                        axs[i,j].axis('off')

                        cnt += 1

                fig.savefig("/nm%d.png" % epoch)

                plt.close()

        def detect_mtcnn(detector, images):

                    nonlocal faces

                    faces = []

                    nonlocal faces2

                    faces2 = []

                    oldface=[]

                    error=[]

                    final=[]

                    for image in images:

                        boxes = detector.detect_faces(image)

                        try:

                            box = boxes[0]['box']

                            face = image[box[1]:box[3]+box[1], box[0]:box[2]+box[0]]

                            oldface=face

                            faces.append(face)

                        except:

                            try:

                                faces.append(oldface)

                            except:

                                pass

            

                    return faces

        """"The below code is working but difficult to understand as I had written it very badly. 

        I can explain it better here. I am having unzipped video files in a folder and the folder is in s3 bucket.

        I am initally reading json file from that folder in the bucket and from that json file I am reading a real,fake,real,

        fake,real,fake .... of all the videos I am sure the video files can be read in many ways but I am just reading that

        way so that the discriminator and the generator will be reading the similar videos when training. 

        I believe the weights will be robust when the training is not done in order. 

        By not in order I mean when a fake video is sent for generator to train, disriminator should use any other video 

        instead of it's real video."""

        total=[]

        final=[]

        import boto3

        s3 = boto3.resource('s3')

        my_bucket = s3.Bucket(yourbucketname)

        li=[]

        

        for object_summary in my_bucket.objects.all():

            l=object_summary.key

            li.append(l)

            import zipfile



        x=li

        #print(x)

        client = boto3.client('s3')

        dire2=[]

        for dire in x: 

            #my_bucket = yourbucketname

            #print(dire)

            #resi=client.list_objects(Bucket=my_bucket,Prefix=str(dire).split('/')[0]+'/')

            #for x in resi.get('Contents', [])[1:]:

            if str(dire).endswith('.json'):

                print(dire)

                break

            else:

                continue

            

        x1=dict()

        x=dict()

        x1['Key']=dire

        #print(li)

        my_bucket = yourbucketname

        df=pd.read_json(smart_open('s3://'+str(my_bucket)+'/'+x1['Key'])).T

        df=df.iloc[:,[1]].reset_index()

        df=df.rename(columns={'index':'fake'})

        tes=[]

        for a,b in zip(df['fake'],df['original']):

            if a in [x.split('/')[-1] for x in li[1:]] and b in [x.split('/')[-1] for x in li[1:]]:

                tes.append(a)

                tes.append(b)

        tes=pd.Series(tes).fillna(method='ffill')

        i=-1

        print(tes)

        for sample in list(tes)[:10]:

            i=i+1

            if i%2==0:

                final=[]

                try:

                    del old



                except:

                    pass

            reader = cv2.VideoCapture('http://'+str(my_bucket)+'.s3.amazonaws.com/'+li[0].split('/')[0]+'/'+sample)

            images_540_960 = []

            for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT)))):

                _, image = reader.read()

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                images_540_960.append(cv2.resize(image, (960, 540)))

            reader.release()

            images_540_960 = np.stack(images_540_960)



            try:

                diff=images_540_960 -old

            except:

                old=images_540_960 

            detector = MTCNN()

            times_mtcnn = []

            print('Detecting faces in 540x960 frames', end='')

            try:

                        faces2, elapsed2 = timer(detector, detect_mtcnn, old)

                        final.append(faces2)

            except:

                        faces, elapsed = timer(detector, detect_mtcnn, images_540_960)

                        final.append(faces)

        final=[[cv2.resize(face, (16, 16)) for face in x] for x in final]

        final=[np.array(x) for x in final]

        try:

            print('printing real faces')

            plot_faces(np.stack([cv2.resize(face, (16, 16)) for face in final]))

        except:

            plot_faces(np.stack([cv2.resize(face, (16, 16)) for face in diff]))



        if final[0].shape[0]>self.total:

            length=final[0].shape[0]-self.total

            np.append(final[0],np.zeros(length,16,16,3))

        else:

            final[0]=final[0][:self.total,:,:,:]

        if final[1].shape[0]>self.total:

            length=final[1].shape[0]-self.total

            np.append(final[0],np.zeros(length,16,16,3))

        else:

            final[1]=final[1][:self.total,:,:,:]

        try:

            X_train.extend(final)

        except:

            X_train= final



        X_train_num=[X_train[x] for x in range(len(X_train)) if x%2==0]

        lat_train_num=[X_train[x] for x in range(len(X_train)) if x%2!=0]

        X_train=np.array(X_train_num)

        lat_train=np.array(lat_train_num)

        valid = np.ones((batch_size, 1))

        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            idx = np.random.randint(0, X_train.shape[0], batch_size)

            imgs = X_train[idx]

            idx = np.random.randint(0, lat_train.shape[0], batch_size)

            global noise

            noise = lat_train[idx]

            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Generate a batch of new images

            print('dcsdc')

            gen_imgs = self.generator.predict(noise)

            try:

                model_json = self.generator.to_json()

                with open("model.json", "w") as json_file:

                    json_file.write(model_json)

                # serialize weights to HDF5

                self.generator.save_weights("model.h5")

            except Exception as e:

                print(e)

            

            # Train the discriminator

            print('dcs')

            print(gen_imgs.shape)

            print(imgs.shape)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)

            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)

            try:

                model_json =self.discriminator.to_json()

                with open("model3.json", "w") as json_file:

                    json_file.write(model_json)

                # serialize weights to HDF5

                self.discriminator.save_weights("model3.h5")

            except Exception as e:

                print(e)

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)



            # ---------------------

            #  Train Generator

            # ---------------------



            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            idx = np.random.randint(0, lat_train.shape[0], batch_size)

            noise = lat_train[idx]



            # Train the generator (to have the discriminator label samples as valid)

            g_loss = self.combined.train_on_batch(noise, valid)

            

           

        

            

            try:

                model_json2 = self.combined.to_json()

                with open("model2.json", "w") as json_file:

                    json_file.write(model_json2)

                # serialize weights to HDF5

                self.combined.save_weights("model2.h5")

            except Exception as e:

                print(e)



            # Plot the progress

            print ("%d [D loss: %f, acc.: %f] [G loss: %f]" % (epoch, d_loss[0], d_loss[1], g_loss))



            # If at save interval => save generated image samples

            if epoch >80:

                self.save_imgs(epoch)

                

    def save_imgs(self, epoch):

        r, c = 5, 5

        images = noise



        # Rescale images 0 - 1

        #images = 0.5 * images + 0.5



        #im_plot = []

        #for i in range(0, 16, 4):

            #im_plot.append(np.concatenate(images[i:i+4], axis=0))

        #im_plot = np.concatenate(im_plot, axis=1)



        fig, ax = plt.subplots(1, 1, figsize=(10.8/2, 19.2/2))

        print('imagesshape')

        print(images.shape)

        ax.imshow(images[0,0,:,:,0],cmap='gray')

        ax.xaxis.set_visible(False)

        ax.yaxis.set_visible(False)

        ax.grid(False)

        fig.savefig("output/mnist_%d.png" % epoch)

        #plt.close()









if __name__ == '__main__':

    gan = GAN()

    gan.train(epochs=100, batch_size=3, sample_interval=2)



# Any results you write to the current directory are saved as output.