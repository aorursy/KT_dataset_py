import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Concatenate,Dense,Input,Flatten

from tensorflow.keras import Model

from tensorflow.keras.applications import DenseNet169

import tensorflow

import tensorflow as tf
#Import DenseNet169 Model without softmax layer

class DenseNetEncoder(Model):

    def __init__(self):

        super(DenseNetEncoder,self).__init__()

        self.base_model=DenseNet169(input_shape=(224,224,3),include_top=False,weights='imagenet')

        self.encoder=Model(inputs=self.base_model.inputs,outputs=self.base_model.outputs)

        print('Base model loaded {}'.format(DenseNet169.__name__))

        

    def call(self,x):

        print('building basemodel')

        return self.encoder(x)
from tensorflow.python.ops.variables import Variable

#Concat DenseNet Output and metadata with a fully connected layer and predict an outcome

class Predictor(Model):

    def __init__(self):

        super(Predictor,self).__init__()

        self.flatten_layer=Flatten()

        self.dense_relu=Dense(16,activation='relu')

        self.dense_sigmoid=Dense(1,activation='sigmoid')

        

    def call(self,x):

        flat = self.flatten_layer(x)

        fcl0=self.dense_relu(flat)

        



        #for k, v in locals().items():

        #    if type(v) is Variable or type(v) is tf.Tensor:

        #        print("{0}: {1}".format(k, v)) 

        return self.dense_sigmoid(fcl0)        
#Combine densenet and predictor module together

class Classifier(Model):

    def __init__(self):

        super(Classifier,self).__init__()

        self.encoder=DenseNetEncoder()

        self.predictor=Predictor()

        print('\nModel created.')

    

    def call(self,x):

        image_features=self.encoder(x)

        return self.predictor(image_features)
#Load dataframes of images to store context

train = pd.read_csv('../input/siim-isic-melanoma-classification/train.csv')

train['path']='../input/siim-isic-melanoma-classification/jpeg/train/'+train['image_name']+'jpg'



test = pd.read_csv('../input/siim-isic-melanoma-classification/test.csv')

test['path']='../input/siim-isic-melanoma-classification/jpeg/train/'+test['image_name']+'jpg'
from sklearn.utils import shuffle

import tensorflow

from tensorflow.image import ResizeMethod



class DataLoader():

    def __init__(self, csv_file='../input/siim-isic-melanoma-classification/train.csv',\

                 img_path='../input/siim-isic-melanoma-classification/jpeg/train/',DEBUG=False):

        self.csv_file = csv_file

        self.img_path = img_path

        

        self.data_read(DEBUG=DEBUG)

    

    def data_read(self,DEBUG=False):

        train_data = pd.read_csv(self.csv_file)

        train_data['path'] = self.img_path+train_data['image_name']+'.jpg'

        

        meta_data = train_data[['path','sex','age_approx','anatom_site_general_challenge',]]

        for col in ['sex','age_approx','anatom_site_general_challenge']:

            meta_data[col].fillna(meta_data[col].mode()[0],inplace=True)

        meta_data = pd.get_dummies(meta_data,columns=['sex','anatom_site_general_challenge'])



        meta_data.drop(columns=['sex_female','anatom_site_general_challenge_head/neck'],inplace=True)



        meta_data = meta_data[['path', 'age_approx','sex_male',\

                               'anatom_site_general_challenge_lower extremity',\

                               'anatom_site_general_challenge_oral/genital',\

                               'anatom_site_general_challenge_palms/soles',\

                               'anatom_site_general_challenge_torso',\

                               'anatom_site_general_challenge_upper extremity']]



        train_data=train_data.to_numpy()

        self.meta_data=meta_data

        # Dataset shuffling happens here

        train_data = shuffle(train_data, random_state=0)

        

        # Test on a smaller dataset

        if DEBUG: train_data = train_data[:10]

            

        self.filenames = [i[8] for i in train_data]

        

        self.features = (meta_data.to_numpy()).flatten()

        

        self.labels = [i[7] for i in train_data]

        

        # Length of dataset

        self.length = len(self.filenames)

    

    def parser(self,filename,label):

        image_decoded = tensorflow.image.decode_jpeg(tensorflow.io.read_file(filename))

        image_decoded = tf.image.resize(image_decoded,size=[224,224])

        rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

        

        #features = self.meta_data.loc[self.meta_data['path']==filename]

        #features.drop(columns='path',inplace=True)

        #features = features.to_numpy()

        #features = features.flatten()



        return rgb,label

    

    def img_resize(self,img,resolution=224):

        from skimage.transform import resize

        return resize(img,(resolution,resolution),preserve_range=True, mode='reflect', anti_aliasing=True)

    

    #Tying the parser function in the generator

    def generator():

        for i in np.random.permutation(len(self.filenames)):

            rgb,features,label = self.parser(self.filenames[i],self.label[i])

            yield {"input1":rgb,"input2":features},label

            

    

    def get_batch_data(self,batch_size):

        self.dataset = tf.data.Dataset.from_tensor_slices((self.filenames,self.labels))

        self.dataset = self.dataset.shuffle(buffer_size=len(self.filenames),reshuffle_each_iteration=True)

        self.dataset = self.dataset.repeat()

        self.dataset = self.dataset.map(map_func=self.parser,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        self.dataset = self.dataset.batch(batch_size=batch_size)

        

        return self.dataset

        

        

    def get_numpy_data(self,batch_size):

        images_appended = []

        meta_data_appended = []

        labels_appended = []

        

        for k in range(batch_size):

            i = np.random.choice(len(self.filenames))

            rgb,features,label = self.parser(self.filenames[i],self.labels[i])

            images_appended.append(rgb)

            meta_data_appended.append(features)

            labels_appended.append(label)

            

        return np.array(images_appended),np.array(meta_data_appended),np.array(labels_appended)

            
batch_size=32

dl = DataLoader()

train_generator = dl.get_batch_data(batch_size)



print('Data loader ready.')
#Define Model



model = Classifier()
optimizer = tensorflow.keras.optimizers.Adam(lr=0.001, amsgrad=True)



model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=optimizer)
import os

checkpoint_path = "cp.ckpt"

checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)
model.fit(train_generator, epochs=15, steps_per_epoch=dl.length//batch_size, callbacks=[cp_callback])
from skimage import io

from skimage.transform import resize

def get_prediction(image_name,model=model):

    filename = '../input/siim-isic-melanoma-classification/jpeg/test/'+image_name+'.jpg'

    image_decoded = tensorflow.image.decode_jpeg(tensorflow.io.read_file(filename))

    image_decoded = tf.image.resize(image_decoded,size=[224,224])

    rgb = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32)

    rgb = np.expand_dims(rgb, axis=0)

    return model.predict(rgb)
Submissions = pd.read_csv('../input/siim-isic-melanoma-classification/sample_submission.csv')
Submissions['target']=Submissions.apply(lambda x: get_prediction(x['image_name']).flatten(), axis=1)

Submissions['target']=Submissions['target'].astype(float)
Submissions.to_csv('submission.csv',index=False)