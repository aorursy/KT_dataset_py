import os

import pydicom

import numpy as np

import pandas as pd

from skimage import measure

from skimage.transform import resize



import tensorflow as tf

from tensorflow import keras
train_labels = pd.read_csv('../input/stage_2_train_labels.csv')

print(train_labels.head())

print(pd.DataFrame(os.listdir('../input/stage_2_train_images/')).head())
pneum_locs = {}

for x in range(0,len(train_labels)):

    row = train_labels.iloc[x]

    patientId = row[0]

    loc = row[1:3]



    if row[4] == '1': #patient has pneumonia

        location = [int(float(i))for i in loc] #data in labels is in string form, must convert

        if(patientId in pneum_locs): #patient already in dictionary

            pneum_locs[patientId].append(loc)

        else:

            pneum_locs[patientId] = [loc]
class datagenerator(keras.utils.Sequence):

    def __init__(self,patientIds, p_locations = None, batch_size = 32,image_size = 256):

        self.patientIds = patientIds

        self.p_locations = p_locations

        self.batch_size = batch_size

        self.image_size = image_size

      

    def __load__(self,pID):#loads a file for retrieval 

        image = pydicom.read_file('../input/stage_2_train_images/%s.dcm' % pID).pixel_array

        mask = np.zeros(image.shape)

        image = resize(image,(self.image_size,self.image_size),mode = 'reflect')

        mask = resize(mask, (self.image_size, self.image_size), mode='reflect') > 0.5

        if pID in self.p_locations :

            for loc in self.p_locations[patID]:

                x,y,w,h = loc

                mask[y:y+h,x:x+w] = 1

        image = np.expand_dims(image,-1) #X = image

        mask = np.expand_dims(mask,-1) #Y = mask



        return image,mask

    def __getitem__(self,index):#mandatory inheritance

        pIDs = self.patientIds[index*self.batch_size:(index + 1)*self.batch_size]

        images,masks = zip(*[self.__load__(patientId) for patientId in pIDs])

        images = np.array(images)

        masks = np.array(masks)

        return images,masks

    def __len__(self): #mandatory inheritance

        return int(len(self.patientIds)/self.batch_size)



        
def create_downsample(channels, inputs):

    x = keras.layers.BatchNormalization()(inputs)

    x = keras.layers.LeakyReLU(0)(x)#LeakyReLU with alpha = 0 is identical to ReLU

    x = keras.layers.Conv2D(channels, 1, padding='same')(x)

    x = keras.layers.MaxPool2D(2)(x)

    return x



def create_resblock(channels, inputs):

    x = keras.layers.BatchNormalization()(inputs)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(channels, 3, padding='same')(x)



    return keras.layers.add([x, inputs])



def create_network(input_size, channels, n_blocks=2, depth=4): #creates a residual block layer

    inputs = keras.Input(shape=(input_size, input_size, 1))

    x = keras.layers.Conv2D(channels, 3, padding='same')(inputs)

    for d in range(depth):

        channels = channels * 2

        x = create_downsample(channels, x)

        for b in range(n_blocks):

            x = create_resblock(channels, x)

    # output

    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.LeakyReLU(0)(x)

    x = keras.layers.Conv2D(1, 1, activation='sigmoid')(x)

    outputs = keras.layers.UpSampling2D(2**depth)(x) ##upsample data to counteract the first downsample

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
def iou_loss(y_true, y_pred):

    y_true = tf.reshape(y_true, [-1])

    y_pred = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true * y_pred)

    score = (intersection + 1.) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection + 1.)

    return 1 - score









model = create_network(input_size=256, channels=32, n_blocks=2, depth=4)

model.compile(optimizer='adam',

              loss=iou_loss,

              metrics=['accuracy'])

val_size = 3000 #about 10% of data used for validation



train_IDs = train_labels['patientId'][val_size:]

val_IDs= train_labels['patientId'][:val_size]

validgen = datagenerator(val_IDs,pneum_locs,batch_size = 1,image_size = 256)

traingen = datagenerator(train_IDs, pneum_locs, batch_size=1, image_size=256)

hist = model.fit_generator(traingen,epochs = 3,validation_data = validgen, workers = 4,use_multiprocessing = True)


test_IDs = pd.DataFrame(os.listdir('../input/stage_2_test_images/'))

for i in range (0, len(test_IDs)):

    test_IDs[0][i]=test_IDs[0][i].split('.')[0]

test_IDs = test_IDs[0]



submission = {}

for pID in test_IDs :

    image = pydicom.read_file('../input/stage_2_test_images/%s.dcm' % pID).pixel_array

    image = resize(image,(256,256),mode = 'reflect')

    image = np.expand_dims(image,-1)

    images = np.zeros((1,256,256,1))

    images[0] = image

    pred = model.predict(images)

    predict = resize(np.squeeze(pred),(1024,1024), mode = 'reflect')

    compute = predict[:,:] >0.5 #transforms values to 1s and 0s

    compute = measure.label(compute)

    predString = ''

    for region in measure.regionprops(compute):

        y,x,y2,x2 = region.bbox

        confidence = np.mean(predict[y:y2,x:x2])

        predictionString += str(conf) + ' ' + str(x) + ' ' + str(y) + ' ' + str(y2-y) + ' ' + str(x2 - x) + ' '

    submission[pID]= predString

    if(len(submission) >= len(test_IDs)): #loop exit control

        break



submit = pd.DataFrame.from_dict(submission,orient ='index')

print("%s predictions recorded." % len(submit))

submit.index.names = ['patientId']

submit.columns = ['PredictionString']

submit.to_csv('submission.csv')
