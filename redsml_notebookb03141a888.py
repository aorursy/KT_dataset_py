%matplotlib inline



import matplotlib

import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

from PIL import Image
# create a map for converting species names to numbers

# listed in the order needed for output

i = 0

species = {}

speciesList = ['Acer_Capillipes','Acer_Circinatum','Acer_Mono','Acer_Opalus','Acer_Palmatum','Acer_Pictum','Acer_Platanoids','Acer_Rubrum','Acer_Rufinerve','Acer_Saccharinum','Alnus_Cordata','Alnus_Maximowiczii','Alnus_Rubra','Alnus_Sieboldiana','Alnus_Viridis','Arundinaria_Simonii','Betula_Austrosinensis','Betula_Pendula','Callicarpa_Bodinieri','Castanea_Sativa','Celtis_Koraiensis','Cercis_Siliquastrum','Cornus_Chinensis','Cornus_Controversa','Cornus_Macrophylla','Cotinus_Coggygria','Crataegus_Monogyna','Cytisus_Battandieri','Eucalyptus_Glaucescens','Eucalyptus_Neglecta','Eucalyptus_Urnigera','Fagus_Sylvatica','Ginkgo_Biloba','Ilex_Aquifolium','Ilex_Cornuta','Liquidambar_Styraciflua','Liriodendron_Tulipifera','Lithocarpus_Cleistocarpus','Lithocarpus_Edulis','Magnolia_Heptapeta','Magnolia_Salicifolia','Morus_Nigra','Olea_Europaea','Phildelphus','Populus_Adenopoda','Populus_Grandidentata','Populus_Nigra','Prunus_Avium','Prunus_X_Shmittii','Pterocarya_Stenoptera','Quercus_Afares','Quercus_Agrifolia','Quercus_Alnifolia','Quercus_Brantii','Quercus_Canariensis','Quercus_Castaneifolia','Quercus_Cerris','Quercus_Chrysolepis','Quercus_Coccifera','Quercus_Coccinea','Quercus_Crassifolia','Quercus_Crassipes','Quercus_Dolicholepis','Quercus_Ellipsoidalis','Quercus_Greggii','Quercus_Hartwissiana','Quercus_Ilex','Quercus_Imbricaria','Quercus_Infectoria_sub','Quercus_Kewensis','Quercus_Nigra','Quercus_Palustris','Quercus_Phellos','Quercus_Phillyraeoides','Quercus_Pontica','Quercus_Pubescens','Quercus_Pyrenaica','Quercus_Rhysophylla','Quercus_Rubra','Quercus_Semecarpifolia','Quercus_Shumardii','Quercus_Suber','Quercus_Texana','Quercus_Trojana','Quercus_Variabilis','Quercus_Vulcanica','Quercus_x_Hispanica','Quercus_x_Turneri','Rhododendron_x_Russellianum','Salix_Fragilis','Salix_Intergra','Sorbus_Aria','Tilia_Oliveri','Tilia_Platyphyllos','Tilia_Tomentosa','Ulmus_Bergmanniana','Viburnum_Tinus','Viburnum_x_Rhytidophylloides','Zelkova_Serrata']

for s in speciesList:

    species[s] = i

    i = i + 1
def normalizeForNN ( data ):

    means = np.mean(data, axis=0)

    stds = np.std(data, axis=0)

    data = (data - means)/stds

    return data/10


# load training and test data.

# in trainging data convert species name to numeric

d = np.loadtxt('../input/train.csv', delimiter=',',skiprows=1,converters={1:lambda s:species[s.decode("utf-8")]})

t = np.loadtxt('../input/test.csv', delimiter=',',skiprows=1)

#np.random.shuffle(d)



ids = d[:,0].astype('int')

d = np.delete (d, 0, 1)  # remove row ids
# seperate labels and data

labels = d[:,0:1].astype('int')

data = normalizeForNN ( d[:,1:] )
imgWH = 32

imgsList = []

for i in ids:

    img = Image.open('../input/images/{}.jpg'.format(i))

    img = img.resize((imgWH,imgWH))

    imgsList.append ( (np.array(img).reshape(imgWH,imgWH,1)))



imgData = np.array ( imgsList )

plt.imshow(imgData[20].reshape(imgWH,imgWH))

m = np.mean(imgData)

s = np.std(imgData)

imgData = (imgData - m)/s/10

[np.mean(imgData), np.std(imgData)]
np.where (labels == 2 )[0]
plt.imshow(imgData[558].reshape(imgWH,imgWH))
from keras.utils import np_utils



train_data = data

train_labels = np_utils.to_categorical(labels)



num_classes = train_labels.shape[1]
# margin, shape, texture: use a different nn for each data type

slen = 64

train_margin_data = train_data[:,slen*0:slen*1]

train_shape_data = train_data[:,slen*1:slen*2]

train_texture_data = train_data[:,slen*2:slen*3]
from keras.models import Sequential

from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Merge

from keras.optimizers import SGD



activ = 'relu'

ninput = 64

nnodes = 40

dropout = 0



train_margin_data = train_margin_data.reshape ( -1, 8,8, 1)

m1 = Sequential([

        Convolution2D(16,4,4, input_shape=(8,8,1), activation='relu'),

        MaxPooling2D(pool_size=(2,2)),

        Flatten(),

#        Dropout(.2)

        ])



train_shape_data = train_shape_data.reshape ( -1, 8,8, 1)

m2 = Sequential([

        Convolution2D(16,4,4, input_shape=(8,8, 1), activation='relu'),

        MaxPooling2D(pool_size=(2,2)),

        Flatten(),

#        Dropout(.2)

        ])



m3 = Sequential([

        Dense(nnodes, input_dim=ninput, activation=activ),

        Dropout(dropout)

        ])



mimg = Sequential ([

    Convolution2D(32,3,3, input_shape=(imgWH,imgWH,1), activation='relu'),

    MaxPooling2D(pool_size=(2,2)),

    Dropout(.2),

    Flatten(),

    Dense(128, activation='relu')

])



m = Merge ( [m1, m2, m3, mimg], mode='concat')



#model = model1(len(train_data[0]), num_classes)

model = Sequential([

        mimg,

#        Dropout(.2),

        Dense(num_classes, activation='softmax')

    ])



model = Sequential([

    Dense(128,input_dim=imgWH*imgWH, activation='relu'),

    Dense(num_classes, activation='softmax')

])



model = Sequential()

model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(imgWH, imgWH,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(num_classes))

model.add(Activation('softmax'))

    

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])



#res = model.fit([train_margin_data, train_shape_data, train_texture_data, imgData], train_labels,

res = model.fit(imgData, train_labels,

          nb_epoch=20,

          batch_size=200,

          validation_split=.3,

          verbose=2)

[res.history['acc'][-1],res.history['val_acc'][-1]]
pred = model.predict(imgData)

imn = 7

np.argmax(pred,axis=1)

ls = labels.reshape(-1)

np.mean(ls == np.argmax(pred, axis=1))

plt.plot (res.history['val_acc'])

plt.plot( res.history['acc'])
# don't know about this

def logloss(labels, pred):

    pred = np.where ( pred == 0, 1e-18, pred)

    return -np.sum(labels * np.log(pred))/len(pred)



llpred = model.predict_proba([train_margin_data, train_shape_data, train_texture_data], batch_size=32)

logloss(train_labels, llpred)
# prepare test data

tids = t[:,0].astype('int')

tdata = normalizeForNN(t[:,1:])



pred = model.predict ( tdata )
sl = np.insert ( speciesList, 0, 'id')

header = ','.join (sl)

res = np.insert(pred,0,tids,axis=1)



savetxt("pred.csv", res, delimiter=',', header=header)

p2 = loadtxt("pred.csv", delimiter=',', skiprows=1)
model.evaluate(train_data, train_labels, batch_size=16, verbose=0)

model.evaluate(val_data, val_labels, batch_size=16, verbose=0)
# look at images 

cval_data = val_data.reshape(-1, 3, 64,1)

ctrain_data = train_data.reshape(-1,3, 64,1)

def model3(ninput, nlabels): 

    model = Sequential()

    model.add(Convolution2D(32, 3, 32, border_mode='valid', input_shape=(3,64,1)))

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(1, 3)))

    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(nlabels))

    model.add(Activation('softmax'))

    return model



cmodel = model3(0, num_classes)

cmodel.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

#cmodel.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

res = cmodel.fit(ctrain_data, train_labels,

          nb_epoch=2,

          batch_size=200,

          verbose=2)

cmodel.evaluate(cval_data, val_labels, batch_size=16, verbose=0)
import os

l = []

for dirname, dirnames, filenames in os.walk('..'):

    l.append([dirname, dirnames])

    

l
i1 = mpimg.imread ( "../input/images/1.jpg")

#plt.imshow(a)

a = train_shape_data.reshape(-1,8,8)

b = train_margin_data.reshape(-1,8,8)

c = train_texture_data.reshape(-1,8,8)

np.shape(a[0])

for i in range (5):

    plt.imshow(a[i])

    plt.imshow(b[i])

    plt.imshow(c[i])

    

np.shape(i1)
img = Image.open('../input/images/50.jpg')

img = img.resize((128,128), Image.LANCZOS)

img

#In [18]: img.thumbnail((64, 64), Image.ANTIALIAS) # resizes image in-place

#In [19]: imgplot = plt.imshow(img)

#i1 = mpimg.imread ( "../input/images/81.jpg")

np.shape(img)