import os

from os.path import join



import numpy as np

import pandas as pd



import tensorflow as tf

import keras

from keras import layers, Input, models

from keras.utils import to_categorical

from keras.wrappers.scikit_learn import KerasClassifier 

from sklearn.model_selection import KFold 

from sklearn.model_selection import cross_val_score



from sklearn.model_selection import train_test_split



import matplotlib.pyplot as plt



datapath = join('data', 'wafer')



print(os.listdir("../input"))

import warnings

warnings.filterwarnings("ignore")
df=pd.read_pickle("../input/LSWMD.pkl")

df.info()
df.head()
df.tail()
import matplotlib.pyplot as plt

%matplotlib inline





uni_Index=np.unique(df.waferIndex, return_counts=True)

plt.bar(uni_Index[0],uni_Index[1], color='gold', align='center', alpha=0.5)

plt.title(" wafer Index distribution")

plt.xlabel("index #")

plt.ylabel("frequency")

plt.xlim(0,26)

plt.ylim(30000,34000)

plt.show()
df = df.drop(['waferIndex'], axis = 1)
def find_dim(x):

    dim0=np.size(x,axis=0)

    dim1=np.size(x,axis=1)

    return dim0,dim1

df['waferMapDim']=df.waferMap.apply(find_dim)

df.sample(5)
sub_df = df.loc[df['waferMapDim'] == (26, 26)]

sub_wafer = sub_df['waferMap'].values



sw = np.ones((1, 26, 26))

label = list()



for i in range(len(sub_df)):

    # skip null label

    if len(sub_df.iloc[i,:]['failureType']) == 0:

        continue

    sw = np.concatenate((sw, sub_df.iloc[i,:]['waferMap'].reshape(1, 26, 26)))

    label.append(sub_df.iloc[i,:]['failureType'][0][0])
x = sw[1:]

y = np.array(label).reshape((-1,1))
# check dimension

print('x shape : {}, y shape : {}'.format(x.shape, y.shape))
# plot 1st data

plt.imshow(x[0])

plt.show()



# check faulty case

print('Faulty case : {} '.format(y[0]))
#add channel

x = x.reshape((-1, 26, 26, 1))
faulty_case = np.unique(y)

print('Faulty case list : {}'.format(faulty_case))
for f in faulty_case :

    print('{} : {}'.format(f, len(y[y==f])))
# One-hot-Encoding faulty categorical variable as channel

new_x = np.zeros((len(x), 26, 26, 3))



for w in range(len(x)):

    for i in range(26):

        for j in range(26):

            new_x[w, i, j, int(x[w, i, j])] = 1
#check new x dimension

new_x.shape
# parameter

epoch=15

batch_size=1024
# Encoder

input_shape = (26, 26, 3)

input_tensor = Input(input_shape)

encode = layers.Conv2D(64, (3,3), padding='same', activation='relu')(input_tensor)



latent_vector = layers.MaxPool2D()(encode)



# Decoder

decode_layer_1 = layers.Conv2DTranspose(64, (3,3), padding='same', activation='relu')

decode_layer_2 = layers.UpSampling2D()

output_tensor = layers.Conv2DTranspose(3, (3,3), padding='same', activation='sigmoid')



# connect decoder layers

decode = decode_layer_1(latent_vector)

decode = decode_layer_2(decode)



ae = models.Model(input_tensor, output_tensor(decode))

ae.compile(optimizer = 'Adam',

              loss = 'mse',

             )
ae.summary()
# start train

ae.fit(new_x, new_x,

       batch_size=batch_size,

       epochs=epoch,

       verbose=2)
# Make encoder model with part of autoencoder model layers

encoder = models.Model(input_tensor, latent_vector)
# Make decoder model with part of autoencoder model layers

decoder_input = Input((13, 13, 64))

decode = decode_layer_1(decoder_input)

decode = decode_layer_2(decode)



decoder = models.Model(decoder_input, output_tensor(decode))
# Encode original faulty wafer

encoded_x = encoder.predict(new_x)
# Add noise to encoded latent faulty wafers vector.

noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64))
# check original faulty wafer data

plt.imshow(np.argmax(new_x[3], axis=2))
# check new noised faulty wafer data

noised_gen_x = np.argmax(decoder.predict(noised_encoded_x), axis=3)

plt.imshow(noised_gen_x[3])
# check reconstructed original faulty wafer data

gen_x = np.argmax(ae.predict(new_x), axis=3)

plt.imshow(gen_x[3])
# augment function define

def gen_data(wafer, label):

    # Encode input wafer

    encoded_x = encoder.predict(wafer)

    

    # dummy array for collecting noised wafer

    gen_x = np.zeros((1, 26, 26, 3))

    

    # Make wafer until total # of wafer to 2000

    for i in range((2000//len(wafer)) + 1):

        noised_encoded_x = encoded_x + np.random.normal(loc=0, scale=0.1, size = (len(encoded_x), 13, 13, 64)) 

        noised_gen_x = decoder.predict(noised_encoded_x)

        gen_x = np.concatenate((gen_x, noised_gen_x), axis=0)

    # also make label vector with same length

    gen_y = np.full((len(gen_x), 1), label)

    

    # return date without 1st dummy data.

    return gen_x[1:], gen_y[1:]
# Augmentation for all faulty case.

for f in faulty_case : 

    # skip none case

    if f == 'none' : 

        continue

    

    gen_x, gen_y = gen_data(new_x[np.where(y==f)[0]], f)

    new_x = np.concatenate((new_x, gen_x), axis=0)

    y = np.concatenate((y, gen_y))
print('After Generate new_x shape : {}, new_y shape : {}'.format(new_x.shape, y.shape))
for f in faulty_case :

    print('{} : {}'.format(f, len(y[y==f])))
# choice index without replace.

none_idx = np.where(y=='none')[0][np.random.choice(len(np.where(y=='none')[0]), size=11000, replace=False)]
# delete choiced index data.

new_x = np.delete(new_x, none_idx, axis=0)

new_y = np.delete(y, none_idx, axis=0)
print('After Delete "none" class new_x shape : {}, new_y shape : {}'.format(new_x.shape, new_y.shape))
for f in faulty_case :

    print('{} : {}'.format(f, len(new_y[new_y==f])))
# make string label data to numerical data

for i, l in enumerate(faulty_case):

    new_y[new_y==l] = i

    

# one-hot-encoding

new_y = to_categorical(new_y)
# split data train, test

x_train, x_test, y_train, y_test = train_test_split(new_x, new_y,

                                                    test_size=0.33,

                                                    random_state=2019)
print('Train x : {}, y : {}'.format(x_train.shape, y_train.shape))

print('Test x: {}, y : {}'.format(x_test.shape, y_test.shape))
def create_model():

    input_shape = (26, 26, 3)

    input_tensor = Input(input_shape)



    conv_1 = layers.Conv2D(16, (3,3), activation='relu', padding='same')(input_tensor)

    conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)

    conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_2)



    flat = layers.Flatten()(conv_3)



    dense_1 = layers.Dense(512, activation='relu')(flat)

    dense_2 = layers.Dense(128, activation='relu')(dense_1)

    output_tensor = layers.Dense(9, activation='softmax')(dense_2)



    model = models.Model(input_tensor, output_tensor)

    model.compile(optimizer='Adam',

                 loss='categorical_crossentropy',

                 metrics=['accuracy'])



    return model
# Make keras model to sklearn classifier.

model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1024, verbose=2) 

# 3-Fold Crossvalidation

kfold = KFold(n_splits=3, shuffle=True, random_state=2019) 

results = cross_val_score(model, x_train, y_train, cv=kfold)

# Check 3-fold model's mean accuracy

print('Simple CNN Cross validation score : {:.4f}'.format(np.mean(results)))
history = model.fit(x_train, y_train,

         validation_data=[x_test, y_test],

         epochs=epoch,

         batch_size=batch_size,

         )
# accuracy plot 

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



# loss plot

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()