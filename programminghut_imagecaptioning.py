import numpy as np

import pandas as pd

import cv2

import os

from glob import glob
images_path = '../input/flickr8k-sau/Flickr_Data/Images/'

images = glob(images_path+'*.jpg')

len(images)
images[:5]
import matplotlib.pyplot as plt



for i in range(5):

    plt.figure()

    img = cv2.imread(images[i])

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)
from keras.applications import ResNet50



incept_model = ResNet50(include_top=True)
from keras.models import Model

last = incept_model.layers[-2].output

modele = Model(inputs = incept_model.input,outputs = last)

modele.summary()
images_features = {}

count = 0

for i in images:

    img = cv2.imread(i)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (224,224))

    

    img = img.reshape(1,224,224,3)

    pred = modele.predict(img).reshape(2048,)

        

    img_name = i.split('/')[-1]

    

    images_features[img_name] = pred

    

    count += 1

    

    if count > 1499:

        break

        

    elif count % 50 == 0:

        print(count)

    

        

    
len(images_features)
caption_path = '../input/flickr8k-sau/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
captions = open(caption_path, 'rb').read().decode('utf-8').split('\n')
len(captions)
captions_dict = {}

for i in captions:

    try:

        img_name = i.split('\t')[0][:-2] 

        caption = i.split('\t')[1]

        if img_name in images_features:

            if img_name not in captions_dict:

                captions_dict[img_name] = [caption]

                

            else:

                captions_dict[img_name].append(caption)

            

    except:

        pass
len(captions_dict)
import matplotlib.pyplot as plt



for i in range(5):

    plt.figure()

    img_name = images[i]

    

    

    img = cv2.imread(img_name)

    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.xlabel(captions_dict[img_name.split('/')[-1]])

    plt.imshow(img)
import matplotlib.pyplot as plt



for k in images_features.keys():

    plt.figure()

    

    img_name = '../input/flickr8k-sau/Flickr_Data/Images/' + k

    

    

    img = cv2.imread(img_name)

    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.xlabel(captions_dict[img_name.split('/')[-1]])

    plt.imshow(img)

    

    break


def preprocessed(txt):

    modified = txt.lower()

    modified = 'startofseq ' + modified + ' endofseq'

    return modified

    
for k,v in captions_dict.items():

    for vv in v:

        captions_dict[k][v.index(vv)] = preprocessed(vv)
count_words = {}

for k,vv in captions_dict.items():

    for v in vv:

        for word in v.split():

            if word not in count_words:



                count_words[word] = 0



            else:

                count_words[word] += 1
len(count_words)
THRESH = -1

count = 1

new_dict = {}

for k,v in count_words.items():

    if count_words[k] > THRESH:

        new_dict[k] = count

        count += 1

        
len(new_dict)
new_dict['<OUT>'] = len(new_dict) 
captions_backup = captions_dict.copy()
captions_dict = captions_backup.copy()
for k, vv in captions_dict.items():

    for v in vv:

        encoded = []

        for word in v.split():  

            if word not in new_dict:

                encoded.append(new_dict['<OUT>'])

            else:

                encoded.append(new_dict[word])





        captions_dict[k][vv.index(v)] = encoded
captions_dict
from keras.utils import to_categorical

from keras.preprocessing.sequence import pad_sequences
MAX_LEN = 0

for k, vv in captions_dict.items():

    for v in vv:

        if len(v) > MAX_LEN:

            MAX_LEN = len(v)

            print(v)
MAX_LEN
captions_dict
Batch_size = 5000

VOCAB_SIZE = len(new_dict)



def generator(photo, caption):

    n_samples = 0

    

    X = []

    y_in = []

    y_out = []

    

    for k, vv in caption.items():

        for v in vv:

            for i in range(1, len(v)):

                X.append(photo[k])



                in_seq= [v[:i]]

                out_seq = v[i]



                in_seq = pad_sequences(in_seq, maxlen=MAX_LEN, padding='post', truncating='post')[0]

                out_seq = to_categorical([out_seq], num_classes=VOCAB_SIZE)[0]



                y_in.append(in_seq)

                y_out.append(out_seq)

            

    return X, y_in, y_out

    

    
X, y_in, y_out = generator(images_features, captions_dict)
len(X), len(y_in), len(y_out)
X = np.array(X)

y_in = np.array(y_in, dtype='float64')

y_out = np.array(y_out, dtype='float64')



X.shape, y_in.shape, y_out.shape
X[1510]
y_in[2]


from keras.preprocessing.sequence import pad_sequences

from keras.utils import to_categorical

from keras.utils import plot_model

from keras.models import Model, Sequential

from keras.layers import Input

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Embedding

from keras.layers import Dropout

from keras.layers.merge import add

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate

from keras.models import Sequential, Model
embedding_size = 128

max_len = MAX_LEN

vocab_size = len(new_dict)



image_model = Sequential()



image_model.add(Dense(embedding_size, input_shape=(2048,), activation='relu'))

image_model.add(RepeatVector(max_len))



image_model.summary()



language_model = Sequential()



language_model.add(Embedding(input_dim=vocab_size, output_dim=embedding_size, input_length=max_len))

language_model.add(LSTM(256, return_sequences=True))

language_model.add(TimeDistributed(Dense(embedding_size)))



language_model.summary()



conca = Concatenate()([image_model.output, language_model.output])

x = LSTM(128, return_sequences=True)(conca)

x = LSTM(512, return_sequences=False)(x)

x = Dense(vocab_size)(x)

out = Activation('softmax')(x)

model = Model(inputs=[image_model.input, language_model.input], outputs = out)



# model.load_weights("../input/model_weights.h5")

model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

model.summary()
model.fit([X, y_in], y_out, batch_size=512, epochs=50)
inv_dict = {v:k for k, v in new_dict.items()}
model.save('model.h5')
model.save_weights('mine_model_weights.h5')
np.save('vocab.npy', new_dict)
def getImage(x):

    

    test_img_path = images[x]



    test_img = cv2.imread(test_img_path)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)



    test_img = cv2.resize(test_img, (299,299))



    test_img = np.reshape(test_img, (1,299,299,3))

    

    return test_img
for i in range(5):

    

    no = np.random.randint(1500,7000,(1,1))[0,0]

    test_feature = modele.predict(getImage(no)).reshape(1,2048)

    

    test_img_path = images[no]

    test_img = cv2.imread(test_img_path)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)





    text_inp = ['startofseq']



    count = 0

    caption = ''

    while count < 25:

        count += 1



        encoded = []

        for i in text_inp:

            encoded.append(new_dict[i])



        encoded = [encoded]



        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=MAX_LEN)





        prediction = np.argmax(model.predict([test_feature, encoded]))



        sampled_word = inv_dict[prediction]



        caption = caption + ' ' + sampled_word

            

        if sampled_word == 'endofseq':

            break



        text_inp.append(sampled_word)

        

    plt.figure()

    plt.imshow(test_img)

    plt.xlabel(caption)