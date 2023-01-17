import pandas as pd
import pickle
import numpy as np
import os
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
from keras.layers import Dense, Flatten,Input, Convolution2D, Dropout, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector,Concatenate
from keras.models import Sequential, Model
from keras.utils import np_utils
import random
from keras.preprocessing import image, sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.applications.inception_v3 import InceptionV3
images_dir = os.listdir("../input/flickr8k/flickr_data/Flickr_Data/")
images_path = '../input/flickr8k/flickr_data/Flickr_Data/Images/'
captions_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr8k.token.txt'
train_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.trainImages.txt'
val_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.devImages.txt'
test_path = '../input/flickr8k/flickr_data/Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt'

captions = open(captions_path, 'r').read().split("\n")
x_train = open(train_path, 'r').read().split("\n")
x_val = open(val_path, 'r').read().split("\n")
x_test = open(test_path, 'r').read().split("\n")
len(x_train)
print(len(captions))
captions[:3]
tokens = {}

for ix in range(len(captions)-1):
    temp = captions[ix].split("#")
    if temp[0] in tokens:
        tokens[temp[0]].append(temp[1][2:])
    else:
        tokens[temp[0]] = [temp[1][2:]]
from IPython.display import Image, display
temp = captions[100].split("#")
z = Image(filename=images_path+temp[0])
display(z)

for ix in range(len(tokens[temp[0]])):
    print(tokens[temp[0]][ix])
x_imgs=[]
captions=[]
for img in x_train:
    if img == '':
        continue
    for capt in tokens[img]:
        caption = "<start> "+ capt + " <end>"
        x_imgs.append(img)
        captions.append(caption)
data={'images':x_imgs,'captions':captions}
df=pd.DataFrame(data)
df.head()
words = [i.split() for i in captions]
unique = []
for i in words:
    unique.extend(i)
unique = list(set(unique))
print(len(unique))
vocab_size = len(unique)
word_2_indices = {val:index for index, val in enumerate(unique)}
indices_2_word = {index:val for index, val in enumerate(unique)}
word_2_indices['UNK'] = 0
indices_2_word[0] = 'UNK'
with open( "w2i.p", "wb" ) as pickle_f:
    pickle.dump(word_2_indices, pickle_f )
with open( "i2w.p", "wb" ) as pickle_f:
    pickle.dump(indices_2_word, pickle_f )
with open('../input/img-cap-w2i-i2w/w2i (2).p', 'rb') as f:
    word_2_indices= pickle.load(f, encoding="bytes")
with open('../input/img-cap-w2i-i2w/i2w (2).p', 'rb') as f:
    indices_2_word= pickle.load(f, encoding="bytes")
vocab_size = len(word_2_indices.keys())
print(vocab_size)
max_len = 0
for i in captions:
    i = i.split()
    if len(i) > max_len:
        max_len = len(i)

print(max_len)
model = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
#model.summary()
def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im
imgs={}
for i in tqdm(x_train[:6000]):
    if i in imgs.keys():
        continue
    path = images_path + i
    img = preprocessing(path)
    pred = model.predict(img).reshape(2048)
    imgs[i]=pred
len(imgs)
with open( "proces_imgs_icep.p", "wb" ) as pickle_f:
    pickle.dump(imgs, pickle_f )
with open('../input/img-cap-63-acc-model/proces_imgs.p', 'rb') as f:
    imgs= pickle.load(f, encoding="bytes")
img = []
for j in tqdm(range(df.shape[0])):
    if df.iloc[j, 0] in imgs.keys():
        img.append(imgs[df.iloc[j, 0]])

img = np.asarray(img)
print(img.shape)
embedding_size = 128
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
model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model.summary()
from keras.utils import to_categorical
def tr_gen(inp_imgs,captions,batch_photos=256):
    x1, x2, y = list(), list(), list()
    n=0
    while True:
        for i in range(df.shape[0]):
            n+=1
            photo=inp_imgs[i]
            text = captions[i].split()
            text = [word_2_indices[i] for i in text]
            for j in range(1, len(text)):
                partial_seqs=text[:j]
                next_words=text[j]
                padded_partial_seqs = sequence.pad_sequences([partial_seqs], 40, padding='post')[0]
                next_words = to_categorical([next_words], num_classes=vocab_size)[0]
                x1.append(photo)
                x2.append(padded_partial_seqs)
                y.append(next_words)
            if n==batch_size:
                yield [np.array(x1), np.array(x2)], np.array(y)
                x1, x2, y = list(), list(), list()
                n=0
train_gen=tr_gen(img,captions,batch_photos=256)
from tensorflow.keras.callbacks import ModelCheckpoint
mc = ModelCheckpoint('best_model_acc.h5', monitor = 'accuracy' , mode = 'max', verbose = 1 , save_best_only = True)
hist=model.fit_generator(train_gen,steps_per_epoch=117,callbacks = [mc],epochs=100)
def preprocessing(img_path):
    im = image.load_img(img_path, target_size=(224,224,3))
    im = image.img_to_array(im)
    im = np.expand_dims(im, axis=0)
    return im
def get_encoding(model, img):
    image = preprocessing(img)
    pred = model.predict(image).reshape(2048)
    return pred
resnet = ResNet50(include_top=False,weights='imagenet',input_shape=(224,224,3),pooling='avg')
im = '../input/flickr8k/flickr_data/Flickr_Data/Images/'+x_test[121]
test_img = get_encoding(resnet, im)
def predict_captions(image):
    start_word = ["<start>"]
    while True:
        par_caps = [word_2_indices[i] for i in start_word]
        par_caps = sequence.pad_sequences([par_caps], maxlen=max_len, padding='post')
        preds = model.predict([np.array([image]), np.array(par_caps)])
        word_pred = indices_2_word[np.argmax(preds[0])]
        start_word.append(word_pred)
        
        if word_pred == "<end>" or len(start_word) > max_len:
            break
            
    return ' '.join(start_word[1:-1])

Argmax_Search = predict_captions(test_img)
z = Image(filename=im)
display(z)

print(Argmax_Search)