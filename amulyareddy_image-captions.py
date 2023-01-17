import numpy as np

import pandas as pd

import cv2,os,glob

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split



# Flickr image captions data from kaggle (https://www.kaggle.com/hsankesara/flickr-image-dataset)



path = '../input/flickr-image-dataset/flickr30k_images/flickr30k_images/flickr30k_images/*'

y_data,X_data,X_labels = [],[],[]

for img in glob.glob(path):

    if '.csv' not in img:

        image = cv2.resize(cv2.imread(img,cv2.IMREAD_GRAYSCALE), (200, 200))

        X_data.append([np.array(image),img.split('/')[-1]])

        X_labels.append(img.split('/')[-1])
y_data = pd.read_csv('../input/flickr-image-dataset/flickr30k_images/flickr30k_images/results.csv', delimiter='|')
y_data.head()
y_data[y_data['image_name']==X_data[0][1]]
y_corresponding = []

for each in X_labels:

    captions = np.array(y_data[y_data['image_name']==each][' comment'])

    y_corresponding.append(captions)

    

print(X_labels[0])

print(y_corresponding[0]) #y
import re

def clean(text):

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split()).lower()
y_clean = []

for each in y_corresponding:

    y_each = []

    for text in each:

        try:

            y_each.append(clean(text))

        except Exception as e:

            print(e)

            pass

    y_clean.append(y_each)
y_clean[0]
plt.imshow(X_data[0][0]) #X
df = pd.DataFrame(y_data)

vocab = np.array(df[' comment'])

print(len(vocab))

vocab = [str(each) for each in vocab] # one array of all sentences
clean_text = clean(' '.join(str(v) for v in vocab))

print(clean_text[:100]) # one continuous string of all words

print(len(set(clean_text.split(' '))))
from keras.applications import VGG16

from keras import models



modelvgg = VGG16(weights='imagenet',include_top=True)

modelvgg.layers.pop() # remove last layer of predictions

modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)

modelvgg.summary()
modelvgg.compile(loss='categorical_crossentropy', optimizer='adam')
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(num_words=18300) # max num of words

tokenizer.fit_on_texts(vocab)
yfit_data = []

yfit_data = [tokenizer.texts_to_sequences(each) for each in y_clean]



print(y_clean[0])

print(yfit_data[0])
print(len(yfit_data),len(X_data))
X_train,X_test,y_train,y_test = train_test_split(X_data, yfit_data, test_size=0.2, random_state=0)
# hist = modelvgg.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=2, verbose=2, batch_size=64)