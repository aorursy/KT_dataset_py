# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%env JOBLIB_TEMP_FOLDER=/tmp



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.gridspec as gridspec

from wordcloud import WordCloud, STOPWORDS 



import nltk  

import numpy as np  

import random  

import string

from collections import Counter

import collections



import bs4 as bs  

import urllib.request  

import re

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer



import tensorflow as tf

import tensorflow_hub as hub





import os





import keras

import tensorflow as tf

import tensorflow_hub as hub

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Input, BatchNormalization, Dropout, Concatenate

from tensorflow.keras.models import Model, Sequential

from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.layers import LSTM

from tensorflow.keras.layers import GlobalMaxPool1D

from tensorflow.keras.layers import Bidirectional





plt.style.use('seaborn')

sns.set_style('whitegrid')

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
a=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

submission= pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')



train_data=a.text.values

train_labels=a.target.values

test_data=test.text.values



a.head()

NAcols=a.columns

for col in NAcols:

    if a[col].dtype == "object":

        a[col] = a[col].fillna("None")
no_dis=a[a['target']==0]

yes_dis=a[a['target']==1]
target_count=a.target.value_counts().reset_index()



plt.pie(target_count.target,colors=("silver","crimson"), 

        autopct='%2.1f%%',textprops={'fontsize': 14, 'weight':'bold'})

lp = {'size': 19}

plt.legend(['Not real', 'Real'],prop=lp, loc='best', bbox_to_anchor=(1, 0.5))

plt.title('Target distribution', weight='bold', fontsize=14)

plt.show()
location_count=a.location.value_counts().sort_values(ascending=False).reset_index().head(20)

location_count= location_count.rename(columns={'index': "name"})



#I imported this design from https://bmanohar16.github.io/blog/customizing-bar-plots

# Figure Size

fig, ax = plt.subplots(figsize=(5,6))



# Horizontal Bar Plot

title_cnt=location_count.location.sort_values(ascending=False).reset_index()

n= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='moccasin', edgecolor='black')









# Remove axes splines

for s in ['top','bottom','left','right']:

    ax.spines[s].set_visible(False)



# Remove x,y Ticks

ax.xaxis.set_ticks_position('none')

ax.yaxis.set_ticks_position('none')



# Add padding between axes and labels

ax.xaxis.set_tick_params(pad=5)

ax.yaxis.set_tick_params(pad=10)



# Add x,y gridlines

ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



# Show top values 

ax.invert_yaxis()



# Add Plot Title

ax.set_title('Most frequent keywords',

             loc='center', pad=10, fontsize=16)

ax.set_xlabel('# of tweets', weight='bold')





# Add annotation to bars

for i in ax.patches:

    ax.text(i.get_width()+50, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

r= range(20)

plt.yticks(r, location_count.name, weight='bold')

plt.xticks(weight='bold')





# Show Plot

plt.show()
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15,8))





fig.suptitle('Keywords in real and not real disaster tweets', weight='bold', fontsize=20)

wordcloud1 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(no_dis['keyword']))

WordCloud.generate_from_frequencies

# Generate plot

wordcloud2 = WordCloud(width=600, height=500, background_color='white').generate(' '.join(yes_dis['keyword']))

WordCloud.generate_from_frequencies

# Generate plot



ax1.set_title('Keywords: No disaster tweets', weight='bold', fontsize=15, color='dodgerblue')

# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)

im1 = ax1.imshow(wordcloud1, aspect='auto')



ax2.set_title('Keywords: Disaster tweets', weight='bold', fontsize=15, color='dodgerblue')

im4 = ax2.imshow(wordcloud2, aspect='auto')



# Make space for title

plt.subplots_adjust(top=0.85)

plt.axis('off')

plt.show()
def word_counter(data): 

        df=data['text'].str.lower()

        #Define the stop words that we don't want the Counter to consider

        swords=[ "i'm",'the', 'a', 'to', 'in', 'of', 'and', 'I', 'you', 'he', 'she', 'it', 'is', 'on','-','The', 

        'my', 'with', 'that', 'at','by','it','from', 'be', 'was', 'have', 'are', 'this', 'like',

        'A', 'as', 'just', 'your', 'up', 'but', 'me', 'so', 'not', 'has', 'out', '??', 'will',

        'via','after', 'an', 'about', 'been', 'get', 'or', 'when', 'all', 'no', 'into', 'over', 'In',

        'who', 'we', 'if', 'I', 'can', 'The', 'how', 'them', 'But', 'So', 'Too', 'too', 'did', 'much', 

        'for', 'his', 'her', 'To', "it's", 'You', 'there', 'If', 'what', 'i', '2', 'they', 'had', 'their',

       'one', 'got', 'us', 'man', 'our', 'two', '&amp;', 'were', 'than']

        

        #Count the words in the tweets except the stop words

        b= collections.Counter([y for x in df.values.flatten() for y in x.split() if y not in swords])

        #Extract top 20 words

        ca= b.most_common(20)

        c=pd.DataFrame(ca)

        c= c.rename(columns={0: "word", 1: "n_word"})  



        

        #Plot the most frequent words

        fig, ax = plt.subplots(figsize=(6,6))

        title_cnt=c.n_word.sort_values(ascending=False).reset_index()

        n= ax.barh(title_cnt.iloc[:,0], title_cnt.iloc[:,1], color='lightgreen', edgecolor='green', height=0.7)

  



        # Remove axes splines

        for s in ['top','bottom','left','right']:

             ax.spines[s].set_visible(False)





          # Remove x,y Ticks

        ax.xaxis.set_ticks_position('none')

        ax.yaxis.set_ticks_position('none')



        # Add padding between axes and labels

        ax.xaxis.set_tick_params(pad=5)

        ax.yaxis.set_tick_params(pad=10)



        # Add x,y gridlines

        ax.grid(b=True, color='grey', linestyle='-.', linewidth=1, alpha=0.3)



        # Show top values 

        ax.invert_yaxis()



        # Add Plot Title

        ax.set_title('Disaster tweets: MOST FREQUENT WORDS',

             loc='center', pad=10, fontsize=16)

        ax.set_xlabel('# of words')





        # Add annotation to bars

        for i in ax.patches:

            ax.text(i.get_width()+5, i.get_y()+0.5, str(round((i.get_width()), 2)),

            fontsize=10, fontweight='bold', color='grey')

        

        r=range(20)

        plt.yticks(r, c.word ,weight='bold')

        





        # Show Plot

        return plt.show()

        
word_counter(no_dis)
word_counter(yes_dis)
hub_layer = hub.KerasLayer("https://tfhub.dev/google/Wiki-words-500-with-normalization/2",

                           input_shape=[], dtype=tf.string)





#Define the model

model = tf.keras.Sequential()

# Add the layer of the pretrained model wiki-word-500

model.add(hub_layer)

#Dense layers +



model.add(tf.keras.layers.Dense(128, activation='relu'))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.BatchNormalization())

model.add(tf.keras.layers.Dense(32, activation='relu'))

model.add(tf.keras.layers.Dropout(0.4))

model.add(tf.keras.layers.BatchNormalization())



model.add(tf.keras.layers.Dense(1, activation='sigmoid'))



model.summary()
model.compile(Adam(lr=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
file_path='my_model_file.h5'

callbacks_list = [

        keras.callbacks.EarlyStopping(

            monitor = 'val_loss', # Use accuracy to monitor the model

            patience = 15 # Stop after 15 steps with lower accuracy

        ),

        keras.callbacks.ModelCheckpoint(

            filepath = file_path, # file where the checkpoint is saved

            monitor = 'val_loss', # Don't overwrite the saved model unless val_loss is worse

            save_best_only = True)]# Only save model if it is the best
#checkpoint = ModelCheckpoint(file_path, monitor='val_loss', save_best_only=True)



history = model.fit(

    train_data, train_labels,

    validation_split=0.3,

    epochs=22,

    callbacks=callbacks_list,

    batch_size=32

)
plt.style.use('seaborn')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,10))

#First Model

ax1 = plt.subplot2grid((2,2),(0,0))

train_loss = history.history['loss']

test_loss = history.history['val_loss']

x = list(range(1, len(test_loss) + 1))

plt.plot(x, test_loss, color = 'cyan', label = 'Test loss')

plt.plot(x, train_loss, label = 'Training losss')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.title('Loss vs. Epoch',weight='bold', fontsize=18)

ax1 = plt.subplot2grid((2,2),(0,1))

train_acc = history.history['accuracy']

test_acc = history.history['val_accuracy']

x = list(range(1, len(test_acc) + 1))

plt.plot(x, test_acc, color = 'cyan', label = 'Test accuracy')

plt.plot(x, train_acc, label = 'Training accuracy')

plt.legend()

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.title('Accuracy vs. Epoch', weight='bold', fontsize=18)

plt.show()
model.load_weights('my_model_file.h5')

prediction = model.predict(test_data)



submission['target'] = prediction.round().astype(int)

submission.to_csv('submission.csv', index=False)
