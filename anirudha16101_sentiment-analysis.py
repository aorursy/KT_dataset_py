# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install tensorflow
amazon_df=pd.read_csv("/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/amazon_cells_labelled.txt",delimiter='\t',

                        header=None, 

                        names=['review', 'sentiment'])



imdb_df = pd.read_csv("/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/imdb_labelled.txt", 

                        delimiter='\t', 

                        header=None, 

                        names=['review', 'sentiment'])



yelp_df = pd.read_csv("/kaggle/input/sentiment-labelled-sentences-data-set/sentiment labelled sentences/sentiment labelled sentences/yelp_labelled.txt", 

                        delimiter='\t', 

                        header=None, 

                        names=['review', 'sentiment'])



amazon_df.head()
imdb_df.head()
yelp_df.head()
data=pd.concat([amazon_df,yelp_df,imdb_df])
data.reset_index(drop='True',inplace=True)

data
#Extracting Reviews and Sentiments

sentences=data['review'].tolist()

label=data['sentiment'].tolist()
# print some examples of sentences and labels



print("Sentences")

for i in range(10):

    #print(sentences[i],end="\n")

    print("{} {}".format(sentences[i],label[i]))

    

!pip install tensorflow_datasets
#Create A Subword Datasets

import tensorflow_datasets as tfds

import tensorflow as tf



from tensorflow.keras.preprocessing.sequence import pad_sequences



vocab_size=1000

tokenizer=tfds.features.text.SubwordTextEncoder.build_from_corpus(sentences,vocab_size,max_subword_length=5)
print("vocab size is",vocab_size)



#check the tokenizer words

num=1

print(sentences[num])

encoded_sentence=tokenizer.encode(sentences[num])

print(encoded_sentence)
for i in encoded_sentence:

    print(tokenizer.decode([i]))
for i,sent in enumerate(sentences):

    sentences[i]=tokenizer.encode(sent)
#print some encoded text

for i in range(10):

    print(sentences[i],end="\n")
import numpy as np

max_length =50

trunc_type='post'

padding_type='post'



#pad all Sequence



sequence_added=pad_sequences(sentences,maxlen=max_length,padding =padding_type,truncating=trunc_type)
#Separate the separate and Sentences on training and test data sets



training_size=int(len(sentences)*0.8)

train_seq=sequence_added[:training_size]

train_labels=label[:training_size]



test_seq=sequence_added[training_size:]

test_labels=label[training_size:]



train_labels=np.array(train_labels)

test_labels=np.array(test_labels)
print("Total no of Training Sequence are",len(train_seq))

print("Total no of Test Sequence are",len(test_seq))
#Create a Model

embedding_dim=16

model=tf.keras.Sequential([

    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim,return_sequences=True)),

    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),

    tf.keras.layers.Dense(6,activation='relu'),

    tf.keras.layers.Dense(1,activation='sigmoid')

])



#fit a model

model.compile(loss="binary_crossentropy",optimizer='adam',metrics=['accuracy'])

model.summary()
history=model.fit(train_seq,train_labels,epochs=50,validation_data=(test_seq,test_labels))
import matplotlib.pyplot as plt





def plot_graphs(history, string):

        plt.plot(history.history[string])

        plt.plot(history.history['val_'+string])

        plt.xlabel("Epochs")

        plt.ylabel(string)

        plt.legend([string, 'val_'+string])

        plt.show()



plot_graphs(history, "accuracy")

plot_graphs(history, "loss")
def predict_review(model, new_sentences, maxlen=max_length, show_padded_sequence=True ):

    # Keep the original sentences so that we can keep using them later

    # Create an array to hold the encoded sequences

    new_sequences = []



    # Convert the new reviews to sequences

    for i, frvw in enumerate(new_sentences):

        new_sequences.append(tokenizer.encode(frvw))



    trunc_type='post' 

    padding_type='post'



    # Pad all sequences for the new reviews

    new_reviews_padded = pad_sequences(new_sequences, maxlen=max_length, 

                                 padding=padding_type, truncating=trunc_type)             



    classes = model.predict(new_reviews_padded)



    # The closer the class is to 1, the more positive the review is

    for x in range(len(new_sentences)):



        # We can see the padded sequence if desired

        # Print the sequence

        if (show_padded_sequence):

              print(new_reviews_padded[x])

        # Print the review as text

        print(new_sentences[x])

        # Print its predicted class

        print(classes[x])

        print("\n")
# Use the model to predict some reviews   

fake_reviews = ["I love this phone", 

                "Everything was cold",

                "Everything was hot exactly as I wanted", 

                "Everything was green", 

                "the host seated us immediately",

                "they gave us free chocolate cake", 

                "we couldn't hear each other talk because of the shouting in the kitchen"

              ]



predict_review(model, fake_reviews)