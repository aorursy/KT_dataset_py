# NATURAL LANGUAGE PROCESSING

# # SENTIMENT ANALYSIS ON IMDB MOVIE REVIEWS DATASET







# Hello Reader,

# Before we begin I would like to tell you I am new to Kaggle and to notebook publishing but I like to write so please feel

# free to comment and give your suggestions. Thank You



# P.S. I am really excited to do this :D
# DEPENDENCIES



import numpy as np 

import pandas as pd 
# Read and Load the data into a dataframe using the read_csv function of pandas



df = pd.read_csv('/kaggle/input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv')

# df.head()



# The values given in the dataset for sentiment are strings ('positive' & 'negative').

# Make a new list to with numeric values for the sentiment. 1- for positive and 0- for negative.

# Although it is not required to do so but I personally prefer to do it this way.





sentiments = []

for sentiment in df['sentiment'] :

    sentiments.append(int(sentiment=='positive'))

    

# drop the older sentiment column that contained string values and replace it with our new sentiment list containing integer values.

df.drop(columns=['sentiment'],axis=1,inplace=True)

df['sentiment']=sentiments

df.head()
from sklearn.model_selection import train_test_split

# Our features here are the review sentences and our label is the sentiment.

X = df['review']

y= df['sentiment']



# Split the data into two sets : Training and Testing set. Use shuffle=True so that our data is shuffled before splitting.

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,shuffle=True)

# print(X_train.shape)

# print(y_train.shape)

# print(X_test.shape)

# print(y_test.shape)

# X_train.head()
# We need to create a list of lists where each inner list is a sentence in the set. 



training_sentences = []

training_labels = []

testing_sentences = []

testing_labels = []
# Append the senteces from training and testing datasets into their respective list along with their labels in their own lists.





for sentence in X_train :

    training_sentences.append(sentence)

for label in y_train :

    training_labels.append(label)

    

for sentence in X_test :

    testing_sentences.append(sentence)

for label in y_test :

    testing_labels.append(label)

    
# Debugging if everything has been done correctly.



# print(len(training_sentences))

# print(len(training_labels))



# print(len(testing_sentences))

# print(len(testing_labels))



# Convert the list of labels into an np.array() that we will later use to train our neural network.





training_labels_final = np.array(training_labels)

testing_labels_final = np.array(testing_labels)
# Parameters.

vocab_size=10000  #Defines the dimesions of vector when each word is converted into a sequence.

embedding_dim = 16 

oov_tok= '<OOV>' # Our test set is definitely gonna have words that don't appear in our training set and to handle them we 

# provide them with a special token known as OOV_token (Out-of-Vocabulary Token) 

trunc_type = 'post' #If the length of our sentence is greater than the specified length ,

# the words are truncated from the beginning we change that so that the words are truncated from the end.

max_length = 250
# Import the tokenizer to convert each sentence into a bunch of tokens and then to sequences using the keras text preprocessing

# Import the pad_sequences to convert the sequences of unequal lengths to an equal length .



from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences



# Instantiate the tokenizer

tokenizer = Tokenizer(num_words=vocab_size,oov_token = oov_tok)

# Fit the tokenizer onto our training sentences

tokenizer.fit_on_texts(training_sentences)



# Contains the word and its token/index value

word_index = tokenizer.word_index



# Convert the training sentences into sequences of token/index values based on the tokenizer fit

training_sequences = tokenizer.texts_to_sequences(training_sentences)



# Pad the sequences so that each sequence is of an equal length.

training_padded = pad_sequences(training_sequences,truncating=trunc_type,maxlen = max_length)



test_sequences = tokenizer.texts_to_sequences(testing_sentences)

test_padded = pad_sequences(test_sequences,maxlen = max_length)
import tensorflow as tf



# Create our model.

# Since we are doing text processing we use the embeddings layer as our first layer.

model = tf.keras.Sequential([

                              tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=max_length),

                              tf.keras.layers.Flatten(),

                              tf.keras.layers.Dense(6,activation='relu'),

                              tf.keras.layers.Dense(1,activation='sigmoid'),

                              ])
# Since our ouput is binary 1/0 we use the loss as binary_crossentropy



model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
num_epochs = 10

history=model.fit(training_padded, training_labels_final, epochs=num_epochs, validation_data=(test_padded, testing_labels_final))
# e = model.layers[0]

# weights = e.get_weights()[0]

# print(weights.shape)

# word_index has a map from word to index.

# We create a reverse word index to map index to the word.



reverse_word_index = dict([(value,key)for (key,value) in word_index.items()])

reverse_word_index
import matplotlib.pyplot as plt





def plot_graphs(history, string):

  plt.plot(history.history[string])

  plt.plot(history.history['val_'+string])

  plt.xlabel("Epochs")

  plt.ylabel(string)

  plt.legend([string, 'val_'+string])

  plt.show()

plot_graphs(history, 'accuracy')

plot_graphs(history, 'loss')
# The validation loss is much greater than our training loss so there is overfitting into the data. But you generally see this 

# when working on text data. Feel free to experiment with other models/parameters and more.
# We still have an accuracy of about 86% on our validation data so we can say our model is doing a pretty good job 

# for this version 1.
test_sentence = ['The movie was horrible and the actors were bad and the actress was disgusting and I did not like the movie and It was the worst film of my life.',

                 'The movie was great and I really liked the movie. It was quite interesting and I am happy to have watched it.']
sq = tokenizer.texts_to_sequences(test_sentence)

# print(sq)

pt = pad_sequences(sq,maxlen=max_length)



prediction = model.predict(pt)

labels = ((prediction > 0.5).astype(np.int)) 

sentiment=[]

for label in labels : 

    if label==1:

        sentiment.append('positive')

    else :

        sentiment.append('negative')

print(sentiment)