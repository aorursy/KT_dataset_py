import os
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
print(os.listdir("../input"))

max_features=50000
maxlen=500
embed_size=300
#EMBEDDING_FILE='bengali-word-embedding.txt'
EMBEDDING_FILE='../input/bengali-word-embedding/bengali-word-embedding.txt'
vocab_vector = {} 
with open(EMBEDDING_FILE,encoding='utf8') as f:  
    for line in f:
        values = line.rstrip().rsplit(' ')
        word_values = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        vocab_vector[word_values] = coefs
f.close()        
#print('Found %s word vectors.' % len(embeddings_index))
#data = pd.read_csv('dataset.csv')
data=pd.read_csv('../input/dataset/dataset.csv')
text = data['doc']
#print(text)
label = data['label']
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(text)
sequence = tokenizer.texts_to_sequences(text)
word_index = tokenizer.word_index
#print(word_index)
#print('Found %s unique tokens.' % len(word_index))
X = pad_sequences(sequence, maxlen=maxlen, padding='post', truncating='post')
t = Tokenizer(num_words=max_features)
t.fit_on_texts(label)
sequence1 = tokenizer.texts_to_sequences(label)
Y = pad_sequences(sequence1, maxlen=maxlen, padding='post', truncating='post')

#print(X)
#print('Shape of data tensor:', X.shape)
#print('Shape of label tensor:', Y.shape)
embedding_matrix=np.zeros((len(word_index)+1,300))
for embed_word,v in word_index.items():
    embedding_vector=vocab_vector.get(embed_word)
    print(embedding_vector)
    #words that cannot be found will be set to 0
    if embedding_vector is not None:
        embedding_matrix[v]=embedding_vector


                        
                        
                        
                     



       
                      
                   
                    


      

 




 

                        