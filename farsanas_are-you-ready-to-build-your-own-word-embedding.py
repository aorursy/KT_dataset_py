from IPython.display import YouTubeVideo      

YouTubeVideo('o9IxCpl7U54')
import numpy as np

import pandas as pd
!unzip ../input/word2vec-nlp-tutorial/unlabeledTrainData.tsv.zip
df = pd.read_csv("./unlabeledTrainData.tsv",delimiter="\t",quoting=3,header=0)
df.head(10)
import re,string
def clean_string(string):                                                         # The entire document is cleaned defining clean_string

  try:

    string=re.sub(r'^https?:\/\/<>.*[\r\n]*','',string,flags=re.MULTILINE)

    string=re.sub(r"[^A-Za-z]"," ",string)

    words=string.strip().lower().split()

    return " ".join(words)

  except:

    return " "
df['clean_review']=df.review.apply(clean_string)                                  # Finally cleaned format is applied on the reviews

print ("No.of samples \n:",(len(df)))

df.head()
!pip install gensim --quiet  
import gensim
Document=[]

for doc in df['clean_review']:

  Document.append(doc.split(' '))    
len(Document)
print(len(Document[10]))                                                          # Lenth of the 10th document ,  It has 524 words in it

print(Document[10])
# Training the Word 2 Vec model

model=gensim.models.Word2Vec(Document,                                           # List of reviews

                          min_count=10,                                          # we want words appearing atleast 10 times in the vocab otherwise ignore 

                          workers=4,                                             # Use these many worker threads to train the model (=faster training with multicore machines

                           size=50,                                              # it means aword is represented by 50 numbers,in other words the number of neorons in hidden layer is 50 

                          window=5)                                              # 5 neighbors on the either side of a word
print(len(model.wv.vocab))   # Now the vocab contains 28322 uinque words
print(model.wv.vector_size) # It means each vector has 50 numbers in it or in other words each word is vector of 5o numbers that we predefined
model.wv.vectors.shape  # Dimension of the the entire corpus  
model.wv.most_similar("beautiful")                                                # 10 similar words beautiful,the maximum similarity is 1,minimum is 0.When they are completely similar the 

                                                                                  # Value will be 1 , when completely dissimilar,the value will be 0.
model.wv.most_similar("princess")                                                  # 10 similar words returned with numbers
model.wv.doesnt_match("she talked to me in the evening publicly".split())         # publicly does not match in the sentence given
model.wv["right"]                                                                  # right word is represented by 50 numbers in other words the word "right" is vector of 50 numbers

                                                                                   # 50 numbers are summarized weights because these numbers are obtained in the hidden layer of predefined 50 neurons
model.wv['great']
model.save("word2vec movie-50.model")                                                    # We save this model for further use.

                                                                                   # Google has such many pre-trained models