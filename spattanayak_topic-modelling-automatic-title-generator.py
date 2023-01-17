import numpy as np

import pandas as pd

from IPython.display import display

from tqdm import tqdm

from collections import Counter

import ast



import matplotlib.pyplot as plt

import matplotlib.mlab as mlab

import seaborn as sb



from sklearn.feature_extraction.text import CountVectorizer

from textblob import TextBlob

import scipy.stats as stats



from sklearn.decomposition import TruncatedSVD

from sklearn.decomposition import LatentDirichletAllocation

from sklearn.manifold import TSNE



from bokeh.plotting import figure, output_file, show

from bokeh.models import Label

from bokeh.io import output_notebook

output_notebook()



%matplotlib inline
npr = pd.read_csv('../input/topicmodelling/npr.csv')

npr=npr.head(50)
npr.iloc[0,1] 
reindexed_data = npr['Article']
# Define helper functions

def get_top_n_words(n_top_words, count_vectorizer, text_data):

    '''

    returns a tuple of the top n words in a sample and their 

    accompanying counts, given a CountVectorizer object and text sample

    '''

    vectorized_headlines = count_vectorizer.fit_transform(text_data.values)

    vectorized_total = np.sum(vectorized_headlines, axis=0)

    word_indices = np.flip(np.argsort(vectorized_total)[0,:], 1)

    word_values = np.flip(np.sort(vectorized_total)[0,:],1)

    

    word_vectors = np.zeros((n_top_words, vectorized_headlines.shape[1]))

    for i in range(n_top_words):

        word_vectors[i,word_indices[0,i]] = 1



    words = [word[0].encode('ascii').decode('utf-8') for 

             word in count_vectorizer.inverse_transform(word_vectors)]



    return (words, word_values[0,:n_top_words].tolist()[0])
count_vectorizer = CountVectorizer(stop_words='english')

words, word_values = get_top_n_words(n_top_words=25,

                                     count_vectorizer=count_vectorizer, 

                                     text_data=reindexed_data)



fig, ax = plt.subplots(figsize=(16,8))

ax.bar(range(len(words)), word_values);

ax.set_xticks(range(len(words)));

ax.set_xticklabels(words, rotation='vertical');

ax.set_title('Top words in headlines dataset (excluding stop words)');

ax.set_xlabel('Word');

ax.set_ylabel('Number of occurences');

plt.show()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
cv
dtm = cv.fit_transform(npr['Article'])
dtm
from sklearn.decomposition import LatentDirichletAllocation
LDA = LatentDirichletAllocation(n_components=7,random_state=42)
# This can take a while as we are dealing with a large amount of documents!

LDA.fit(dtm)
lda_topic_matrix = LDA.fit_transform(dtm)
len(cv.get_feature_names())
import random
for i in range(10):

    random_word_id = random.randint(0,500)

    print(cv.get_feature_names()[random_word_id])
len(LDA.components_)
LDA.components_
len(LDA.components_[0])
single_topic = LDA.components_[0]
# Returns the indices that would sort this array.

single_topic.argsort()
# Word least representative of this topic

single_topic[100]
# Word most representative of this topic

single_topic[1294]
# Top 10 words for this topic:

single_topic.argsort()[-10:]
top_word_indices = single_topic.argsort()[-10:]
for index in top_word_indices:

    print(cv.get_feature_names()[index])
for index,topic in enumerate(LDA.components_):

    print(f'THE TOP 15 WORDS FOR TOPIC #{index}')

    print([cv.get_feature_names()[i] for i in topic.argsort()[-60:]])

    print('\n')
dtm
dtm.shape
len(npr)
topic_results = LDA.transform(dtm)
topic_results.shape
topic_results[0]
topic_results[0].round(2)
topic_results[0].argmax()
npr.head()
topic_results.argmax(axis=1)
npr['Topic'] = topic_results.argmax(axis=1)
npr.head(10)
tf_feature_names = cv.get_feature_names()
first_topic = LDA.components_[0]

second_topic = LDA.components_[1]

third_topic = LDA.components_[2]

fourth_topic = LDA.components_[3]
first_topic.shape
first_topic_words = [tf_feature_names[i] for i in first_topic.argsort()[:-50 - 1 :-1]]

second_topic_words = [tf_feature_names[i] for i in second_topic.argsort()[:-50 - 1 :-1]]

third_topic_words = [tf_feature_names[i] for i in third_topic.argsort()[:-50 - 1 :-1]]

fourth_topic_words = [tf_feature_names[i] for i in fourth_topic.argsort()[:-50 - 1 :-1]]
type(first_topic_words)
first_topic_words
x=list(first_topic_words)

x
# Generating the wordcloud with the values under the category dataframe

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

firstcloud = WordCloud(

                          stopwords=STOPWORDS,

                          background_color='black',

                          width=2500,

                          height=1800

                         ).generate(" ".join(second_topic_words))

plt.imshow(firstcloud)

plt.axis('off')

plt.show()
npr_article=npr[(npr["Topic"] == 1) ][["Article"]]

from collections import Counter

result = Counter(" ".join(npr_article['Article'].values).split(" ")).items()

result
npr_article1=npr

npr_article1
npr_article1['Topic_join'] = npr_article1.apply(lambda x: ','.join([x['Article']] ), axis=1)

npr_article2=npr_article1.groupby('Topic')['Topic_join'].apply(list)



npr_article2
a = pd.Series.to_frame(npr_article2)



a['Topic_id'] = list(a.index)
a
sent= pd.Series.to_frame(a[(a['Topic_id'] ==3)]["Topic_join"])
sent=sent.iat[0,0]
sent3=''.join(sent)
sent3 = sent3.replace("'", "")

sent3 = sent3.replace('"', "")



sent3=sent3.replace("”","")

sent3=sent3.replace("’","")
import spacy

import string

nlp = spacy.load('en')    

sent1 = sent3

import re





clean = re.sub(r"""

               [,.;@#?!&$]+  # Accept one or more copies of punctuation

               \ *           # plus zero or more copies of a space,

               """,

               " ",          # and replace it with a single space

               sent1, flags=re.VERBOSE)

sent=clean



doc=nlp(sent)



my_list = sent.split()



sub_toks = [tok for tok in doc if ((tok.dep_ == "nsubj") )]

print(sub_toks)



nc= [x for x in doc.noun_chunks]

print(nc)

x=0

for i,token in enumerate(doc):



    if token.pos_ in ('PROPN'):

        x=i

    if token.pos_ in ('PRON'):

        y=i

        

        try:

            my_list[y]=my_list[x]

        except:

            print(i)



        

print(my_list)





print(x)

my_new_string = " ".join(my_list)

print(my_new_string)
from nltk.corpus import stopwords

my_new_string = ' '.join([word for word in my_new_string.split() if word not in (stopwords.words('english'))])
my_new_string
from collections import Counter

from nltk.tag import pos_tag

from nltk.tokenize import word_tokenize

count= Counter([j for i,j in pos_tag(word_tokenize(my_new_string))])

print (count)
import spacy

from collections import Counter

nlp = spacy.load('en')

doc = nlp(my_new_string)



# all tokens that arent stop words or punctuations

words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True]

# five most common tokens

word_freq = Counter(words)

common_words = word_freq.most_common(5)





# noun tokens that arent stop words or punctuations

nouns = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]

# five most common noun tokens

noun_freq = Counter(nouns)

common_nouns = noun_freq.most_common(5)





propnouns = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "PROPN"]

propnoun_freq = Counter(propnouns)

common_propnouns = propnoun_freq.most_common(5)



adjs = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "ADJ"]

adjs_freq = Counter(adjs)

common_adjs = adjs_freq.most_common(5)



print(common_words)

print(common_nouns)

print(common_propnouns)

print(common_adjs)
common_propnouns[0][0]
common_adjs[0][0]
from collections import defaultdict, Counter

pos_counts = defaultdict(Counter)

for token in doc:

    pos_counts[token.pos][token.orth] += 1



for pos_id, counts in sorted(pos_counts.items()):

    pos = doc.vocab.strings[pos_id]

    for orth_id, count in counts.most_common():

        print(pos, count, doc.vocab.strings[orth_id])
import string

Title= common_adjs[0][0]  + " " + common_propnouns[0][0]

print(Title.upper())
from nltk import ne_chunk, pos_tag, word_tokenize

from nltk.tree import Tree

 

def get_continuous_chunks(text):

     chunked = ne_chunk(pos_tag(word_tokenize(text)))

     continuous_chunk = []

     current_chunk = []

     for i in chunked:

             if type(i) == Tree:

                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))

             elif current_chunk:

                     named_entity = " ".join(current_chunk)

                     if named_entity not in continuous_chunk:

                             continuous_chunk.append(named_entity)

                             current_chunk = []

             else:

                     continue

     return continuous_chunk



get_continuous_chunks(my_new_string)

import nltk

for sent in nltk.sent_tokenize(my_new_string):

   for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):

      if hasattr(chunk, 'label'):

         print(chunk.label(), ' '.join(c[0] for c in chunk))
import spacy

nlp = spacy.load('en',disable=['parser', 'tagger','ner'])



nlp.max_length = 1198623
def separate_punc(doc_text):

    return [token.text.lower() for token in nlp(doc_text) if token.text not in '\n\n \n\n\n!"-#$%&()--.*+,-/:;<=>?@[\\]^_`{|}~\t\n ']
d = my_new_string

tokens = separate_punc(d)
tokens
len(tokens)
# organize into sequences of tokens

train_len = 25+1 # 50 training words , then one target word



# Empty list of sequences

text_sequences = []



for i in range(train_len, len(tokens)):

    

    # Grab train_len# amount of characters

    seq = tokens[i-train_len:i]

    

    # Add to list of sequences

    text_sequences.append(seq)
' '.join(text_sequences[0])
' '.join(text_sequences[1])
' '.join(text_sequences[2])
len(text_sequences)
from keras.preprocessing.text import Tokenizer
# integer encode sequences of words

tokenizer = Tokenizer()

tokenizer.fit_on_texts(text_sequences)

sequences = tokenizer.texts_to_sequences(text_sequences)
sequences[0]
tokenizer.index_word
for i in sequences[0]:

    print(f'{i} : {tokenizer.index_word[i]}')
tokenizer.word_counts
vocabulary_size = len(tokenizer.word_counts)
import numpy as np

sequences = np.array(sequences)

sequences
import keras

from keras.models import Sequential

from keras.layers import Dense,LSTM,Embedding
def create_model(vocabulary_size, seq_len):

    model = Sequential()

    model.add(Embedding(vocabulary_size, 25, input_length=seq_len))

    model.add(LSTM(150, return_sequences=True))

    model.add(LSTM(150))

    model.add(Dense(150, activation='relu'))



    model.add(Dense(vocabulary_size, activation='softmax'))

    

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

   

    model.summary()

    

    return model
from keras.utils import to_categorical
sequences
# First 49 words

sequences[:,:-1]
# last Word

sequences[:,-1]
X = sequences[:,:-1]

y = sequences[:,-1]

y = to_categorical(y, num_classes=vocabulary_size+1)

seq_len = X.shape[1]

seq_len
# define model

model = create_model(vocabulary_size+1, seq_len)
from pickle import dump,load
# fit model

model.fit(X, y, batch_size=128, epochs=100,verbose=1)
import os

print(os.listdir('../'))
# save the model to file

model.save('../working/epochBIG.h5')


# save the tokenizer

dump(tokenizer, open('epochBIG', 'wb'))
from random import randint

from pickle import load

from keras.models import load_model

from keras.preprocessing.sequence import pad_sequences
def generate_text(model, tokenizer, seq_len, seed_text, num_gen_words):

    '''

    INPUTS:

    model : model that was trained on text data

    tokenizer : tokenizer that was fit on text data

    seq_len : length of training sequence

    seed_text : raw string text to serve as the seed

    num_gen_words : number of words to be generated by model

    '''

    

    # Final Output

    output_text = []

    

    # Intial Seed Sequence

    input_text = seed_text

    

    # Create num_gen_words

    for i in range(num_gen_words):

        

        # Take the input text string and encode it to a sequence

        encoded_text = tokenizer.texts_to_sequences([input_text])[0]

        

        # Pad sequences to our trained rate (50 words in the video)

        pad_encoded = pad_sequences([encoded_text], maxlen=seq_len, truncating='pre')

        

        # Predict Class Probabilities for each word

        pred_word_ind = model.predict_classes(pad_encoded, verbose=0)[0]

        

        # Grab word

        pred_word = tokenizer.index_word[pred_word_ind] 

        

        # Update the sequence of input text (shifting one over with the new word)

        input_text += ' ' + pred_word

        

        output_text.append(pred_word)

        

    # Make it look like a sentence.

    return ' '.join(output_text)
import random

random.seed(101)

random_pick = random.randint(0,len(text_sequences))

random_seed_text = text_sequences[random_pick]

random_seed_text
seed_text = ' '.join(random_seed_text)

seed_text
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=50)
seed_text=common_adjs[0][0]  + " " + common_propnouns[0][0]
seed_text
generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=4)
Title=seed_text + " " + generate_text(model,tokenizer,seq_len,seed_text=seed_text,num_gen_words=4)

Title.upper()