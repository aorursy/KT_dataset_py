import numpy as np 
import pandas as pd
from collections import Counter
import pickle
#!pip install gensim
#!pip install pyLDAvis
import gensim
from gensim import corpora
import pyLDAvis.gensim
import nltk
#Download the nltk dependencies 
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download('stopwords')

from nltk.tokenize import sent_tokenize, word_tokenize 
from nltk.stem import WordNetLemmatizer 
lemmatizer = WordNetLemmatizer() 
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

punc="!#$%&'()*+-/:;<=>?@[\]^_`{|}~@."
stop_words = list(set(stopwords.words('english')))+['dont']

df_train=pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
#df_test=pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
df_train.tail(2)
text_data=list(df['text'])
#Removal of punctuation tokens
for i in range(len(text_data)):
    text_data[i]=text_data[i].translate(str.maketrans('', '', punc))
    
#Tokeniztion of data
word_list=[]
for i in range(0,len(text_data)):
    word_list += nltk.word_tokenize(text_data[i]) 
print(*word_list[:20])
words_lemma_list=[]

for i in range(0,len(word_list)):
    word=lemmatizer.lemmatize(word_list[i].lower())
    if(word not in stop_words and len(word)>2):
        words_lemma_list.append(word)
print(*words_lemma_list[:20])
pos_list=[]
for i in range(0,len(words_lemma_list)):
    pos_list+=nltk.pos_tag([words_lemma_list[i]])
print(*pos_list[20:25])
nouns=[]
for i in range(len(pos_list)):
    if((pos_list[i][1] in ['NN','NNS'])):
        nouns.append(pos_list[i][0])
print("Nouns in the dataset :",*nouns[:10])
print("Number of Nouns in dataset :",len(nouns))
print("Number of distinct Nouns in the dataset : ",len(set(nouns)))
print("Most common Nouns in the dataset : \n",*Counter(nouns).most_common(5))
nouns=[[nouns[i]] for i in range(0,len(nouns))]
dictionary = corpora.Dictionary(nouns)
corpus = [dictionary.doc2bow(text) for text in nouns]

#Use pickle files
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')
NUM_TOPICS = 3
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=10)
for topic in topics:
    print(topic)
dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda = gensim.models.ldamodel.LdaModel.load('model5.gensim')

lda_display = pyLDAvis.gensim.prepare(lda, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)
