# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



from keras.preprocessing.text import Tokenizer

from gensim.models.fasttext import FastText

import numpy as np

import matplotlib.pyplot as plt

import nltk

from string import punctuation

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import sent_tokenize

from nltk import WordPunctTokenizer



#import wikipedia

import nltk

nltk.download('punkt')

nltk.download('wordnet')

nltk.download('stopwords')

en_stop = set(nltk.corpus.stopwords.words('english'))



%matplotlib inline
import pandas as pd

df = pd.read_csv("../input/flipkart-products/flipkart_com-ecommerce_sample.csv")
df.head()
df = df[['product_name', 'product_category_tree', 'description']]

df.head()
import re

from nltk.stem import WordNetLemmatizer



stemmer = WordNetLemmatizer()



def preprocess_text(document):

        # Remove all the special characters

        document = re.sub(r'\W', ' ', str(document))



        # remove all single characters

        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)



        # Remove single characters from the start

        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)



        # Substituting multiple spaces with single space

        document = re.sub(r'\s+', ' ', document, flags=re.I)



        # Removing prefixed 'b'

        document = re.sub(r'^b\s+', '', document)



        # Converting to Lowercase

        document = document.lower()



        # Lemmatization

        tokens = document.split()

        tokens = [stemmer.lemmatize(word) for word in tokens]

        tokens = [word for word in tokens if word not in en_stop]

        tokens = [word for word in tokens if len(word) > 3]



        preprocessed_text = ' '.join(tokens)

        

        return preprocessed_text
df1 = df.copy()
df1['new'] = df1['product_name'] + ' ' + df1['product_category_tree'] + ' ' + df1['description']

df1.head()
final_corpus = [preprocess_text(sentence) for sentence in df1['new'] if str(sentence).strip() !='']

final_corpus[0]
word_punctuation_tokenizer = nltk.WordPunctTokenizer()

word_tokenized_corpus = [word_punctuation_tokenizer.tokenize(sent) for sent in final_corpus]

word_tokenized_corpus[0]
embedding_size = 60

window_size = 4

min_word = 5

down_sampling = 1e-2
%%time

ft_model = FastText(word_tokenized_corpus,

                      size=embedding_size,

                      window=window_size,

                      min_count=min_word,

                      sample=down_sampling,

                      sg=1,

                      iter=100)
'''from gensim.test.utils import get_tmpfile

fname = get_tmpfile("../output/fasttext.model")

ft_model.save(fname)

ft_model = FastText.load(fname)'''

#ft_model.save_model('../input/ft_model.bin')
word = "eternal"

word in ft_model.wv.vocab
word = "crazy"

word in ft_model.wv.vocab
ft_model.wv.vocab
ft_model['pet']
print(ft_model.most_similar("clothes"))

print(ft_model.most_similar("kitchen"))

print(ft_model.most_similar("mobile"))
print(ft_model.most_similar("kids"))
print(ft_model.most_similar("footwear"))
sent = [final_corpus[0]]

for word in sent:

    print(word)
doc = 'alisha solid woman cycling short clothing woman clothing lingerie sleep swimwear short alisha short alisha solid woman cycling short feature alisha solid woman cycling short cotton lycra navy navy specification alisha solid woman cycling short short detail number content sale package pack fabric cotton lycra type cycling short general detail pattern solid ideal woman fabric care gentle machine wash lukewarm water bleach additional detail style code altht_3p_21 short'

words = [word for word in [doc]]

print(words)
def get_mean_vector(model, words):

    # remove out-of-vocabulary words

    wrds = [word for word in [words] if word in model.wv.vocab]

    print(wrds)

    if len(words) >= 1:

        return np.mean(model[words], axis=0)

    else:

        return []
dict = {} 

  

dict[1]='anjani'

dict[2]='vikas'

dict
'''for doc in final_corpus:

    for word in doc.split():

        if word in ft_model.wv.vocab:

            print(word+'')'''
def ConvertVectorSetToVecAverageBased(vectorSet, ignore = []):

    if len(ignore) == 0: 

        return np.mean(vectorSet, axis = 0)

    else: 

        return np.dot(np.transpose(vectorSet),ignore)/sum(ignore)
vectorSet = []

for doc in final_corpus:

    wrds = [word for word in doc.split() if word in ft_model.wv.vocab]

    for aWord in wrds:

        try:

            wordVector=ft_model[aWord]

            vectorSet.append(wordVector)

        except:

            pass

    ConvertVectorSetToVecAverageBased(vectorSet)   
i=0

vec = {}

for doc in final_corpus:

    i = i+1

    #print(doc)

    #vec[i] = get_mean_vector(ft_model, doc)

    wrds = [word for word in doc.split() if word in ft_model.wv.vocab]

    #wrds = [word for word in doc]

    #print(wrds)

    #print(wrds)

    if len(words) >= 1:

        #vec.append(np.mean(ft_model[wrds], axis=0))

        vec[i] = np.mean(ft_model[np.array(wrds)])

        #print(i)

    
vec
for doc in final_corpus:

    print(doc + '*****')