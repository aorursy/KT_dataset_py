!pip install PyPDF2
import PyPDF2

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
filename = '../input/manifesto/manibook'

open_filename = open(filename, 'rb')



ind_manifesto = PyPDF2.PdfFileReader(open_filename)

ind_manifesto.getDocumentInfo()
total_pages = ind_manifesto.numPages

total_pages
!pip install textract
import textract   
count = 0

text  = ''



# Lets loop through, to read each page from the pdf file

while(count < total_pages):

    # Get the specified number of pages in the document

    mani_page  = ind_manifesto.getPage(count)

    # Process the next page

    count += 1

    # Extract the text from the page

    text += mani_page.extractText()

   
if text != '':

    text = text

    

else:

    textract.process(open_filename, method='tesseract', encoding='utf-8', langauge='eng' )    
!pip install autocorrect
from autocorrect import Speller

from nltk.tokenize import word_tokenize





def to_lower(text):



    """

    Converting text to lower case as in, converting "Hello" to  "hello" or "HELLO" to "hello".

    """

    

    # Specll check the words

    spell  = Speller(lang='en')

    

    texts = spell(text)

    

    return ' '.join([w.lower() for w in word_tokenize(text)])



lower_case = to_lower(text)

print(lower_case)
import nltk

import re

import string

from nltk.corpus import stopwords, brown

from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer

from nltk.stem import WordNetLemmatizer

from autocorrect import spell
def clean_text(lower_case):

    # split text phrases into words

    words  = nltk.word_tokenize(lower_case)

    

    

    # Create a list of all the punctuations we wish to remove

    punctuations = ['.', ',', '/', '!', '?', ';', ':', '(',')', '[',']', '-', '_', '%']

    

    # Remove all the special characters

    punctuations = re.sub(r'\W', ' ', str(lower_case))

    

    # Initialize the stopwords variable, which is a list of words ('and', 'the', 'i', 'yourself', 'is') that do not hold much values as key words

    stop_words  = stopwords.words('english')

    

    # Getting rid of all the words that contain numbers in them

    w_num = re.sub('\w*\d\w*', '', lower_case).strip()

    

    # remove all single characters

    lower_case = re.sub(r'\s+[a-zA-Z]\s+', ' ', lower_case)

    

    # Substituting multiple spaces with single space

    lower_case = re.sub(r'\s+', ' ', lower_case, flags=re.I)

    

    # Removing prefixed 'b'

    lower_case = re.sub(r'^b\s+', '', lower_case)

    

    

    

    # Removing non-english characters

    lower_case = re.sub(r'^b\s+', '', lower_case)

    

    # Return keywords which are not in stop words 

    keywords = [word for word in words if not word in stop_words  and word in punctuations and  word in w_num]

    

    return keywords
# Lemmatize the words

wordnet_lemmatizer = WordNetLemmatizer()



lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in clean_text(lower_case)]



# lets print out the output from our function above and see how the data looks like

clean_data = ' '.join(lemmatized_word)

print(clean_data)
import pandas as pd

df = pd.DataFrame([clean_data])

df.columns = ['script']

df.index = ['Itula']

df
#  Counting the occurrences of tokens and building a sparse matrix of documents x tokens.

from sklearn.feature_extraction.text import CountVectorizer

import numpy as np



corpus = df.script

vect = CountVectorizer(stop_words='english')



# Transforms the data into a bag of words

data_vect = vect.fit_transform(corpus)
feature_names = vect.get_feature_names()

data_vect_feat = pd.DataFrame(data_vect.toarray(), columns=feature_names)

data_vect_feat.index = df.index

data_vect_feat
data = data_vect_feat.transpose()

data
import matplotlib.pyplot as plt

import seaborn as sn



# Find the top 1000 words written in the manifesto

top_dict = {}

for c in data.columns:

    top = data[c].sort_values(ascending=False)

    top_dict[c]= list(zip(top.index, top.values))



    

for x in list(top_dict)[0:100]:

    print("key {}, value {} ".format(x,  top_dict[x]))

    
# Look at the most common top words --> add them to the stop word list

from collections import Counter



# Let's first pull out the top 100 words for each comedian

words = []

for president in data:

    top = [word for (word, count) in top_dict[president]]

    for t in top:

        words.append(t)



print(words[:10])
from wordcloud import WordCloud, STOPWORDS

import imageio

import matplotlib.pyplot as plt

import nltk



# Image used in which our world cloud output will be

img1 = imageio.imread("../input/manifesto/itula.jpeg")

hcmask1 = img1



# Get 100 words based on the 

words_except_stop_dist = nltk.FreqDist(w for w in words[:100]) 

wordcloud = WordCloud(stopwords=set(STOPWORDS),background_color='black',mask=hcmask1).generate(" ".join(words_except_stop_dist))

plt.imshow(wordcloud,interpolation = 'bilinear')

fig=plt.gcf()

fig.set_size_inches(10,12)

plt.axis('off')

plt.title("Top most common 100 words from Dr. Itula's Manifesto 2019",fontsize=20)

plt.tight_layout(pad=0)

plt.savefig('Manifesto_top_100.jpeg')
!pip install vaderSentiment
from collections import defaultdict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from textblob import TextBlob
blob = TextBlob(clean_data)

blob.sentiment
# Importing all the necessary libraries

from nltk.cluster.util import cosine_distance

from nltk.tokenize import sent_tokenize

import numpy as np

import networkx as nx

import warnings

warnings.filterwarnings("ignore",category=DeprecationWarning)
def sentence_similarity(sent1, sent2, stopwords=None):

    if stopwords is None:

        stopwords = []

 

    sent1 = [w.lower() for w in sent1]

    sent2 = [w.lower() for w in sent2]

 

    all_words = list(set(sent1 + sent2))

 

    vector1 = [0] * len(all_words)

    vector2 = [0] * len(all_words)

 

    # build the vector for the first sentence

    for w in sent1:

        if w in stopwords:

            continue

        vector1[all_words.index(w)] += 1

 

    # build the vector for the second sentence

    for w in sent2:

        if w in stopwords:

            continue

        vector2[all_words.index(w)] += 1

 

    return 1 - cosine_distance(vector1, vector2)

 

# One out of 5 words differ => 0.8 similarity

print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split()))

 

# One out of 2 non-stop words differ => 0.5 similarity

print(sentence_similarity("This is a good sentence".split(), "This is a bad sentence".split(), stopwords.words('english')))

 

# 0 out of 2 non-stop words differ => 1 similarity (identical sentences)

print(sentence_similarity("This is a good sentence".split(), "This is a good sentence".split(), stopwords.words('english')))

 

# Completely different sentences=> 0.0

print(sentence_similarity("This is a good sentence".split(), "I want to go to the market".split(), stopwords.words('english')))

#print(sentences)



# get the english list of stopwords

#stop_words = stopwords.words('english')

 

def build_similarity_matrix(lower_case, stopwords=None):

    # Create an empty similarity matrix

    S = np.zeros([len(lower_case), len(lower_case)])

 

 

    for idx1 in range(len(lower_case)):

        for idx2 in range(len(lower_case)):

            if idx1 == idx2:

                continue

 

            S[idx1][idx2] = sentence_similarity(lower_case[idx1], lower_case[idx2], stop_words)

 

    # normalize the matrix row-wise

    for idx in range(len(S)):

        S[idx] /= S[idx].sum()

 

    return S
#len(sentences)

#S = build_similarity_matrix(sentences, stop_words)    

#S
def generate_summary(lower_case, top_n=5):

    # Remove all the stopwords in the document

    stop_words = stopwords.words('english')

    summarize_text = []

    

    

    

    #Read text and tokenize

    #lower_case  = nltk.word_tokenize(lower_case)

    

   

    

    #Generate similarity matrix across sentences

    sentence_similarity  = build_similarity_matrix((lower_case, stop_words))

    

    #Rank sentences in similarity matrix

    sentence_similiraty_graph = nx.from_numpy_array(sentence_similarity)

    scores = nx.pagerank(sentence_similiraty_graph)

    

    

    #Sort the rank and pick top sentences

    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(lower_case)), reverse=True)    

    print("Indexes of top ranked_sentence order are ", ranked_sentence) 

    

    for i in range(top_n):

        summarize_text.append(' '.join(ranked_sentence[i][1]))

        

    #Output the summarized text

    print('Summarized Text: \n', '. '.join(summarize_text))

       
#generate_summary(lower_case, 2)
# imports

from gensim.summarization.summarizer import summarize
# Print out our summarized text of the document which was converted to lower case, remember we could have opted to remove stopwords as well.



print(summarize(lower_case))
# Gensim

import gensim

import gensim.corpora as corpora

from gensim.utils import simple_preprocess

from gensim.models import CoherenceModel



# Plotting tools

import pyLDAvis

#import graphlab as gl

#import pyLDAvis.graphlab

import pyLDAvis.gensim  # don't skip this



# Enable logging for gensim - optional

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)
data  = []

data.append(clean_text(lower_case))
# This time we use spacy for lemmatizarion 

import spacy



# Second lemmatization of our data

def lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):

    """https://spacy.io/api/annotation"""

    texts_output = []

    for sent in data:

        doc = nlp(" ".join(sent)) 

        texts_output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])

    return texts_output



nlp = spacy.load('en', disable=['parser', 'ner'])



# Lemmatize keeping only noun, adj, vb, adv

data_lemmatized = lemmatization(data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
# Create Dictionary

id2word = corpora.Dictionary(data_lemmatized)



# Create Corpus

texts = data_lemmatized



# Term Document Frequency

corpus = [id2word.doc2bow(text) for text in texts]





# View

print(corpus[:1])
# LDA model



lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=id2word, alpha='auto', num_topics=20, random_state=100,

                                           update_every=1, passes=20, per_word_topics=True)

# Lets view the topics in our model

print(lda_model.print_topics())

doc_lda  = lda_model[corpus]
# Print model perplexity

print('\nPerplexity:', lda_model.log_perplexity(corpus))





# Coherence Score



coherence_model_lda = CoherenceModel(lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')

coherence_lda = coherence_model_lda.get_coherence()

print('Coherence Score:', coherence_lda)
pyLDAvis.enable_notebook()

vis_topics = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
vis_topics