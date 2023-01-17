import pandas as pd

#from bs4 import BeautifulSoup

import re

from nltk.corpus import stopwords

from gensim.models import word2vec

import pickle

import nltk.data

import os

# Load the punkt tokenizer

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# Read data from files 

path = '../input/'

TRAIN_DATA_FILE=f'{path}train.csv'

TEST_DATA_FILE=f'{path}test.csv'





train = pd.read_csv(TRAIN_DATA_FILE, header=0)

test = pd.read_csv(TEST_DATA_FILE, header=0)



# Verify the number of comments that were read

print("Read %d labeled train reviews and  %d unlabelled test reviews" % (len(train),len(test)))

all_comments = train['comment_text'].fillna("_na_").tolist() + test['comment_text'].fillna("_na_").tolist() 





with open("all_comments.csv", "w+") as comments_file:

    i=0

    for comment in all_comments:

        comment = re.sub("[^a-zA-Z]"," ",str(comment))

        comments_file.write("%s\n" % comment)

        
class FileToComments(object):    

    def __init__(self, filename):

        self.filename = filename

        self.stop = set(nltk.corpus.stopwords.words('english'))

        

    def __iter__(self):

        

        def comment_to_wordlist(comment, remove_stopwords=True):

            comment = str(comment)

            words = comment.lower().split()

            #if remove_stopwords:

            #    stops = set(stopwords.words("english"))

            #    words = [w for w in words if not w in stops]

            return(words)

    

        for line in open(self.filename, 'r'):

            #line = unicode(line, 'utf-8')

            tokenized_comment = comment_to_wordlist(line, tokenizer)

            yield tokenized_comment

        

all_comments = FileToComments('all_comments.csv')
from gensim.models import Phrases

from gensim.models.phrases import Phraser



# Train Tokenizer on all comments

bigram = Phrases(all_comments, min_count=30, threshold=15)

bigram_phraser = Phraser(bigram) 
all_tokens = [bigram_phraser[comment] for comment in all_comments]



stops = set(stopwords.words("english"))



clean_all_tokens = []

for token in all_tokens:

    words = [w for w in token if not w in stops]

    clean_all_tokens += [words]

print('tokens cleaned')
#Pickle the tokens file for further use

import pickle

with open('tokenized_all_comments.pickle', 'wb') as filename:

    pickle.dump(clean_all_tokens, filename, protocol=pickle.HIGHEST_PROTOCOL)

print('files saved to tokenized_all_comments.pickle...')
#Load Pre-saved tokenized comments

with open('tokenized_all_comments.pickle', 'rb') as filename:

    all_comments = pickle.load(filename)

    

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\

    level=logging.INFO)





# Set values for various parameters

num_features = 300    # Word vector dimensionality                      

min_word_count = 20   # Minimum word count                        

num_workers = 16       # Number of threads to run in parallel

context = 10          # Context window size                                                                                    

downsampling = 1e-3   # Downsample setting for frequent words



# Initialize and train the model (this will take some time)

print("Training model...")

model = word2vec.Word2Vec(all_comments,

                          workers=num_workers,

                          size=num_features,

                          min_count = min_word_count,

                          window = context,

                          sample = downsampling

                         )



# init_sims will make the model much more memory-efficient.

model.init_sims(replace=True)

model_name = "%sfeatures_%sminwords_%scontext" % (num_features,min_word_count,context)

model.save(model_name)
# You can load the model later using this:

#from gensim.models import Word2Vec

#import gensim

#w2v_model = Word2Vec.load("300features_20minwords_10context")



# You can also retrain existing models by loading the features and retraining

# I'll probably publish another iteration in the next few days