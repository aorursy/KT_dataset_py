import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import random
from IPython.display import display, HTML
import email
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer 
from nltk import FreqDist

spam_path = '../input/spam_2/spam_2/'
easy_ham_path = '../input/easy_ham/easy_ham/'
hard_ham_path = '../input/hard_ham/hard_ham/'
# label messagges according to folder
email_files = {'spam':     os.listdir(spam_path),
               'easy_ham': os.listdir(easy_ham_path),
               'hard_ham': os.listdir(hard_ham_path)
              }    
email_dict = {}
invalid_list = []
vocabulary = FreqDist()
# take 20 random samples from spam path and create a dictionary {email code: word_count dictionary} 
# create also the overall FreqDist object summing up each file FreqDist --> the vocabulary
path = spam_path
file_list = random.sample(os.listdir(path), 100)
for filename in file_list:
    try:
        file = open(path + filename,'r',errors='ignore')
        content = file.read()

        msg = email.message_from_string(content)
        if msg.is_multipart():
            body = []
            for payload in msg.get_payload():
                # if payload.is_multipart(): ...
                body.append(payload.get_payload())
            body = ' '.join(body)

        else:
            body = msg.get_payload()

        body_pp = body.lower()
        body_pp = re.sub(r"<[^<>]+>", " html ", body_pp)
        body_pp = re.sub(r"[0-9]+", " number ", body_pp)
        body_pp = re.sub(r"(http|https)://[^\s]*", ' httpaddr ', body_pp)
        body_pp = re.sub(r"[^\s]+@[^\s]+", ' emailaddr ', body_pp)
        body_pp = re.sub(r"[$]+", ' dollar ', body_pp)
        body_pp = re.sub(r"[^a-zA-Z0-9]",' ', body_pp)
        body_token = word_tokenize(body_pp)
        stemmer = PorterStemmer()
        body_stem = [stemmer.stem(token) for token in body_token]
        body_freq_dict = FreqDist(body_stem)
        # add email dictionary entry
        email_dict[filename] = body_freq_dict
        # update the vocabulary
        vocabulary = vocabulary + body_freq_dict
    except:
      invalid_list.append(filename)  

print('Number of succesfully processed emails: %d' %len(email_dict))    
print('Number of not processed emails: %d' %len(invalid_list))
print('Number of Vocabulary entries: %d' %vocabulary.N())
print('Most Common word \n')
print(vocabulary.most_common()[:10])