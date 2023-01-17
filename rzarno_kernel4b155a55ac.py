# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json

import nltk

nltk.download('stopwords')

from nltk.corpus import stopwords

from textblob import Word

from nltk.stem import PorterStemmer

st = PorterStemmer()



stop = stopwords.words('english')



counter_all = 0

counter_title = 0

counter_abstract = 0

counter_article = 0



related_articles = {

    'ethic': [],

    'education': [],

    'public health': []

}



search_words = ['ethic', 'public health', 'education']



def normalize(text: str):

    text = text.lower() #lowercase

    text = " ".join(x for x in text.split() if not x.isdigit())#remove digits

    text = " ".join(x for x in text.split() if x not in stop) #remove stopwords

    text =  " ".join([Word(word).lemmatize() for word in text.split()]) #lemmatize

    return text



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        file_path = os.path.join(dirname, filename)

        print(file_path)

        for search_word in search_words:

            if filename.endswith(".json"):

                print('\n%s\n%s\n%s\n%s' % (counter_all, counter_title, counter_abstract, counter_article))

                #print(related_articles)

                counter_all += 1

                found = False

                with open(file_path, 'r') as f:

                    article_dict = json.load(f)

                    title = normalize(article_dict['metadata']['title'])

                    paragraphs = []

                    if search_word in title:

                        counter_title += 1

                        found = True

                    try:

                        if search_word in normalize(article_dict['abstract'][0]['text']):

                            counter_abstract += 1

                            found = True

                            #print('abstract:' + article_dict['abstract'][0]['text'] + '\n\n\n\n')

                        for index, text in enumerate(article_dict['body_text']):

                            if search_word in normalize(text['text']):

                                counter_article += 1

                                #print('text:' + text['text'] + '\n\n\n\n')

                                found = True

                                paragraphs.append(index)

                    except TypeError:

                        continue

                    except IndexError:                

                        continue

                    except KeyError:                

                        continue

                if found:

                    related_articles[search_word].append([article_dict['metadata']['title'], paragraphs])

                continue

            else:

                continue

                        

        

# Any results you write to the current directory are saved as output.

import json

json = json.dumps(related_articles)

f = open("results.json","w")

f.write(json)

f.close()