# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input/amazon-fine-food-reviews'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import sqlite3
import pandas as pd
import numpy as np
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from nltk.stem.porter import PorterStemmer

import re
# Tutorial about Python regular expressions: https://pymotw.com/2/re/
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle

from tqdm import tqdm
import os
con = sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite') 

filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews 
                                     WHERE Score != 3 LIMIT 5000""", con) 

# filtered_data.head()

# Rating Score>3 a positive rating, and score<3 a negative rating.
def partition(x):
    if x < 3:
        return 0
    return 1

actual_score = filtered_data['Score']
PosNeg = actual_score.map(partition)
filtered_data['Score'] = PosNeg

print("Data-Points in data", filtered_data.shape)
filtered_data.head(3)


display = pd.read_sql_query("""

select * from Reviews
where Score!=3 AND UserId="AR5J8UI46CURR"
order by ProductId
""", con)

display.head()
# Sorting data according to ProductId in Asc. Order
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, 
                                       kind='quicksort', na_position='last')
# drop duplicate entries
final = sorted_data.drop_duplicates(subset={"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace=False)
final.shape
display = pd.read_sql_query("""select * from Reviews
                            where Score!=3 AND Id=44737 OR Id=64422
                            order by ProductId""", con)
display.head()
final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
print(final.shape)

# number of +ve and -ve reviews 
final['Score'].value_counts()
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer


stop = set(stopwords.words('english'))
sno = nltk.stem.SnowballStemmer('english')


print(stop)
print("="*50)

print(sno.stem('tasty'))
# FUnction to clean HTML
def cleanHTML(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

# Function to clean Punctuation
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned

i = 0
str= ' '

final_string=[]
all_positive_words= []
all_negative_words=[]

s=''

for sent in final['Text'].values:
    filtered_sentence=[]
    
    
    sent = cleanHTML(sent)
    for word in sent.split():
        for cleaned_words in cleanPunc(word).split():
            
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if(cleaned_words.lower() not in stop):
                    s = (sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    
                    
                    if(final['Score'].values)[i] == 'positive':
                        all_positive_words.appned(s)
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.appned(s)
                else:
                    continue
            else:
                continue
                
                

                        
    str1 = b" ".join(filtered_sentence)
        
    final_string.append(str1)
    i+=1
# adding a Cleaned_text column

final['CleanedText'] = final_string
final.head()

# storing this in table

conn = sqlite3.connect('final.sqlite')
c=conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace')

# Bag_of_Words
count_vec = CountVectorizer() # in Scikit 
final_counts = count_vec.fit_transform(final['CleanedText'].values)
# type(final_counts)
print(count_vec.get_feature_names()[:100])
final_counts.get_shape()

count_vec = CountVectorizer(ngram_range=(1, 2))
final_bigram_counts = count_vec.fit_transform(final['Text'].values)
final_bigram_counts.get_shape()
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))
final_tf_idf = tf_idf_vect.fit_transform(final['Text'].values)
final_tf_idf.get_shape()
feature = tf_idf_vect.get_feature_names()
len(feature)
feature[140000:140010]
print(final_tf_idf[3,:].toarray()[0])
