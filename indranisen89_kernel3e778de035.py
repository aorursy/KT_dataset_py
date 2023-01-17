# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from textblob import TextBlob, Word, Blobber


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

reviews = pd.read_csv("../input/consumer-reviews-of-amazon-products/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv", index_col=0)


print("Setup complete.")
reviews.head()
df = pd.DataFrame(reviews, columns = ['keys', 'name','reviews.title','reviews.rating','reviews.text', 'reviews.username',])

#df.drop(df.index, inplace=True)


# df
renamed = df.rename(columns={'keys': 'key', 
                             'reviews.rating':'label', 
                             'name':'product_id',
                             'reviews.username':'reviewer_id', 
                             'reviews.text':'review_text', 
                             'reviews.title':'review_summary'})
# renamed


renamed.head()

#Text cleaning tecniques
renamed['id_value'] = range(1, len(df.index)+1)

renamed['id_value'].is_unique
renamed = renamed.set_index('id_value')
print(renamed.loc[1, 'review_text'][-50:])


# Convert all columns data into lower case
# Remove stop words
# Remove Punctuation

renamed = renamed.replace('[^a-zA-Z]+', ' ', regex=True)
stop = stopwords.words('english')

renamed = renamed.applymap(lambda x: " ".join(str(x).lower() for x in str(x).split() if x not in stop))

#Generate a series which displays the frequency of each word and output the data to review it manually
common_words = pd.Series(' '.join(renamed.stack()).split()).value_counts()
common_words[:20]

#After reviewing we decide to remove the top 10 most frequent words
renamed = renamed.applymap(lambda x: " ".join(x for x in x.split() if x not in common_words[:10]))
 

rare_words = pd.Series(' '.join(renamed.stack()).split()).value_counts()
rare_words[-20:]

#After reviewing we decide to remove the 10 most infrequent words
renamed = renamed.applymap(lambda x: " ".join(x for x in x.split() if x not in rare_words[-10:]))

#Tokenization
renamed = renamed.applymap(word_tokenize)
renamed

 



#Stemming    
st = PorterStemmer()
renamed = renamed.applymap(lambda x: [st.stem(word) for word in x])

renamed



