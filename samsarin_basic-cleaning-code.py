import pandas as pd

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import word_tokenize

import re

from tqdm import tqdm

tqdm.pandas()
df = pd.read_csv('/kaggle/input/clean-data-with-all-columns/Clean Data with all columns.csv',encoding = 'latin-1')

df.head()
#df = df[['Consumer complaint narrative','Company response to consumer','Complaint ID']]

df.dropna(inplace = True)

df.drop_duplicates(subset = 'Consumer complaint narrative',inplace = True)

df.reset_index(drop = True,inplace = True)

df.head()
df.shape
def removal(text):

    text = re.sub('[^A-Za-z]',' ',text)

    text = re.sub('xxxx','',text)

    text = re.sub('xxx','',text)    

    text = re.sub('xx','',text)

    text = re.sub('xx\/xx\/\d+','',text)

    #text = re.sub('UNKNOWN   UNKNOWN','UNKNOWN',text)

    text = re.sub('\n',' ',text)

    text = re.sub(' +',' ',text)

    

    return text



stop_words = stopwords.words('english')

words = []

for i in tqdm(range(len(stop_words))):

    words.append(re.sub('[^A-Za-z]','',stop_words[i]))

    

stop_words = list(set(stop_words+words))



lem = WordNetLemmatizer()

pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
def cleaning(text):

    text = text.lower()

    text = pattern.sub(' ', text)

    text = removal(text)

    text = word_tokenize(text)

    text = [lem.lemmatize(w,'v') for w in text]

    text = ' '.join(text)

    text = re.sub(r'\b\w{1,3}\b','', text)

    text = re.sub(' +', ' ',text)

    return text
df['Total Clean'] = df['Consumer complaint narrative'].progress_apply(cleaning)
df.head()
df.to_csv('Clean Data with all columns.csv',index = False)