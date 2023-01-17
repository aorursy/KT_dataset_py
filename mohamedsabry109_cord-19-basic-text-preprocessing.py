# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import gc
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Any results you write to the current directory are saved as output.
root_path = '/kaggle/input/CORD-19-research-challenge/'
metadata_path = f'{root_path}/metadata.csv'
meta_df = pd.read_csv(metadata_path, dtype={
    'pubmed_id': str,
    'Microsoft Academic Paper ID': str, 
    'doi': str
})

meta_df.head()

all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
# see how many papers are there
len(all_json)
class FileReader:
    def __init__(self, file_path):
        with open(file_path) as file:
            content = json.load(file)
            print(content.keys())
            self.paper_id = content['paper_id']
            self.abstract = []
            self.body_text = []
            # Abstract
            try :
                for entry in content['abstract']:
                    self.abstract.append(entry['text'])
            except Exception as e:
                pass
            # Body text
            for entry in content['body_text']:
                self.body_text.append(entry['text'])
            self.abstract = '\n'.join(self.abstract)
            self.body_text = '\n'.join(self.body_text)

                
    def __repr__(self):
        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'
first_row = FileReader(all_json[0])
print(first_row)

def get_breaks(content, length):
    data = ""
    words = content.split(' ')
    total_chars = 0

    # add break every length characters
    for i in range(len(words)):
        total_chars += len(words[i])
        if total_chars > length:
            data = data + "<br>" + words[i]
            total_chars = 0
        else:
            data = data + " " + words[i]
    return data


dict_ = {'paper_id': [], 'abstract': [], 'body_text': [],'publish_year':[],'url':[],'WHO #Covidence':[] ,'license':[],'authors': [], 'title': [], 'journal': [], 'abstract_summary': []}
for idx, entry in enumerate(all_json[:10000]):
    if idx % (len(all_json) // 100) == 0:
        print(f'Processing index: {idx} of {len(all_json)}')
    content = FileReader(entry)
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    # no metadata, skip this paper
    if len(meta_data) == 0:
        continue
    
    dict_['paper_id'].append(content.paper_id)
    dict_['abstract'].append(content.abstract)
    dict_['body_text'].append(content.body_text)
    dict_['publish_year'].append(pd.DatetimeIndex(meta_data['publish_time']).year.values[0])   
    dict_['url'].append(meta_data['url'].values[0])
    dict_['WHO #Covidence'].append(meta_data['WHO #Covidence'].values[0])
    dict_['license'].append(meta_data['license'].values[0])
    # also create a column for the summary of abstract to be used in a plot
    if len(content.abstract) == 0: 
        # no abstract provided
        dict_['abstract_summary'].append("Not provided.")
    elif len(content.abstract.split(' ')) > 100:
        # abstract provided is too long for plot, take first 300 words append with ...
        info = content.abstract.split(' ')[:100]
        summary = get_breaks(' '.join(info), 40)
        dict_['abstract_summary'].append(summary + "...")
    else:
        # abstract is short enough
        summary = get_breaks(content.abstract, 40)
        dict_['abstract_summary'].append(summary)
        
    # get metadata information
    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]
    
    try:
        # if more than one author
        authors = meta_data['authors'].values[0].split(';')
        if len(authors) > 2:
            # more than 2 authors, may be problem when plotting, so take first 2 append with ...
            dict_['authors'].append(". ".join(authors[:2]) + "...")
        else:
            # authors will fit in plot
            dict_['authors'].append(". ".join(authors))
    except Exception as e:
        # if only one author - or Null valie
        dict_['authors'].append(meta_data['authors'].values[0])
    
    # add the title information, add breaks when needed
    try:
        title = get_breaks(meta_data['title'].values[0], 40)
        dict_['title'].append(title)
    # if title was not provided
    except Exception as e:
        dict_['title'].append(meta_data['title'].values[0])
    
    # add the journal information
    dict_['journal'].append(meta_data['journal'].values[0])
    
df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'publish_year','url','license','WHO #Covidence','body_text', 'authors', 'title', 'journal', 'abstract_summary'])
df_covid.head()
df_covid['word_count'] = df_covid['body_text'].apply(lambda x: len(str(x).split(" ")))
df_covid['char_count'] = df_covid['body_text'].str.len() ## this also includes spaces
df_covid.head()
def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))

df_covid['avg_word'] = df_covid['body_text'].apply(lambda x: avg_word(x))
df_covid[['body_text','avg_word']].head()
from nltk.corpus import stopwords
#stop is a list of all stop words in english, we might need to add stop words of other languages, we will see
stop = stopwords.words('english')
#calculating number of stop words presents in body text of each paper
df_covid['stopwords'] = df_covid['body_text'].apply(lambda x: len([x for x in x.split() if x in stop]))
df_covid[['body_text','stopwords']].head()
df_covid['numerics'] = df_covid['body_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
df_covid[['body_text','numerics']].head()
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if not x.isdigit()))
df_covid['body_text'].head()
df_covid['body_text'] = df_covid['body_text'].str.replace('[^\w\s]','')
df_covid['body_text'].head()
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str

df_covid['body_text'] = df_covid['body_text'].apply(lambda x: lower_case(x))
df_covid['abstract'] = df_covid['abstract'].apply(lambda x: lower_case(x))
df_covid['body_text'].head()
from nltk.corpus import stopwords
stop = stopwords.words('english')
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
df_covid['body_text'].head()
freq = pd.Series(' '.join(df_covid['body_text']).split()).value_counts()[:10]
freq
freq = list(freq.index)
freq
freq = ['the', 'et','al','in','also']
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
df_covid['body_text'].head()
freq = pd.Series(' '.join(df_covid['body_text']).split()).value_counts()[:10]
freq
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#get most 100 frequent words
freq = pd.Series(' '.join(df_covid['body_text']).split()).value_counts()[:100]
wordcloud = WordCloud().generate_from_frequencies(freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
stop_words = ["fig","figure", "et", "al", "table",  
        "data", "analysis", "analyze", "study",  
        "method", "result", "conclusion", "author",  
        "find", "found", "show", "perform",  
        "demonstrate", "evaluate", "discuss", "google", "scholar",   
        "pubmed",  "web", "science", "crossref", "supplementary", "A" ,"this" ,"that"]
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
df_covid['body_text'].head()
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#get most 100 frequent words
freq = pd.Series(' '.join(df_covid['body_text']).split()).value_counts()[:100]
wordcloud = WordCloud().generate_from_frequencies(freq)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
freq = pd.Series(' '.join(df_covid['body_text']).split()).value_counts()[-30:]
freq
# freq = list(freq.index)
# df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
# df_covid['body_text'].head()
from textblob import TextBlob
TextBlob(df_covid['body_text'][1]).words
from nltk.stem import PorterStemmer
st = PorterStemmer()
df_covid['body_text'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
df_covid['body_text'][:5].head()
from textblob import Word
df_covid['body_text'] = df_covid['body_text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
df_covid['body_text'].head()
TextBlob(df_covid['body_text'][0]).ngrams(2)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
text_vect = tfidf.fit_transform(df_covid['body_text'])

print(text_vect[0])
