# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob
import json
import nltk

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import string
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
class File:
    def __init__(self, file_path):
        content = json.load(open(file_path))
        self.paper_id = file_path
        self.body_text = ''
        for input in content['body_text']:
            self.body_text += input['text']    
root_path = '/kaggle/input/CORD-19-research-challenge'
sub_dirs = ['/biorxiv_medrxiv/biorxiv_medrxiv/pdf_json',
            '/comm_use_subset/comm_use_subset/pdf_json',
            '/comm_use_subset/comm_use_subset/pmc_json',
            '/custom_license/custom_license/pdf_json',
            '/custom_license/custom_license/pmc_json',
            '/noncomm_use_subset/noncomm_use_subset/pdf_json',
            '/noncomm_use_subset/noncomm_use_subset/pmc_json'
           ]
all_paths = []

for sub_dir in sub_dirs:
    all_paths.append(glob.glob(f'{root_path}{sub_dir}/*.json'))

# merging sublists into a single list
all_paths = [item for sublist in all_paths for item in sublist]
len(all_paths)

#Loading articles
articles = []
for iterator, path in enumerate(all_paths):
    if iterator % (len(all_paths)//10) == 0:
        print(f'{iterator} files processed')
    articles.append(File(path))

#We will use only the first 500 articles to speed up the computation process
temp_articles = [article.body_text for article in articles[:500]]
temp_articles[0]
temp_articles[4]
#In the first step of data preprocessing the punctation and numerical characters are removed.
#Besides that words under 3 characters are omitted as they don't carry any significant meaning
for i in range(len(temp_articles)):
    temp_articles[i] = temp_articles[i].replace('-',' ')
    temp_articles[i] = temp_articles[i].translate(str.maketrans('','',string.punctuation + string.digits))
    temp_articles[i] = [w.lower() for w in temp_articles[i].split() if len(w)>2 and w.isalpha()]
    
#Removal of stopwords and http links
from nltk.corpus import stopwords 

for i in range(len(temp_articles)):    
    temp_articles[i] = [word for word in temp_articles[i] if word not in stopwords.words('english') and word[:4] != 'http']
    for j in range(len(temp_articles[i])):
        if temp_articles[i][j] == "license":
            temp_articles[i] = temp_articles[i][:j-1]
            break
    
print(temp_articles[0])
len(temp_articles[0])
#Stemming using the nltk SnowballStemmer
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

for i in range(len(temp_articles)):    
    temp_articles[i] = [stemmer.stem(word) for word in temp_articles[i]]
    
len(temp_articles[0])


#Tf-idf function
from sklearn.feature_extraction.text import TfidfVectorizer

#The text is already tokenized so the default tf-idf tokenizer is not needed here.
#It will be swapped with the dummy function below
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(analyzer='word',
                        tokenizer=dummy_fun,
                        preprocessor=dummy_fun,
                        token_pattern=None)

tfidf_matrix = tfidf.fit_transform(temp_articles).toarray()
#In the second task there are 5 major fields in which we want to harvest information 
topics = ['Risk factors', 'Transmission dynamics', 'Severity of disease', 'Susceptibility of population', 'Mitigation measures']

#Listed below are the arbitrary chosen keywords for each topic
keywords = {
 'Risk factors':['risk', 'factors', 'smoke', 'tobacco', 'cigarette', 'pneumonic', 'pulmonary', 'coexisting', 'coinfections', 'comorbidities', 'preexisting', 'chronic', 'neonates', 'mother', 'child', 'pregnancy', 'cancer', 'addiction', 'rich', 'poor', 'background', 'welfare', 'prosperity', 'immune'],
 'Transmission dynamics': ['reproductive', 'number', 'incubation', 'period', 'serial', 'interval', 'transmission', 'spread', 'environment', 'circumstances', 'respiratory', 'droplets'],
 'Severity of disease': ['fatality', 'risk', 'severe', 'hospitalize', 'mortality', 'death', 'rate', 'serious', 'mild'],
 'Susceptibility of population': ['susceptibility', 'receptivity', 'sensitivity', 'age', 'old', 'young', 'ill', 'cold'],
 'Mitigation measures': ['mitigate', 'measures', 'action', 'public', 'health', 'healthcare', 'reaction', 'counteraction', 'flatten', 'capacity', 'mask', 'gloves', 'soap', 'lockdown', 'wash', 'clean', 'sterile', 'prevent', 'slow', 'fast', 'block']}

#Stemming the keywords so they match the stemmed tokens from the tfidf vect

for topic in topics:    
    keywords[topic] = [stemmer.stem(word) for word in keywords[topic]]

# Remove those initial keywords that don't appear in articles
for topic in topics:    
    keywords[topic] = [word for word in keywords[topic] if word in tfidf.get_feature_names()]
    
#Getting indices of our keywords in the tfidf_array
keywords_indices = {}

for topic in topics:    
    keywords_indices[topic] = [tfidf.get_feature_names().index(word) for word in keywords[topic]]

print(keywords_indices)


def evaluate_article(article, keywords):
    article_value = 0
    for word in keywords:
        article_value += tfidf_matrix[article][word]
    return article_value      
#Insert adds an item into a sorted list
def insert(list, doc, doc_value): 
    global top_number
    if len(list) < top_number: 
            list.append([doc,doc_value])
            return list
    for index in range(top_number):
        if list[index][1] <= doc_value:
            list = list[:index] + [[doc,doc_value]] + list[index:]
            return list[:top_number]
    return list
def get_best_articles(keywords_indices):
    best_articles = {topic: [] for topic in topics}
    for topic in topics:
        for article in range(len(tfidf_matrix)):        
            article_value = evaluate_article(article, keywords_indices[topic])
            best_articles[topic] = insert(best_articles[topic],article,article_value)
    return best_articles
#Best articles contains the most informative articles in each topic
top_number = 10 # number of top articles that we would like to assign to each topic
best_articles = get_best_articles(keywords_indices)
best_articles
#Extra kewords holds the most important keywords in each topic
extra_keywords = {topic: [] for topic in topics}
for topic in topics:
    for keyword in range(len(tfidf.get_feature_names())):
        keyword_value = 0
        for doc in best_articles[topic]:
            keyword_value += tfidf_matrix[doc[0]][keyword]
        extra_keywords[topic] = insert(extra_keywords[topic],keyword,keyword_value)
extra_keywords
current_keywords = {}
top20_keywords = {}
for topic in topics:
    current_keywords[topic] = [keyword for keyword in extra_keywords[topic] if keyword[0] in keywords_indices[topic]]
    top20_keywords[topic] =  [tfidf.get_feature_names()[extra_keywords[topic][doc][0]] for doc in range(len(extra_keywords[topic]))]
top20_keywords
#Some of the extra_keywords don't hold much information as they appear in articles from various fields i.e: case, covid

to_remove = {topic: [] for topic in topics}
for topic in topics:
    for keyword in extra_keywords[topic]:
        keyword_value = 0
        for doc in range(len(tfidf_matrix)):
            keyword_value += tfidf_matrix[doc][keyword[0]]
        keyword_value /= (0.5 * len(tfidf_matrix))
        if(keyword_value > keyword[1]/top_number):
            to_remove[topic].append(keyword)
    print(topic + ':')
    print( [' ' + tfidf.get_feature_names()[to_remove[topic][doc][0]] for doc in range(len(to_remove[topic]))])

for topic in topics: 
    extra_keywords[topic] = [keyword for keyword in extra_keywords[topic] if keyword not in to_remove[topic]]

extra_keywords_indices = {}
for topic in topics:
    extra_keywords_indices[topic] = [word[0] for word in extra_keywords[topic]]
    
for topic in topics:
    keywords_indices[topic] += extra_keywords_indices[topic]
    keywords_indices[topic] = list(set(keywords_indices[topic]))
    

new_articles = get_best_articles(keywords_indices)

for topic in topics:
    top20_keywords[topic] =  [tfidf.get_feature_names()[extra_keywords[topic][doc][0]] for doc in range(len(extra_keywords[topic]))]
top20_keywords
import seaborn as sns
data = {}
for topic in topics:
    data[topic] = [item[1] for item in best_articles[topic]]

sns.set(palette='Blues_d', style="whitegrid")
df = pd.DataFrame.from_dict(data)
boxplot = df.boxplot(figsize=(16,8))

def present_article(file_path):
    content = json.load(open(file_path))
    title = content['metadata']['title']
    authors = []
    abstract = ''
    for author in content['metadata']['authors']:
        authors.append(author['first'] +' '+ author['last'])
    for paragraph in content['abstract']:
        abstract += '\n\n' + paragraph['text']
    print(f'Title: {title}')
    for author in authors:
        print(f'{author}')
    print(abstract)
    
present_article(articles[0].file_path)

from wordcloud import WordCloud

word_clouds = {}
for topic in topics:
    word_clouds[cluster] = WordCloud().generate_from_frequencies()
