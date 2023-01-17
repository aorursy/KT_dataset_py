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
from datetime import datetime

start = datetime.now()
print(f'Execution started at: {start.strftime("%m/%d/%Y, %H:%M:%S")}')

# Any results you write to the current directory are saved as output.
class Article:
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
articles_number = 10000      # to speed up computing, we will work on a smaller number than 50k. However, feel free to type a bigger one. 
articles = []
for index in range(articles_number):
    if index % (articles_number//5) == 0:
        print(f'{index/(articles_number)*100}% of files processed: {index}')
    articles.append(Article(all_paths[index]))
print('Files loading finished')    

articles[3].body_text

# In the first step of data preprocessing the punctation and numerical characters are removed.
# Besides that words under 3 letters are omitted as they don't carry any significant meaning.
for i in range(articles_number):
    articles[i].body_text = articles[i].body_text.replace('-',' ')
    articles[i].body_text = articles[i].body_text.translate(str.maketrans('','',string.punctuation + string.digits))
    articles[i].body_text = [w.lower() for w in articles[i].body_text.split() if len(w)>2 and w.isalpha()]
from nltk.corpus import stopwords 
noise_words = ['medrxiv','biorxiv','covid','sars','preprint','authorfunder','license','available','copyright','peer','granted','perpetuityis','display','coronavirus','doi','also']

for index in range(articles_number):    
    articles[index].body_text = [word for word in articles[index].body_text if word not in stopwords.words('english') + noise_words and word[:4] != 'http']  
    if index % (articles_number//5) == 0:
        print(f'{index/(articles_number)*100}% of files processsed: {index}')
print('Stopwords removal finished')
    

# Stemming using the nltk SnowballStemmer
from nltk.stem import SnowballStemmer

stemmer = SnowballStemmer('english')

for i in range(len(articles)):    
    articles[i].body_text = [stemmer.stem(word) for word in articles[i].body_text]
" ".join(articles[3].body_text)

# Tf-idf function
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# The text is already tokenized so the default tf-idf tokenizer is not needed here.
# It will be swapped with the dummy function below

def dummy_fun(gunwo):
    return gunwo

tfidf = TfidfVectorizer(analyzer='word',
                        tokenizer=dummy_fun,
                        preprocessor=dummy_fun,
                        token_pattern=None)

tfidf_matrix = coo_matrix(tfidf.fit_transform([article.body_text for article in articles]))
tfidf_csr = tfidf_matrix.tocsr()
tfidf_csc = tfidf_matrix.tocsc()

print('Matrix size: (articles, unique_tokens) = ' + str(tfidf_matrix.shape))
# In the second task there are 5 major fields in which we want to harvest information 
topics = ['Risk factors', 'Transmission dynamics', 'Severity of disease', 'Susceptibility of population', 'Mitigation measures']

# Listed below are the arbitrarily chosen keywords for each topic
init_keywords = {
 'Risk factors':['risk', 'factors', 'smoke', 'tobacco', 'cigarette', 'pneumonic', 'pulmonary', 'coexisting', 'coinfections', 'comorbidities', 'preexisting', 'chronic', 'neonates', 'mother', 'child', 'pregnancy', 'cancer', 'addiction', 'rich', 'poor', 'background', 'welfare', 'prosperity', 'immune'],
 'Transmission dynamics': ['reproductive', 'number', 'incubation', 'period', 'serial', 'interval', 'transmission', 'spread', 'environment', 'circumstances', 'respiratory', 'droplets'],
 'Severity of disease': ['fatality', 'risk', 'severe', 'hospitalize', 'mortality', 'death', 'rate', 'serious', 'mild'],
 'Susceptibility of population': ['susceptibility', 'receptivity', 'sensitivity', 'age', 'old', 'young', 'ill', 'cold'],
 'Mitigation measures': ['mitigate', 'measures', 'action', 'public', 'health', 'healthcare', 'reaction', 'counteraction', 'flatten', 'capacity', 'mask', 'gloves', 'soap', 'lockdown', 'wash', 'clean', 'sterile', 'prevent', 'slow', 'fast', 'block']}

keywords = init_keywords.copy()
# Stemming the keywords so they match the stemmed tokens from the tfidf vect

for topic in topics:    
    keywords[topic] = [stemmer.stem(word) for word in keywords[topic]]

# Remove those initial keywords that don't appear in articles
for topic in topics:    
    keywords[topic] = [word for word in keywords[topic] if word in tfidf.get_feature_names()]
    
# Getting indices of our keywords in the tfidf_array
keywords_indices = {}

for topic in topics:    
    keywords_indices[topic] = [tfidf.get_feature_names().index(word) for word in keywords[topic]]



def get_tfidf_value(article,keyword):
    start_index = tfidf_csr.indptr[article]
    end_index = tfidf_csr.indptr[article+1]
    for i in tfidf_csr.indices[start_index:end_index]:
        if tfidf_csr.indices[i] == keyword:
            return tfidf_csr.data[i]
    return 0.0

def evaluate_article(article, keywords):
    start_index = tfidf_csr.indptr[article] 
    end_index = tfidf_csr.indptr[article+1] 
    article_value = 0
    matching_indices = [i for i in range(start_index,end_index) if tfidf_csr.indices[i] in keywords]
    for i in matching_indices:
        article_value += tfidf_csr.data[i]
    return article_value    
 
def evaluate_keyword(articles, keyword):
    articles_indices = [article[0] for article in articles]
    start_index = tfidf_csc.indptr[keyword]
    end_index = tfidf_csc.indptr[keyword+1]
    keyword_value = 0
    matching_indices = [i for i in range(start_index,end_index) if tfidf_csc.indices[i] in articles_indices]
    for i in matching_indices:
        keyword_value += tfidf_csc.data[i]
    return keyword_value  
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
        for article in range(len(articles)):        
            article_value = evaluate_article(article, keywords_indices[topic])
            best_articles[topic] = insert(best_articles[topic],article,article_value)
    return best_articles
def evaluate_col(col):
    start_index = tfidf_csc.indptr[col]
    end_index = tfidf_csc.indptr[col+1]
    value = 0
    for i in tfidf_csc.data[start_index:end_index]:
        value += i
        
    return value
#Best articles contains the most informative articles in each topic
top_number = 20 # number of top articles that we would like to assign to each topic
best_articles = get_best_articles(keywords_indices)
# Extra kewords holds the most important keywords in each topic
extra_keywords = {topic: [] for topic in topics}
for topic in topics:
    for keyword in range(len(tfidf.get_feature_names())):
        keyword_value = evaluate_keyword(best_articles[topic],keyword)
        extra_keywords[topic] = insert(extra_keywords[topic],keyword,keyword_value)

        
top20_keywords = {}
for topic in topics:
    top20_keywords[topic] =  [tfidf.get_feature_names()[extra_keywords[topic][doc][0]] for doc in range(len(extra_keywords[topic]))]
    print(f'{topic}: {top20_keywords[topic][1:-1]}')

# Some of the extra_keywords don't hold much information as they appear in articles from various fields i.e: case, covid

to_remove = {topic: [] for topic in topics}

# Collect these keywords
for topic in topics:
    for keyword in extra_keywords[topic]:
        keyword_value = evaluate_col(keyword[0])/(0.3 * len(articles))
        if(keyword_value > keyword[1]/top_number):
            to_remove[topic].append(keyword)
    print(f'{topic}: { [tfidf.get_feature_names()[to_remove[topic][doc][0]] for doc in range(len(to_remove[topic]))] }')

for topic in topics: 
    extra_keywords[topic] = [keyword for keyword in extra_keywords[topic] if keyword not in to_remove[topic]]
    print(f'{topic}: { [tfidf.get_feature_names()[extra_keywords[topic][doc][0]] for doc in range(len(extra_keywords[topic]))] }')    

extra_keywords_indices = {}
for topic in topics:
    extra_keywords_indices[topic] = [word[0] for word in extra_keywords[topic]]
    
for topic in topics:
    keywords_indices[topic] += extra_keywords_indices[topic]
    keywords_indices[topic] = list(set(keywords_indices[topic]))   

new_articles = get_best_articles(keywords_indices)

keywords = {}
for topic in topics:
    keywords[topic] = {}
    for i in keywords_indices[topic]:
        keyword_value = 0
        for article in new_articles[topic]:
            keyword_value += get_tfidf_value(article[0],i)
        if keyword_value != 0.0:
            keywords[topic][tfidf.get_feature_names()[i]] = keyword_value


for topic in topics: 
    best_articles[topic] = [ [item[0], item[1]/len(init_keywords[topic])] for item in best_articles[topic]]
    new_articles[topic] = [ [item[0], item[1]/len(keywords_indices[topic])] for item in new_articles[topic]]

import seaborn as sns

def create_boxplot(articles):
    data = {}
    for topic in topics:
        data[topic] = [item[1] for item in articles[topic]]
    sns.set(palette='Blues_d', style="whitegrid")
    df = pd.DataFrame.from_dict(data)
    boxplot = df.boxplot(figsize=(16,8))
    
create_boxplot(best_articles)
for topic in topics: 
    print(len(init_keywords[topic]))
    print(len(keywords_indices[topic]))
create_boxplot(new_articles)
from wordcloud import WordCloud
import matplotlib.pyplot as plt

word_clouds = {}
for topic in topics:
    print(topic)
    word_clouds[topic] = WordCloud(background_color='white').generate_from_frequencies(keywords[topic])
    plt.figure(figsize=(16,8))
    plt.imshow(word_clouds[topic])
    plt.axis('off')
    plt.show()
def present_article(file_path):
    content = json.load(open(file_path))
    title = content['metadata']['title']
    body_text = ''

    print(f'\nTitle: {title}\n')
      
    for input in content['body_text']:
        body_text += input['text']
    print(f'Text: {body_text[:300]}')
   
    

for topic in topics:
    print(f'{topic}: \n')
    
    for i in range(3):
        present_article(articles[new_articles[topic][i][0]].paper_id)
end = datetime.now()
total = end - start
print(f'Execution finished at: {end.strftime("%m/%d/%Y, %H:%M:%S")} \nDuration: {total}')