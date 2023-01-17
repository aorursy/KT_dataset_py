# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
!pip install langdetect
import os
import json
import nltk
import re
import numpy as np
import pandas as pd
import glob
from tqdm.notebook import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling 
from collections import Counter

# NLP
import re
from nltk.util import ngrams
from nltk.corpus import stopwords
from collections import Counter

#TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Wordcloud
from wordcloud import WordCloud, STOPWORDS

# Topic Modelling
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Visualization LDA
import pyLDAvis
import pyLDAvis.sklearn

# Language detection
from langdetect import DetectorFactory
from langdetect import detect
from langdetect import detect_langs
path = '../input/CORD-19-research-challenge'
df_meta_files = pd.read_csv(f'{path}/metadata.csv')
covid_filenames = glob.glob(f'{path}/**/*.json', recursive=True)
class ReadFile:
    
    def __init__(self, filename):
               
        with open(filename, 'r') as f:
            file = json.load(f)
            
        #Abstract
        try:            
            abstract = ''
            for par in file['abstract']:
                abstract += par['text']
        except:
            abstract = None
        
        #Body text
        body = ''
        for par in file['body_text']:
            body += par['text']
        
        #Bib entries
        bib_entries = []
        bib_titles = []
        for key, value in file['bib_entries'].items():
            try:
                DOI = value['other_ids']['DOI'][0]
            except:
                DOI = None

            bib_entries.append({"title": value["title"], \
                                "year": value["year"], \
                                "venue": value["venue"], \
                                "DOI": DOI})
            bib_titles.append(value["title"])
        
        self.paper_id = file['paper_id']
        self.title = file['metadata']['title']
        self.abstract = abstract
        self.body_text = body
        self.bib_entries = bib_entries
        self.bib_titles = bib_titles
def read_files(filenames):
    clean_files = []

    for filename in tqdm(filenames, position=0, leave=True):
        file = ReadFile(filename)
        
        clean_files.append({"paper_id": file.paper_id if file.paper_id != ''  else None, \
                        "title": file.title if file.title != ''  else None, \
                        "abstract": file.abstract if file.abstract != '' else None, \
                        "body_text": file.body_text if file.body_text != '' else None, \
                        "bib_entries": file.bib_entries, \
                        "bib_titles": file.bib_titles})

    df_clean_files = pd.DataFrame(clean_files)
    
    return df_clean_files
df_covid_files = read_files(covid_filenames)
#pandas_profiling.ProfileReport(df_meta_files)
#pandas_profiling.ProfileReport(df_covid_files[['title', 'abstract','body_text']])
print(f'There are {sum(df_meta_files.sha.value_counts()>1)} papers with do not have a distinct SHA')
df_meta_files[df_meta_files.groupby('sha')['sha'].transform('size') > 1].sort_values('sha')
#Check papers with the same name (sample)
df_covid_files[df_covid_files.title=='Original Article'].sample(5)
# Set seed 
# DetectorFactory.seed = 0

# languages = []
# for idx in tqdm(range(0, len(df_meta_files))):
    
#     # Extract title and abstract
#     title = df_meta_files.loc[idx]['title']
#     abstract = df_meta_files.loc[idx]['abstract']
    
#     try:
#         lang = detect(abstract[0:100]) #First check the abstract, take the first 200 words from the abstract
#         languages.append(lang)
#     except:
#         try:
#             lang = detect(title) #If there is no abstract try to detect the language from the title
#             languages.append(lang)
#         except:
#             languages.append("NA")
# Distribution of languages
# languages_count = Counter(languages)
# df_languages = pd.DataFrame\
#                         .from_dict(languages_count, orient='index')\
#                         .reset_index()\
#                         .rename(columns={'index':'language', 0:'count'})
# ax = sns.barplot(x="language", y="count", data=df_languages.sort_values("count", ascending=False))
# ax.set_title("Distribution of languages")
# plt.show()
# percentage_english = df_languages.loc[df_languages.language=='en','count'][0]/sum(df_languages['count'])

# print(f'The percentage of non-English papers is {round((1-percentage_english)*100, 2)}%')
# Drop non-English papers
df_meta_files['language'] = languages
df_meta_files = df_meta_files[df_meta_files.language=='en']

# Drop duplicates SHA
df_meta_without_sha = df_meta_files[df_meta_files.sha.isnull()]
df_meta_with_sha = df_meta_files[~df_meta_files.sha.isnull()]

df_meta_with_sha.drop_duplicates(subset=['sha'])

df_meta_files = pd.concat([df_meta_with_sha, df_meta_without_sha])

# Drop duplicates body text
df_covid_files.drop_duplicates(subset=['body_text'])
# Merge meta data with the JSON files
df_meta_files['id'] = df_meta_files['sha']
df_covid_files['id'] = df_covid_files['paper_id']
df = pd.merge(df_meta_files, df_covid_files[['id','body_text','bib_entries','bib_titles']], 
              how='left', on='id').drop(columns=['id'])
df = df[['sha','source_x','title','doi','abstract','publish_time','body_text','bib_entries','bib_titles']]
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
!pip install scispacy

import scispacy
import spacy
import en_core_sci_sm
from spacy.lang.en.stop_words import STOP_WORDS

nlp = en_core_sci_sm.load()

stopwords_spacy = list(STOP_WORDS)
# Parser
nlp = en_core_sci_sm.load(disable=['tagger', 'parser', 'ner', 'textcat'])

def spacy_tokenizer(doc):

    tokens = ''
    for token in nlp(doc):
        if not (token.like_num or token.is_punct or token.is_space) \
        and not token.lower_ in stopwords_spacy \
        and token.lemma_!='-pron-':
            tokens += f' {str(token.lemma_.lower())}'

    return tokens.strip()
# Data
df_titles = df.dropna(subset=['title']).reset_index()
df_abstracts = df.dropna(subset=['abstract']).reset_index()

all_titles = df_titles['title'].apply(spacy_tokenizer)
all_abstracts = df_abstracts['abstract'].apply(spacy_tokenizer)
def generate_wordcloud(text):

    wordcloud = WordCloud(max_words=1000,
                        background_color='white').generate(text)
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
generate_wordcloud("".join(all_titles))
generate_wordcloud("".join(all_abstracts))
# Vectorize text
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,
    min_df=0.1)

tfidf = tfidf_vectorizer.fit_transform(list(all_abstracts))
# LDA
lda_model = LatentDirichletAllocation(
    n_components=10,
    max_iter=10,
    learning_method='online',
    learning_offset=50,
    random_state=0)

lda_model = lda_model.fit(tfidf)
# Visualising topic models with pyLDAvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.sklearn.prepare(lda_model, tfidf, tfidf_vectorizer)
vis
def get_top_n_results(dataframe, vectorizer, ft_vectorizer, query, n=5) -> pd.DataFrame():
    """
    Returns a dataframe with the top 5 most similar papers
    
    input
    -----
    df: pd.DataFrame()
    vectorizer: TfidfVectorizer or CountVectorizer
    query: (str) The query you want to retrieve papers for 
    n: (int) The top number of papers you want to retrieve
    
    returns
    -------
    pd.Dataframe: Title, abstract and similarity score for a given top n
    """
    query_matrix = vectorizer.transform([spacy_tokenizer(query)])
    
    scores = cosine_similarity(ft_vectorizer, query_matrix).flatten()
    
    shas, titles, abstracts, similarity = [], [], [], []
    
    for idx in np.argsort(-scores)[:n]:
        shas.append(dataframe.loc[idx, 'sha'])
        titles.append(dataframe.loc[idx, 'title'])
        abstracts.append(dataframe.loc[idx, 'abstract'])
        similarity.append(round(scores[idx], 2))
        
    return pd.DataFrame({'title': titles, 'abstract': abstracts, 'similarity': similarity})
df_abstracts
tfidf = TfidfVectorizer(max_df=0.95, min_df=0.05)
tfidf_matrix = tfidf.fit_transform(list(all_abstracts))
questions = ["Range of incubation periods for the disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.",
"Prevalence of asymptomatic shedding and transmission (e.g., particularly children).",
"Seasonality of transmission.",
"Physical science of the coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).",
"Persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).",
"Persistence of virus on surfaces of different materials (e,g., copper, stainless steel, plastic).",
"Natural history of the virus and shedding of it from an infected person",
"Implementation of diagnostics and products to improve clinical processes",
"Disease models, including animal models for infection, disease and transmission",
"Tools and studies to monitor phenotypic change and potential adaptation of the virus",
"Immune response and immunity",
"Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings",
"Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings",
"Role of the environment in transmission"]
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[0])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[1])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[2])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[3])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[4])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[5])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[6])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[7])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[8])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[9])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[10])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[11])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[12])
get_top_n_results(df_abstracts, tfidf, tfidf_matrix, questions[13])