import os

import glob

import json

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.decomposition import LatentDirichletAllocation

from tqdm.notebook import tqdm
all_json_paths = glob.glob(f'/kaggle/input/CORD-19-research-challenge/comm_use_subset/comm_use_subset/*.json', recursive=True)

len(all_json_paths)
#source:https://www.kaggle.com/amogh05/cord-19-eda-question-topic-modeling-starter

#add more vars as required



class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.title = content['metadata']['title']

            self.abstract = []

            self.body_text = []

            self.biblio = []

            self.biblio_doi = []

            self.img_tables = []

            self.back_matter = []

            

            

            # Abstract

            for entry in content['abstract']:

                self.abstract.append(entry['text'])

            self.abstract = '\n'.join(self.abstract)

            

            # Body text

            for entry in content['body_text']:

                self.body_text.append(entry['text'])          

            self.body_text = '\n'.join(self.body_text)

            

            # bibliography

            for bib_id, details in content['bib_entries'].items():

                self.biblio.append(details['title'])

                self.biblio_doi.append(details['other_ids'])

            self.biblio = '\n'.join(self.biblio)

            #self.biblio_doi = '\n'.join(self.biblio_doi)

            

            #img and table references

            for ref_id,details in content['ref_entries'].items():

                self.img_tables.append(details['text'])

            self.img_tables = '\n'.join(self.img_tables)

            

            #back_matter

            for entry in content['back_matter']:

                self.back_matter.append(entry['text'])

            self.back_matter = '\n'.join(self.back_matter)

            

    def __repr__(self):

        return f'{self.paper_id}:{self.title}-{self.abstract}... {self.body_text}...{self.biblio}...{self.img_tables}...{self.back_matter}'

        

    

dict_ = {'paper_id': [],'title':[], 'abstract': [], 'body_text': [],'biblio':[],'bidoi':[],'img_tables':[]}

for idx, entry in enumerate(all_json_paths):

    if idx % (len(all_json_paths) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json_paths)}')

    #print(entry)

    content = FileReader(entry)

    dict_['paper_id'].append(content.paper_id)

    dict_['title'].append(content.title)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    dict_['biblio'].append(content.biblio)   

    dict_['img_tables'].append(content.img_tables)  

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text','biblio','img_tables'])

df_covid.head()



#identify dups

df_covid.describe(include='all')



df_covid.drop_duplicates(['abstract'], inplace=True)

df_covid.describe(include='all')
df_covid['all_text'] = df_covid['abstract'] + '' + df_covid['body_text'] 
#This approach does not work well:  defining a list is better

import nltk 

from nltk.corpus import wordnet 

synonyms = [] 



  

for syn in wordnet.synsets('exposure'): 

    for l in syn.lemmas(): 

        synonyms.append(l.name()) 

        if l.antonyms(): 

            antonyms.append(l.antonyms()[0].name())

print(set(synonyms))
#defining a list better

stage_syn_list = ['exposure','vulnerability','vulnerable'] 
disease_stage_list = ['exposure' ,'acquisition' ,'progression', 'development' ,'complications' ,'fatality', 'disability']
def filterByStage(text,stage_syn_list):

    paper_list =[]

    

    for idx_num,row in text.iterrows():

        for stage in stage_syn_list:

            stage_found = False

            if stage in row.all_text.split():

                stage_found = True

            else:

                pass 

        if stage_found==True:

            paper_list.append(row.all_text)

    return paper_list
stage_dict = {}



stage = disease_stage_list[0]



stage_dict[stage] = filterByStage(df_covid,stage_syn_list)
#for later ease while searching for relevant papers

exposure = pd.DataFrame(stage_dict[stage])
!pip install scispacy scipy

!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz

!pip install tqdm -U

!pip install spacy-langdetect
import spacy

import en_core_sci_lg

nlp = en_core_sci_lg.load()



# We also need to detect language, or else we'll be parsing non-english text 

# as if it were English. 

from spacy_langdetect import LanguageDetector

nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)



nlp.max_length=2000000



# New stop words list 

customize_stop_words = [

    'doi', 'preprint', 'copyright', 'peer', 'reviewed', 'org', 'https', 'et', 'al', 'author', 'figure', 

    'rights', 'reserved', 'permission', 'used', 'using', 'biorxiv', 'fig', 'fig.', 'al.',

    'di', 'la', 'il', 'del', 'le', 'della', 'dei', 'delle', 'una', 'da',  'dell',  'non', 'si'

]



# Mark them as stop words

for w in customize_stop_words:

    nlp.vocab[w].is_stop = True
def spacy_tokenizer(sentence):

    return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)] 

    # remove numbers (e.g. from references [1], etc.)
tf_vectorizer = CountVectorizer(tokenizer = spacy_tokenizer, max_features=800000) 

tf = tf_vectorizer.fit_transform(tqdm(stage_dict[stage]))



print(tf.shape)



import joblib

joblib.dump(tf_vectorizer, '/kaggle/working/tf_vectorizer.csv')

joblib.dump(tf, '/kaggle/working/tf.csv')
lda_tf = LatentDirichletAllocation(n_components=50, random_state=0)

lda_tf.fit(tf)

joblib.dump(lda_tf, '/kaggle/working/lda.csv')
tfidf_feature_names = tf_vectorizer.get_feature_names()



def print_top_words(model, feature_names, n_top_words):

    for topic_idx, topic in enumerate(model.components_):

        message = "\nTopic #%d: " % topic_idx

        message += " ".join([feature_names[i]

                             for i in topic.argsort()[:-n_top_words - 1:-1]])

        print(message)

    print()

    

print_top_words(lda_tf, tfidf_feature_names, 25)
topic_dist = pd.DataFrame(lda_tf.transform(tf))

topic_dist.to_csv('/kaggle/working/topic_dist.csv', index=False)
topic_dist.head()
#get most similar paper

from scipy.spatial import distance

def get_k_nearest_docs(doc_dist, k=5, lower=1950, upper=2020, only_covid19=False, get_dist=False):

    '''

    doc_dist: topic distribution (sums to 1) of one article

    

    Returns the index of the k nearest articles (as by Jensen–Shannon divergence in topic space). 

    '''

    

    #relevant_time = df.publish_year.between(lower, upper)

    

   # if only_covid19:

   #     is_covid19_article = df.body_text.str.contains('COVID-19|SARS-CoV-2|2019-nCov|SARS Coronavirus 2|2019 Novel Coronavirus') #TODO: move outside

   #     topic_dist_temp = topic_dist[relevant_time & is_covid19_article]

   #     

   # else:

    #    topic_dist_temp = topic_dist[relevant_time]

    

    distances = topic_dist.apply(lambda x: distance.jensenshannon(x, doc_dist), axis=1)

    k_nearest = distances[distances != 0].nsmallest(n=k).index

    

    if get_dist:

        k_distances = distances[distances != 0].nsmallest(n=k)

        return k_nearest, k_distances

    else:

        return k_nearest

    

#d = get_k_nearest_docs(topic_dist[1].iloc[0],k=10)
def relevant_articles(df,tasks, k=3, lower=1950, upper=2020, only_covid19=False):

    tasks = [tasks] if type(tasks) is str else tasks 

    

    tasks_tf = tf_vectorizer.transform(tasks)

    tasks_topic_dist = pd.DataFrame(lda_tf.transform(tasks_tf))



    for index, bullet in enumerate(tasks):

        print(bullet)

        recommended = get_k_nearest_docs(tasks_topic_dist.iloc[index], k, lower, upper, only_covid19)

        print(list(recommended))

        recommended = df.iloc[recommended] #stage_dict[stage][','.join(list(recommended))]#

    return recommended
task = ['exposure']

relevant_articles(exposure,task,k=10) #k is the number of relevant articles