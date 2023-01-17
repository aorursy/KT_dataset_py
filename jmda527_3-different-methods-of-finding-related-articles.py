import os
import gc
import re
import nltk
import json
import spacy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import spatial
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models.word2vec import Word2Vec
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

tqdm.pandas()
# nltk.download('stopwords')
warnings.filterwarnings("ignore")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
BASEDIR = '/kaggle/input/CORD-19-research-challenge/'
#1.read csv
df = pd.read_csv(BASEDIR+'metadata.csv')
JSON_DIR = BASEDIR + 'document_parses/'
#2.find path
# path_list_pdf = ['comm_use_subset/comm_use_subset/pdf_json/',
#              'noncomm_use_subset/noncomm_use_subset/pdf_json/',
#              'custom_license/custom_license/pdf_json/',
#              'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/',
#              'arxiv/arxiv/pdf_json/']
# path_list_pmc = ['comm_use_subset/comm_use_subset/pmc_json/',
#              'noncomm_use_subset/noncomm_use_subset/pmc_json/',
#              'custom_license/custom_license/pmc_json/',
#              'biorxiv_medrxiv/biorxiv_medrxiv/pmc_json/']

# def find_path_sha(filename):
#     for i in path_list_pdf:
#         path = BASEDIR+i+'{}.json'.format(filename)
#         if os.path.exists(path):
#             return path
#     return None

# def find_path_pmc(filename):
#     for i in path_list_pmc:
#         path = BASEDIR+i+'{}.xml.json'.format(filename)
#         if os.path.exists(path):
#             return path
#     return None
def find_path_sha(filename):
    path = JSON_DIR+'pdf_json/'+'{}.json'.format(filename)
    if os.path.exists(path):
        return path
    return None

def find_path_pmc(filename):
    path = JSON_DIR+'pmc_json/'+'{}.xml.json'.format(filename)
    if os.path.exists(path):
        return path
    return None

#3.delete rows without json file
df['sha'] = df['sha'].fillna('')
df['pmcid'] = df['pmcid'].fillna('')
df['file_path'] = df['sha'].progress_apply(find_path_sha)
df['file_path_2'] = df['pmcid'].progress_apply(find_path_pmc)
df['file_path'] = df['file_path'].fillna(df['file_path_2'])
df.drop('file_path_2',axis=1,inplace=True)
df.dropna(subset=['file_path'],inplace=True)
df.index = range(0,len(df))
#4.get year
df['year'] = df['publish_time'].map(lambda x: int(re.findall(r'\d{4}',x)[0]))
#5.add details from json files to csv

# define stopwords
nlp = spacy.load('en_core_web_sm')
stopword_list = list(spacy.lang.en.stop_words.STOP_WORDS)

#add the reference's titles to df
def read_ref(path):
    with open(path,'r') as f:
        jsonfile = json.load(f)
    reference = ''
    for i in jsonfile['bib_entries'].keys():
        try:
            reference += jsonfile['bib_entries'][i]['title'].lower()
            reference += '|'
        except:
            pass
    reference = reference.split(' ')
    reference = [i for i in reference if not i in stopword_list]
    reference = ' '.join(reference)
    return reference


#add the authors' country to df
def read_country(path):
    with open(path,'r') as f:
        jsonfile = json.load(f)
    try:
        return jsonfile['metadata']['authors'][0]['affiliation']['location']['country'].lower()
    except:
        return None

    
#add contents from json to df
def read_content(path):
    with open(path,'r') as f:
        jsonfile = json.load(f)
    try:
        text = ''
        for i in jsonfile['body_text']:
            text += (i['text'].lower())
        text = text.split(' ')
        text = [i for i in text if not i in stopword_list]
        text = ' '.join(text)
        return text
    except:
        return None
df['country']    = df['file_path'].progress_apply(read_country)
df['ref_titles'] = df['file_path'].progress_apply(read_ref)
df['content']    = df['file_path'].progress_apply(read_content)
#6.remove irregular words
for i in ['title', 'abstract' , 'ref_titles' , 'content']:
    df[i] = df[i].fillna('')
    df[i] = df[i].str.lower()
    df[i] = df[i].progress_apply(lambda x : re.sub(r"[^a-z0-9\-\' ]",' ',x))
    df[i] = df[i].progress_apply(lambda x : ' '.join([i for i in x.split() if not i in stopword_list]))
    df[i] = df[i].progress_apply(lambda x : ' '.join([i for i in x.split() if len(i)>2]))
#7.merge text
for i in ['title', 'abstract' , 'ref_titles']:
    df['content'] = df['content'] + ' ' + df[i]
#1.create Vectorizer
c = CountVectorizer()
vectorizer = c.fit_transform(df['content'])
#2.create lda model
lda = LatentDirichletAllocation(n_components=50, random_state=0)
df_topic = lda.fit_transform(vectorizer)
#3.find related articles
def finding_related_articles_lda(keyword,df,fromYear,toYear,topn):
    df = df.loc[(df['year']>=fromYear) & (df['year']<=toYear)]
    search_vec = c.transform([keyword])
    search_topic = lda.transform(search_vec)
    df['distances'] = pd.DataFrame(df_topic).iloc[df.index].apply(lambda x: jensenshannon(x, search_topic[0]), axis=1)
    return df.sort_values('distances').iloc[0:topn,:]
finding_related_articles_lda('What do we know about non-pharmaceutical interventions',df,2019,2020,10)
#1.create w2v_model
w2v_model = Word2Vec(df['content'].apply(lambda x : x.split()),min_count=1)
#2.find related articles
def finding_related_articles_w2v(keyword,df,fromYear,toYear,topn,keynum):
    df = df.loc[(df['year']>=fromYear) & (df['year']<=toYear)]
    
    
    keyword_filtered = keyword.split('')
    keyword_filtered = [i for i in keyword_filtered if not i in stopword_list]
    keyword_filtered = w2v_model.most_similar(positive=keyword_filtered,topn=keynum)
    keyword_filtered = list(np.array(keyword_filtered)[:,0])
    
    def containKey(string):
        string = string.split()
        for i in keyword_filtered:
            if i in string:
                return True
        return False
    df['hasKey'] = df['content'].progress_apply(containKey)
    
    df = df[df['hasKey'] == True]
    df = df.drop('hasKey',axis=1)
    
    df['distances'] = df['content'].progress_apply(lambda x : w2v_model.wv.wmdistance(keyword.split(),str(x).split()))
    return df.sort_values('distances').iloc[0:topn,:]
finding_related_articles_w2v('What do we know about non-pharmaceutical interventions',df,2019,2020,10)
#create TFIDF Vectorizer
T = TfidfVectorizer()
tdidf = T.fit_transform(df['content'])

#dict{number:word}
tfidf_dict = pd.Series(data=pd.Series(T.vocabulary_).index,index=pd.Series(T.vocabulary_).values)

#dict{word : number}
tf_idf_dict = pd.Series(T.vocabulary_)

#delete words not in w2v_model
tf_idf_dict_filtered = tf_idf_dict[w2v_model.wv.index2word]
tf_idf_dict_filtered.dropna(inplace=True)
tf_idf_dict_filtered = tf_idf_dict_filtered.astype('int')

#dict{word : number} => dict{number : word}
tf_idf_dict_filtered = pd.Series(data=tf_idf_dict_filtered.index,index=tf_idf_dict_filtered.data)
# transfer words to tfidf vector
def tfidf_calc(string):
    tfidf_vec = T.transform([string]).toarray()[0]
    return tfidf_vec

# transfer tfidf vector to word2vec vector(Weighted average)
def vector_calc(tfidf_vec):
    #remove 0s in vector(words not in content)
    tfidf_index = np.concatenate(np.argwhere(tfidf_vec))
    tfidf_index_filtered = tf_idf_dict_filtered[tfidf_index].dropna()
    
    #words after filtered
    tfidf_words = tfidf_index_filtered.values
    #tfidf values after filtered
    tfidf_value = tfidf_vec[tfidf_index_filtered.index]
    
    #tfidf values * word2vec values (shape=(x,100))
    mix_array = w2v_model[tfidf_words]*tfidf_value.reshape(-1,1)
    
    # mean values of weight average values(shape=(100,)) 
    vector_final = np.mean(mix_array,axis=0)
    
    return vector_final

# mix of two function above
def stringToArray(string):
    return list(vector_calc(tfidf_calc(string)))
#.find related articles
def finding_related_articles_w2v_TFIDF(keyword,df,fromYear,toYear,topn):
    keyword = keyword.lower()
    keyword = stringToArray(keyword)
    df = df.loc[(df['year']>=fromYear) & (df['year']<=toYear)]
    df['vector'] = df['content'].progress_apply(stringToArray)
    df['distance'] = df['vector'].progress_apply(lambda x : np.abs(spatial.distance.cosine(x, keyword)))
    return df.sort_values('distances').iloc[0:topn,:]
finding_related_articles_w2v_TFIDF('What do we know about non-pharmaceutical interventions',df,2019,2020,10)
#11.plot
def plot_wc(df_plot):
    plt.figure(figsize=(10,5))
    for i in range(6):
        plt.subplot(2,3,i+1)
        wc = WordCloud().generate(df_rank['content'].iloc[i])
        plt.imshow(wc, interpolation='bilinear')
        plt.title('paper{}'.format(i))
        plt.axis("off")