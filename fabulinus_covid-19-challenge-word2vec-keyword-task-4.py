# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import re
import string
import collections
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import MiniBatchKMeans

from time import time
%matplotlib inline
import os
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
import spacy
import spacy.cli
from spacy.matcher import Matcher 
from spacy.matcher import PhraseMatcher

spacy.cli.download("en")
spacy.cli.download("en_core_web_lg")
nlp = spacy.load('en_core_web_lg')
## set english loanguage
stop_words = set(stopwords.words('english'))

## declaration of Porter stemmer.
porter=PorterStemmer()

## Clean Null Record in dataframe
def cleanEmptyData(columnName,df):
    return df[df[columnName].notnull()]

## Remove Punctuation
def remove_punctuation(columnName,df):
    return df.loc[:,columnName].apply(lambda x: re.sub('[^a-zA-z\s]','',x))

## Convert To Lower Case
def lower_case(input_str):
    input_str = input_str.lower()
    return input_str  

## Remove duplicate item in the dataframe
def removeDuplicate(df,list):
    df.drop_duplicates(list, inplace=True)    

## Remove nlp stop words    
def remove_stop_words(columnName,df):
  return df.loc[:,columnName].apply(lambda x: [word for word in x.split() if word not in stop_words])

##Remove single character from the sentence
def remove_one_character_word(columnName,df):
  return df.loc[:,columnName].apply(lambda x: [i for i in x if len(i) > 1])

## Join as a single text with seperator
def join_seperator(columnName,df):
  seperator = ', '
  return df.loc[:,columnName].apply(lambda x: seperator.join(x))

## apply stemmer to data frame fields
def apply_stemmer(columnName,df):
  return df.loc[:,columnName].apply(lambda x: [porter.stem(word) for word in x])

## Data Cleaning Process function
def dataCleaningProcess(dataFrame):
    ## remove duplicate records
    removeDuplicate(dataFrame,['abstract', 'text_body'])
    
    ## clean null value records
    clean_data = cleanEmptyData('text_body',dataFrame)
    clean_data.loc[:,'text_body_clean'] = clean_data.loc[:,'text_body'].apply(lambda x: lower_case(x))
    
    ## removing punctuation 
    clean_data.loc[:,'text_body_clean'] = remove_punctuation('text_body_clean',clean_data)
    
    ## apply stop words
    clean_data.loc[:,'text_body_clean'] = remove_stop_words('text_body_clean',clean_data)
    
    ## apply stemmer for each tokens
    clean_data.loc[:,'text_body_clean'] = apply_stemmer('text_body_clean',clean_data)
    
    ## removing single charter word in the sentence
    clean_data.loc[:,'text_body_clean'] = remove_one_character_word('text_body_clean',clean_data)
    
    ## join as a single text from words token
    clean_data.loc[:,'text_body_clean'] = join_seperator('text_body_clean',clean_data)
    
    ## remove coma after join
    clean_data.loc[:,'text_body_clean'] = remove_punctuation('text_body_clean',clean_data)
    
    return clean_data
## get words token from text
def getWordsFromText(_text):
    words = []
    for i in range(0,len(_text)):
        words.append(str(_text.iloc[i]['text_body']).split(" "))
    return words

# Read Excel data as Data Frame
def readExcelToDataFrame(path):
    research_dataframe = pd.read_csv(path,index_col=False)
    research_dataframe.drop(research_dataframe.columns[research_dataframe.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    return research_dataframe

## basic scatter plot
def showScatterPlot(_X,title):
    # sns settings
    sns.set(rc={'figure.figsize':(15,15)})
    # colors
    palette = sns.color_palette("bright", 1)
    # plot
    sns.scatterplot(_X[:,0], _X[:,1], palette=palette)
    plt.title(title)
    # plt.savefig("plots/t-sne_covid19.png")
    plt.show()

## scatter plot with cluster
def showClusterScatterPlot(_X, _y_pred, title):
    # sns settings
    sns.set(rc={'figure.figsize':(10,10)})
    # colors
    palette = sns.color_palette("bright", len(set(_y_pred)))
    # plot
    sns.scatterplot(_X[:,0], _X[:,1], hue=_y_pred, legend='full', palette=palette)
    plt.title(title)
    # plt.savefig("plots/t-sne_covid19_label.png")
    plt.show()


## drop clumns
def getTargetData(dataFrame):
    text_body = dataFrame.drop(["doc_id", "source", "title", "abstract"], axis=1)
    return getWordsFromText(text_body)

## train model for tSNE clustering visualization
def trainEmbededData(_perplexity,dataFrame,total_cluster, _n_iter):
    ## convert text to word frequency vectors
    vectorizer = TfidfVectorizer(max_features=2**12)
    
    ## training the data and returning term-document matrix.
    _X = vectorizer.fit_transform(dataFrame['text_body_clean'].values)
    
    ## tsne declartion
    tsne = TSNE(verbose=1, perplexity=_perplexity,learning_rate=200, random_state=0, n_iter=_n_iter)
    _X_embeded = tsne.fit_transform(_X.toarray())
    
    ## clusterring for tsne
    _kmeans = MiniBatchKMeans(n_clusters=total_cluster)
    return _X_embeded,_kmeans,_X

## predicting cluster centers and predict cluster index for each sample
def predict(_kmeans,_X):
    return _kmeans.fit_predict(_X)

## reusable fucntion for TSNE K-Mean Clustering with TF-IDF
def analyse(pplexity,data_frame,cluster,iter):
    ## train model for tSNE clustering visualization
    embeded,kmeans,x = trainEmbededData(pplexity,data_frame,cluster,iter)
    pred = predict(kmeans,x)
    ## visualized the scatter plot
    showClusterScatterPlot(embeded,pred,'t-SNE Covid-19 - Clustered(K-Means) - Tf-idf with Plain Text')
    return embeded,kmeans,x
research_dataframe = readExcelToDataFrame('/kaggle/input/coviddata4/data.csv')
research_dataframe.head()
clean_data =dataCleaningProcess(research_dataframe)
clean_data.head()
clean_process_data = clean_data.drop(["doc_id", "source", "title", "abstract"], axis=1)
clean_process_data.head(20)
meta_data = readExcelToDataFrame('/kaggle/input/covidmeta/meta.csv')
meta_data.head()
def prepare_search_data(_meta_data_frame,research_dataframe):
    ## add a field doc_id
    _meta_data_frame["doc_id"] = _meta_data_frame["sha"]
    
    ## clean NUll record
    _meta_data_frame = cleanEmptyData('doc_id', _meta_data_frame)
    _meta_data_frame = cleanEmptyData('publish_time', _meta_data_frame)
    
    ## select only 2019 & 2020 published records
    meta_data_filter = _meta_data_frame[_meta_data_frame['publish_time'].str.contains('2019') | _meta_data_frame['publish_time'].str.contains('2020')]  
    
     ## clean NUll record
    research_dataframe_clean = cleanEmptyData('doc_id', research_dataframe)
    research_dataframe_clean = cleanEmptyData('text_body', research_dataframe_clean)
    
    ## merging of Research data and meta data on doc_id
    tmp_data_frame  = research_dataframe_clean.merge(meta_data_filter, on='doc_id', how='right')
    
    ## remove un used fields
    clean_process_data = tmp_data_frame.drop(["source", "abstract_x",  "abstract_x","sha","source_x","title_y","pmcid","pubmed_id","license","abstract_y","journal","Microsoft Academic Paper ID","WHO #Covidence"], axis=1)
    
    ## clean NUll record
    clean_process_data = cleanEmptyData('text_body', clean_process_data)
    clean_process_data = clean_process_data.rename(columns={'title_x': 'title'}) 
    
    # reordering the column index
    columns = ["doc_id","doi", "publish_time", "authors","url","title", "text_body"]
    clean_process_data = clean_process_data.reindex(columns=columns)
    
    return clean_process_data
def process_title(x):
  if not str(x['title_x']).lower() =='nan':
    return str(x['title_x']) + ' (' +  str(x['url']) + ')'
  else:
    return str(x['url'])
filter_data = prepare_search_data(meta_data,research_dataframe)
filter_data.head()
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def show_WordCloud(filter_data):
    comment_words = ' '
    stopwords = set(STOPWORDS) 
  
# iterate through the csv file 
    for val in filter_data: 

        # typecaste each val to string 
        val = str(val) 

        # split the value 
        tokens = val.split() 
        #print(val) 
        # Converts each token into lowercase 
        for i in range(len(tokens)): 
            tokens[i] = tokens[i].lower() 

        for words in tokens: 
         comment_words = comment_words + words + ' '
         #print(comment_words)

    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white',
                    max_words = 200, 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 

    # plot the WordCloud image                        
    plt.figure(figsize = (10, 10), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 

    plt.show()


show_WordCloud(filter_data.title)
embeded,kmeans,x = analyse(5000,clean_process_data,10,15000)
papers = research_dataframe['text_body'].astype('str')
len(papers)
papers.head()
#meta_data_filter = _meta_data_frame[_meta_data_frame['publish_time'].str.contains('2019') | _meta_data_frame['publish_time'].str.contains('2020')]   
%%time
import nltk
import tqdm
nltk.download('wordnet')

stop_words = nltk.corpus.stopwords.words('english')
wtk = nltk.tokenize.RegexpTokenizer(r'\w+')
wnl = nltk.stem.wordnet.WordNetLemmatizer()

def normalize_corpus(papers):
    norm_papers = []
    for paper in tqdm.tqdm(papers):
        paper = paper.lower()
        paper_tokens = [token.strip() for token in wtk.tokenize(paper)]
        paper_tokens = [wnl.lemmatize(token) for token in paper_tokens if not token.isnumeric()]
        paper_tokens = [token for token in paper_tokens if len(token) > 1]
        paper_tokens = [token for token in paper_tokens if token not in stop_words]
        paper_tokens = list(filter(None, paper_tokens))
        #if paper_tokens:
        norm_papers.append(paper_tokens)
            
    return norm_papers
    
norm_papers = normalize_corpus(papers)
print(len(norm_papers))
import gensim

bigram = gensim.models.Phrases(norm_papers, min_count=20, threshold=20, delimiter=b'_') # higher threshold fewer phrases.
bigram_model = gensim.models.phrases.Phraser(bigram)

print(bigram_model[norm_papers[0]][:50])
print(bigram_model[norm_papers[1]][:50])
norm_corpus_bigrams = [bigram_model[doc] for doc in norm_papers]

# Create a dictionary representation of the documents.
dictionary = gensim.corpora.Dictionary(norm_corpus_bigrams)
print('Sample word to number mappings:', list(dictionary.items())[:15])
print('Total Vocabulary Size:', len(dictionary))
# Filter out words that occur less than 20 documents, or more than 60% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.6)
print('Total Vocabulary Size:', len(dictionary))
bow_corpus = [dictionary.doc2bow(text) for text in norm_corpus_bigrams]
print(bow_corpus[1][:50])
print([(dictionary[idx] , freq) for idx, freq in bow_corpus[1][:50]])
print('Total number of papers:', len(bow_corpus))
import joblib
lda_model = joblib.load('/kaggle/input/coviddata41/lda_model.jl')
topics_assigned = lda_model[bow_corpus]
len(topics_assigned)
b= pd.DataFrame(topics_assigned,columns = ['T0','T1','T2','T3','T4','T5','T6','T7','T8','T9'])
d= pd.concat([research_dataframe['text_body'],b],axis=1)
d.to_csv("Topic_paper_07042020_v4.csv")
for topic_id, topic in lda_model.print_topics(num_topics=50, num_words=20):
    print('Topic #'+str(topic_id+1)+':')
    print(topic)
    print()
import numpy as np
topics_coherences = lda_model.top_topics(bow_corpus, topn=20)
avg_coherence_score = np.mean([item[1] for item in topics_coherences])
print('Avg. Coherence Score:', avg_coherence_score)
topics_with_wts = [item[0] for item in topics_coherences]
print('LDA Topics with Weights')
print('='*50)
for idx, topic in enumerate(topics_with_wts):
    print('Topic #'+str(idx+1)+':')
    print([(term, round(wt, 3)) for wt, term in topic])
    print()
print('LDA Topics without Weights')
print('='*50)
for idx, topic in enumerate(topics_with_wts):
    print('Topic #'+str(idx+1)+':')
    print([term for wt, term in topic])
    print()
cv_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, 
                                                      texts=norm_corpus_bigrams,
                                                      dictionary=dictionary, 
                                                      coherence='c_v')
avg_coherence_cv = cv_coherence_model_lda.get_coherence()

umass_coherence_model_lda = gensim.models.CoherenceModel(model=lda_model, corpus=bow_corpus, 
                                                         texts=norm_corpus_bigrams,
                                                         dictionary=dictionary, 
                                                         coherence='u_mass')
avg_coherence_umass = umass_coherence_model_lda.get_coherence()

perplexity = lda_model.log_perplexity(bow_corpus)

print('Avg. Coherence Score (Cv):', avg_coherence_cv)
print('Avg. Coherence Score (UMass):', avg_coherence_umass)
print('Model Perplexity:', perplexity)
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
stopwords = set(STOPWORDS)


def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        max_words=2000,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
with open("/kaggle/input/coviddata41/Topic_paper_07042020_v4.csv",encoding = 'utf8', errors='ignore') as f:
    df_topic = pd.read_csv(f)
    
#df = df.replace("\n"," ").dropna()
show_WordCloud(df_topic.loc[df_topic['Dominant_topic'] == 0]['text_body'])
#show_WordCloud(filter_data.title)
show_WordCloud(df_topic.loc[df_topic['Dominant_topic'] == 1]['text_body'])

from __future__ import print_function

__author__ = 'maxim'

import numpy as np
import gensim
import string
from gensim.models import Word2Vec
from keras.callbacks import LambdaCallback
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils.data_utils import get_file
from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import re
import json
import pandas as pd
from gensim.models.fasttext import FastText
from os import listdir
from os.path import isfile, join

from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re

import gensim
import nltk
print('loading model')

word_model = Word2Vec.load("/kaggle/input/coviddata41/word2vec_1000ITR.model")

print('model loaded')
print(word_model.most_similar(positive=['covid19'])) 
print(word_model.most_similar(positive=['treatment','options' ])) 

arr_most_similar = [['covid19','medicine','treatment'],['covid19','medicine','treatment','chuanxiong'],['covid19','medicine','treatment','lopinavirritonavir','ribavirin'],['covid19','antivirals','remdesivir']]
arr_predict=[['viral','inhibitors','replication']]     

#Question: 9     
#Public health mitigation measures that could be effective for control
#Intial input fed to word2vec  model to extract related words that could lead to the answers of this question
#['public','health','mitigation','measures',  'effective', 'control','covid19','disease']]
#RESULT from first Query: [('interventions', 0.7447171807289124), ('prevention', 0.7438710927963257), ('preventive', 0.7241473197937012), ('policies', 0.7131460905075073), ('intervention', 0.7103219628334045), ('management', 0.7056314945220947), ('implementing', 0.6939526200294495), ('policy', 0.6897070407867432), ('quarantine', 0.6833001971244812), ('planning', 0.6516326069831848), ('implementation', 0.641217827796936), ('awareness', 0.6297034025192261), ('containment', 0.6278428435325623), ('preparedness', 0.622099757194519), ('epidemic', 0.6220937967300415), ('timely', 0.6204564571380615), ('community', 0.6184203624725342), ('government', 0.6134730577468872), ('outbreak', 0.6104072332382202), ('pandemic', 0.6079082489013672)]]
#Updated input taken from first querying of word2vec model after choosing relevant keywords
#['mitigation','measures','control','covid19','quarantine','containment','awareness','policies']]
#RESULT from second Query: [[('interventions', 0.7805880308151245), ('intervention', 0.7159266471862793), ('policy', 0.690924346446991), ('preventive', 0.6865450143814087), ('implementing', 0.6757345795631409), ('implementation', 0.6531403064727783), ('planning', 0.6507831811904907), ('practices', 0.6400246620178223), ('prevention', 0.6383914947509766), ('management', 0.6378570199012756), ('government', 0.6369956731796265), ('preparedness', 0.6348874568939209), ('restrictions', 0.6139740943908691), ('campaigns', 0.6061151027679443), ('behaviors', 0.5989149808883667), ('plans', 0.5976110696792603), ('decisions', 0.5968020558357239), ('timely', 0.591980516910553), ('governmental', 0.5905696153640747), ('biosecurity', 0.5887465476989746)]]

#Most Similar Keywrods Detection 

len(arr_most_similar)
arrans=[]
print(len(arr_most_similar))
count=0
for i in arr_most_similar:
    print('--------->',i)
    answers=word_model.most_similar(positive=i,topn=30)
    
    arrans.append(answers)
    count +=1
    print('=========',count)
print(arrans)
print(len(arrans))


#Predicted Keywords Detection

arr_predict
len(arr_predict)
arrans_arr_predict=[]
print(len(arr_predict))
count=0
for j in arr_predict:
    print('--------->',j)
    answers1=word_model.predict_output_word(j,topn=30)
    
    arrans_arr_predict.append(answers1)
    count +=1
    print('=========',count)
print(arrans_arr_predict)
print(len(arrans_arr_predict))


for q in arrans:
    print(q)
#arrans
for k in arrans_arr_predict:
    print(k)
## constant for spliting sentence
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

 
## spliting to sentence from text
def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

## search inference text by key words and return all the matches sentence
def search_inference_keys(text, keywords):
    sentences = split_into_sentences(text)
    txt = ''
    for sent in sentences:
        r = re.compile(keywords,flags=re.IGNORECASE)
        if len(r.findall(sent))>0:         
            txt = str(txt) + str(sent)
    return txt


## check key words exist or not in a sentence
def check_exist_multiple_keywords(text, keywords):
    r = re.compile(keywords, flags=re.IGNORECASE)
    if len(r.findall(text))>0:    
        return True
    else:
        return False

## Search Inference and download results as excel 
def searc_by_keys_as_excel(keyword, src_data_frame):
    data_frame = src_data_frame
    ## check exist to slice down the related contents
    data_frame['search_key_status'] =data_frame.loc[:,'text_body'].apply(lambda x: check_exist_multiple_keywords(x,keyword))
    ## select only target data
    process_data_frame = data_frame.query('search_key_status == True')
  
    ## filter on corona and covid 19 related data
    process_data_frame['search_covid_content'] =process_data_frame.loc[:,'text_body'].apply(lambda x: check_exist_multiple_keywords(x,'covid-19|sars-cov-2|2019-ncov|ncov-19|coronavirus'))
  
    ## get only covid-19|sars-cov-2|2019-ncov|ncov-19|coronavirus data
    process_data_frame = process_data_frame.query('search_covid_content == True')
    process_data_frame.loc[:,'inference'] = process_data_frame.loc[:,'text_body'].apply( lambda x: search_inference_keys(x,keyword))
    
    ## remove unused fields
    final_data = process_data_frame.drop(["search_key_status","text_body"], axis=1)
    ## download as excel
    final_data.to_excel(str(keyword) + '_result.xlsx', sheet_name='keyword')
  
    return final_data
# Search Inference for "incubation period" and download results as excel 
search_data = prepare_search_data(meta_data,research_dataframe)
final_data =searc_by_keys_as_excel('protease|inhibitor|',search_data)


final_data.head()
final_data =searc_by_keys_as_excel('ribavirin|chloroquine',search_data)

final_data =searc_by_keys_as_excel('ribavirin|chloroquine',search_data)
final_data =searc_by_keys_as_excel('remdesivir|lopinavir|ritonavir|oseltamivir|favipiravir|sofosbuvir|corticosteroids',search_data)