import numpy as np 
import pandas as pd 
import joblib
import re
import os
import sys
import glob
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import spacy
from spacy import displacy
from nltk.corpus import stopwords
nlp = spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import TfidfVectorizer
data_dir = "/kaggle/input/donald-trumps-rallies/"
text_files = glob.glob(f"{data_dir}*.txt")
print(f"Total Number of Documents : {len(text_files)}")
uscitymap = pd.read_csv("/kaggle/input/uscity2statemap/uscities_map.csv")
uscitymap.head(2)
text_place_date_ls = []
for i,val in enumerate(text_files):
    with open(text_files[i]) as f:
        speech_file = f.read()
        key = val.split("/")[-1].split(".")[0]
        try:
            text_place_date_ls.append([key,key[:-10],pd.to_datetime(key[-10:],format="%b%d_%Y"),speech_file])
        except:
            text_place_date_ls.append([key,key[:-9],pd.to_datetime(key[-9:],format="%b%d_%Y"),speech_file])

df = pd.DataFrame(text_place_date_ls,columns = ['filename','place','date','speech_text'])        

## correcting the name of place as for two part name it is without space between two parts but the second part start with capital letter
df['correct_place'] = df.place.apply(lambda x : re.findall('[A-Z][^A-Z]*',x))
df['locate_'] = df['correct_place'].apply(lambda x: sum([1 for i in x if i.find("-")>=0]))
df['correct_place'] = df.apply(lambda x : "".join(x['correct_place']) if x['locate_']>0 else " ".join(x['correct_place']),axis=1)
df.drop(["locate_"],axis=1,inplace=True)

df = pd.merge(df,uscitymap[['city','state_name']].drop_duplicates()
              ,right_on='city',left_on='correct_place',how='left')

df['number_lines'] = df.speech_text.apply(lambda x: len(x.split('.'))) ## assuming sentence ends at '.'
df['place_name_count'] = df.apply(lambda x : len([m.start() for m in re.finditer(x['correct_place'],x['speech_text'])]),axis=1) 
df['state_name_count'] = df.apply(lambda x : len([m.start() for m in re.finditer(str(x['state_name']),x['speech_text'])]),axis=1) 
df['self_name_count'] = df.apply(lambda x : len([m.start() for m in re.finditer('Donald Trump',x['speech_text'])]),axis=1) 

df['words_ls'] = df.speech_text.apply(lambda x: x.split(" "))
df['total_words'] = df.words_ls.apply(lambda x: sum([1 for i in x if i.isalpha()]))
df['total_words_remove_stopwords'] = df.words_ls.apply(lambda x: sum([1 for i in x if i not in stopwords.words('english')]))

df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['weekday'] = df['date'].dt.weekday
df['month'] = df['date'].dt.month
df_dict = df[['filename','speech_text']].to_dict(orient='records')
for sub_dict in df_dict:
    ent_ls = nlp(sub_dict['speech_text'])
    ent_ls = [[ent.text,ent.label_] for ent in ent_ls.ents]
    sub_dict['named_entity_dict'] = ent_ls
all_entity_df = pd.DataFrame()
for i in range(len(df_dict)):
    entity_type_df = pd.DataFrame(df_dict[i]['named_entity_dict'],columns=['value','label']).reset_index().groupby(['label'],as_index=False).value.count()
    entity_type_df['filename'] = df_dict[i]['filename']
    all_entity_df = all_entity_df.append(entity_type_df)

all_entity_val_df = pd.DataFrame()
for i in range(len(df_dict)):
    entity_val_type_df = pd.DataFrame(df_dict[i]['named_entity_dict'],columns=['value','label']).reset_index().groupby(['value','label'],as_index=False).count()
    entity_val_type_df['filename'] = df_dict[i]['filename']
    all_entity_val_df = all_entity_val_df.append(entity_val_type_df)


speech_text_ls = df[['filename','speech_text']].drop_duplicates()['speech_text'].values

tfidf = TfidfVectorizer(stop_words=stopwords.words('english'))
tfidf_vec = tfidf.fit_transform(speech_text_ls)

tfidf_df = pd.DataFrame(tfidf_vec.todense())
tfidf_df.columns = tfidf.get_feature_names()
tfidf_df['filename'] = list(df[['filename']].drop_duplicates().filename)
joblib.dump({"df_dict":df_dict,'df':df,
             'all_entity_df':all_entity_df,'all_entity_val_df':all_entity_val_df,
             'tfidf_df':tfidf_df},"data.pkl")
# data_saved = joblib.load("/kaggle/input/processed-data/data.pkl")
# df_dict = data_saved['df_dict']
# df = data_saved['df']


data_saved = joblib.load("/kaggle/input/processed/data-2.pkl")
tfidf_df = data_saved['tfidf_df']
df = data_saved['df']
all_entity_df = data_saved['all_entity_df']
all_entity_val_df = data_saved['all_entity_val_df']
df_dict = data_saved['df_dict']
all_entity_val_df_agg = all_entity_val_df.groupby(['value','label'],as_index=False)['index'].sum()
tfidf_df.drop(columns=['filename']).iloc[0].sort_values(ascending=False).head(20).plot(kind='bar')
plt.bar('month','date',data  = df.groupby(['month'],as_index=False).date.count())
plt.bar('weekday','date',data  = df.groupby(['weekday'],as_index=False).date.count())##0 is monday
all_entity_df.groupby(['label'],as_index=False).value.sum().plot(x='label',y='value',kind='bar')
plt.figure(figsize=(10,15))
plt.subplots_adjust(wspace=0.8,hspace=1.5)
cnt = 1
for label in all_entity_val_df_agg.label.unique():
    plt.subplot(6,3,cnt)
    cnt+=1
    plt.barh('value','index',
             data = all_entity_val_df_agg[all_entity_val_df_agg.label==label].sort_values(['index'],ascending=False).head(5))
    plt.xticks(rotation=90)
    plt.title(label)
sns.distplot(df['total_words'])
sns.distplot(df['self_name_count'])
sns.distplot(df.number_lines)
## example of named entity tagging
doc = nlp(df_dict[0]['speech_text'][:1000])
displacy.render(doc, style="ent",jupyter=True)
from spacy.lang.en import English
import string
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from gensim import corpora
import gensim
import pyLDAvis.gensim
punctuations= string.punctuation
parser = English()
all_stopwords = nlp.Defaults.stop_words
external_stopwords = ['people','know','going','say','great','right','want','like','get','good','come','guy','think','thank','...','years','time',
                     'tell','look','american','country','lot','new','state','way','go','ok','sir','let','actually','america','little','okay','happen',
                      'try','remember','hear','best','thing','numbers','pay','beautiful','take'
                     ]
def tokenize(txt):
    tokens = parser(txt)
    lda_tokens = []
    for token in tokens :
        if token.orth_.isspace(): continue
        elif token.lower_ in all_stopwords : continue
        elif token.lower_ in punctuations : continue
        else :
            lda_tokens.append(token.lower_)
    return lda_tokens

def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None: return word
    else : return lemma

def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


def prepare_data_for_lda(txt):
    tokens = tokenize(txt)
    tokens = [get_lemma(word) for word in tokens]
    tokens = [word for word in tokens if word not in external_stopwords]
    return tokens
data_saved = joblib.load("/kaggle/input/processed/data-2.pkl")
df = data_saved['df']

text_data = [prepare_data_for_lda(txt) for txt in df[['speech_text','filename']].drop_duplicates().speech_text]

dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(txt) for txt in text_data]

num_topics = 5
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,num_topics=num_topics,id2word=dictionary,passes=200)
topics = lda_model.print_topics(num_words=20)
plt.figure(figsize=(25,4))
plt.subplots_adjust(wspace=0.5,hspace=0.9)
for idx in range(len(topics)):
    plt.subplot(1,num_topics,idx+1)
    split_topics = pd.DataFrame([[float(i.split("*")[0]),i.split("*")[1]] for i in topics[idx][1].split("+")],columns=['weight','word'])
    plt.barh("word",'weight',data=split_topics)
    plt.xticks(rotation=90)
    plt.title(f"Topic Number {idx}")
lda_display = pyLDAvis.gensim.prepare(lda_model,corpus,dictionary,sort_topics=False)
pyLDAvis.display(lda_display)
# !pip install umap-learn
# !pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import DBSCAN,KMeans
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


warnings.filterwarnings("ignore")
data = df[['speech_text','filename']].drop_duplicates().speech_text.values

model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(data, show_progress_bar=True)
## creating clusters
cluster_bert  =  KMeans(n_clusters = 4 ).fit(embeddings)

## visualization
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster_bert.labels_
plt.scatter("x","y",c='labels',data = result)

docs_df = pd.DataFrame(data,columns =['doc'])
docs_df['labels'] = cluster_bert.labels_
docs_df['row_num'] = range(len(docs_df))

docs_per_id = docs_df.groupby(['labels'],as_index=False).agg({'doc':" ".join})

def c_tf_idf(documents, m,external_stopwords, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words=list(all_stopwords)+external_stopwords).fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  

external_stopwords = external_stopwords + ['said','done','ve','got']
tf_idf, count = c_tf_idf(docs_per_id.doc.values,len(data),external_stopwords)
n_words=20
tf_idf_transposed = tf_idf.T
words = count.get_feature_names()
labels = docs_per_id.labels.values

indices = np.argsort(tf_idf_transposed,axis=1)[:,:n_words]

top_n_words = {i:[(words[j],tf_idf_transposed[i,j]) for j in indices[i]] for i in labels}

topic_sizes = (docs_df.groupby(['labels'],as_index=False).doc.count()
               .rename(columns={"doc":'size'})
               .sort_values(by=['size'],ascending=False))



plt.figure(figsize=(15,10))
for i in range(len(top_n_words)):
    plt.subplot(2,3,i+1)
    sns.barplot("tfidf","word",data=pd.DataFrame(top_n_words[i],columns =['word','tfidf']))

