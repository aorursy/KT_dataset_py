%%time

!pip install tensorflow==1.15

!pip install bert-serving-server==1.10.0

!pip install bert-serving-client==1.10.0



!cp /kaggle/input/biobert-pretrained /kaggle/working -r

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.index /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.index

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.data-00000-of-00001 /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.data-00000-of-00001

%mv /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/model.ckpt-1000000.meta /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed/bert_model.ckpt.meta



!pip install transformers

!pip install sentence-transformers

!pip install rake-nltk



print('installation done')
import subprocess

import pickle as pkl

import pandas as pd

import numpy as np 

from sentence_transformers import SentenceTransformer

from transformers import BertTokenizer, BertModel

from ipywidgets import interact, widgets # this is what makes the dataframe interactive

from scipy.spatial.distance import cdist

from scipy.spatial.distance import jensenshannon

from IPython.display import HTML, display

import matplotlib.pyplot as plt

from os import path

from PIL import Image

from textblob import TextBlob

import pyLDAvis.gensim

import pyLDAvis

import gensim

import spacy

import os

from scipy import spatial



plt.style.use("dark_background")

df = pkl.load(open('../input/bertbiobertdataframe/BERT-BioBERT-dataframe.pkl', "rb"))
meta_df=pd.read_csv('../input/CORD-19-research-challenge/metadata.csv')

meta_df = meta_df.dropna(subset=['url'])
from collections import Counter

meta_df = meta_df.dropna(subset=['journal'])

journals=meta_df['journal'].tolist()



count = Counter(journals)

freq=count.most_common(10)



paper_count=pd.DataFrame(freq,columns=['journals','number of papers'])

paper_count.sort_values('number of papers', ascending=False).set_index('journals')[:20].sort_values('number of papers', ascending=True).plot(kind='barh')
# df['abstract_word_count'] = df['abstract'].apply(lambda x: len(x.strip().split()))

# df['body_word_count'] = df['body_text'].apply(lambda x: len(x.strip().split()))

# df['title_word_count'] = df['title'].apply(lambda x: len(x.strip().split()))

# df[['abstract_word_count', 'body_word_count','title_word_count']].plot(kind='box', title='Boxplt of Word Count', figsize=(10,6))

# plt.show()

import seaborn as sns



headline_length=df['title'].str.len()

sns.distplot(headline_length)

plt.show()

headline_length=df['abstract'].str.len()

sns.distplot(headline_length)

plt.show()

headline_length=df['body_text'].str.len()

sns.distplot(headline_length)

plt.show()
df['polarity'] = df['abstract'].map(lambda text: TextBlob(text).sentiment.polarity)

df['abstract_len'] = df['abstract'].astype(str).apply(len)

import cufflinks as cf

cf.go_offline()

cf.set_config_file(offline=False, world_readable=True)

df['polarity'].iplot(kind='hist',bins=50,xTitle='polarity',linecolor='black',yTitle='count',title='Sentiment Polarity Distribution')

def Text(list):  

    str1 = ""    

    for element in list:  

        str1 += element 

    return str1

def getwordcloud(text): 

    wordcloud = WordCloud(max_font_size=256, max_words=500, background_color="white",stopwords = set(STOPWORDS),random_state=42, width=500, height=500).generate(text)

    plt.figure()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()

   
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



abstracts=Text(df['abstract'][:10].tolist())

common_abstract_word=getwordcloud(abstracts)
df2=df

df2['title']=df2['title'].astype(str)

title=Text(df2['title'].tolist())

common_abstract_word=getwordcloud(title)
#Bert

bert_client = SentenceTransformer('bert-base-nli-max-tokens')



#Biobert

bio_bert_command = 'bert-serving-start -model_dir /kaggle/working/biobert-pretrained/biobert_v1.1_pubmed -max_seq_len=None -max_batch_size=32 -num_worker=2'

process = subprocess.Popen(bio_bert_command.split(), stdout=subprocess.PIPE)

from bert_serving.client import BertClient

biobert_client = BertClient(ignore_all_checks=True)
bio_vectors = np.array(df.biobert_vector.tolist())

bert_vectors = np.array(df.bert_vector.tolist())
from scipy import spatial

def score(model_vectors,model_encode,size):

    score = []

    for i in range(size):

        result = 1 - spatial.distance.cosine(model_vectors[i],model_encode)

        score.append(result)

    return score
def avrage (model1_score,model2_score):

    l=[sum(n) for n in zip(*[model1_score,model2_score])]

    final_score = [x * 0.5 for x in l]

    return final_score 
default_question = 'Neonates and pregnant women'

pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('max_colwidth', 180)



results=[]

total_docs=df.shape[0]

@interact

def search_articles(

    query=default_question,

    num_results=[10, 25, 100],show_scores=[False, True],score_type=['Cosine']):



    bio_encode =biobert_client.encode([query])

    bert_encode = bert_client.encode([query])

   

    if score_type is 'Cosine':

        bert_score=score(bert_vectors,bert_encode,total_docs)

        bio_score=score(bio_vectors,bio_encode,total_docs)

        f_score=avrage(bert_score,bio_score)

        df["score"] = f_score

        select_cols = ['title', 'abstract', 'authors', 'score','url']

        results = df[select_cols].sort_values(by=['score'], ascending=False).head(num_results)

        results = results.dropna(subset=['title'])

        

#     print("results : {}".format(results[20703	]))

    if (len(results.index) == 0):

        print('NO RESULTS')

        

        return None

    else:

        



        top_row = results.iloc[0]



        print('TOP RESULT OUT OF ' + str(total_docs) + ' DOCS FOR QUESTION:\n' + query + '\n')

        print('TITLE: ' + str(top_row['title']) + '\n')

        print('ABSTRACT: ' + top_row['abstract'] + '\n')

        #print('PREDICTED TOPIC: ' + topic_list[int(top_row['best_topic'].replace('topic_', ''))])



        print('\nAUTHORS: ' + str(top_row['authors']))



        select_cols.remove('authors')

        

        return results[select_cols]
df_LDA = pkl.load(open('../input/lda-pkl/lda_output_final.pkl', "rb"))

df['license']=meta_df['license']
def relevant_articles_lda(query, k,show_LDA,show_Bert ):

    

    bert_encode = bert_client.encode([query])    

    bert_vectors = np.array(df.bert_vector.tolist())

    

    LDA_encode = pkl.load(open('../input/lda-pkl/'+str(query)+ '.pkl', "rb"))

    LDA_vectors = np.array(df_LDA)

    

    bert_score = score(bert_vectors,bert_encode,df_LDA.shape[0])

    LDA_score = score(LDA_vectors,LDA_encode,df_LDA.shape[0])

       

    df["bert_scores"]=bert_score

    df["LDA_scores"]=LDA_score

    df["LDA_scores"] = df["LDA_scores"].apply(lambda x: x*15)

    

    scores = pd.DataFrame({'bert_scores':bert_score,'LDA_scores': df["LDA_scores"]})

    df['avg_scores'] = scores.mean(axis=1)

    

    out_avg=df.sort_values(by=['avg_scores'],ascending=False)

    

    out_LDA=df.sort_values(by=['LDA_scores'],ascending=False)

    out_bert=df.sort_values(by=['bert_scores'],ascending=False)

    

    h = '<br/>'.join(['<a href="' + str(l) + '" target="_blank">'+ str(n) + '</a>' for l, n in out_avg[['url','title']].head(k).values])

    display(HTML(h))

    

    if(show_LDA):

        h = '<br/>'.join(['<a href="' + str(l) + '" target="_blank">'+ str(n) + '</a>' for l, n in out_LDA[['url','title']].head(k).values])

        display(HTML(h))

   

    if(show_Bert):

        h = '<br/>'.join(['<a href="' + str(l) + '" target="_blank">'+ str(n) + '</a>' for l, n in out_bert[['url','title']].head(k).values])

        display(HTML(h))

query_list=Risk_factors_list=["cerebrovascular","Age" ,"Cancer","Chronic","Diabetes","Drinking","Heart disease","Hypertension","Obese","Respiratory","smoke","Liver","Male gender","COPD"]



pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('max_colwidth', 180)



@interact

def search_articles(

    query=query_list,num_results=[5, 10, 25],show_LDA=[False,True],show_Bert=[False,True]):

    

    relevant_articles_lda(query,num_results,show_LDA,show_Bert)
license=df.pivot_table(index=['license'], aggfunc='size')

print(len(license))

license
out_avg=df.sort_values(by=['avg_scores'],ascending=False)

license_list=["bronze-oa","cc-by","cc-by-nc","gold-oa","green-oa"]

license_color=["bronze",   "red",    "pink",  "gold",  "green"]



def highlight_license(s):

    for i in range(len(license_list)):

        if s.license == license_list[i]:

            return ['background-color:'+ str(license_color[i])]*2

    return ['background-color:white']*2
out_colored=out_avg[["license","title"]].iloc[70:100]

out_colored.style.apply(highlight_license, axis=1)
