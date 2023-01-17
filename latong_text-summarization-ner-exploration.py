import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
import glob
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from  collections import OrderedDict


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
meta=pd.read_csv("/kaggle/input/CORD-19-research-challenge/metadata.csv")
meta.head()
meta.shape
meta=meta[((meta['has_pdf_parse']==True) |(meta['has_pmc_xml_parse']==True))]
meta_sm=meta[['cord_uid','sha','pmcid','title','abstract','publish_time','url']]
meta_sm.drop_duplicates(subset ="title", keep = False, inplace = True)
meta_sm.loc[meta_sm.publish_time=='2020-12-31'] = "2020-03-31"
meta_sm.head()
meta_sm.shape
sys.path.insert(0, "../")

root_path = '/kaggle/input/CORD-19-research-challenge/'
#inspired by this kernel. Thanks to the developer ref. https://www.kaggle.com/fmitchell259/create-corona-csv-file
# Just set up a quick blank dataframe to hold all these medical papers. 

df = {"paper_id": [], "text_body": []}
df = pd.DataFrame.from_dict(df)
df
collect_json = glob.glob(f'{root_path}/**/*.json', recursive=True)

for i,file_name in enumerate (collect_json):
    row = {"paper_id": None, "text_body": None}
    if i%2000==0:
        print ("====processed " + str(i)+ ' json files=====')
        print()

    with open(file_name) as json_data:
            
        data = json.load(json_data,object_pairs_hook=OrderedDict)
        
        row['paper_id']=data['paper_id']
        
        body_list = []
       
        for _ in range(len(data['body_text'])):
            try:
                body_list.append(data['body_text'][_]['text'])
            except:
                pass

        body = "\n ".join(body_list)
        
        row['text_body']=body 
        df = df.append(row, ignore_index=True)
  
df.shape
#merge metadata df with parsed json file based on sha_id
merge1=pd.merge(meta_sm, df, left_on='sha', right_on=['paper_id'])
merge1.head()
len(merge1)
#merge metadata set with parsed json file based on pcmid
merge2=pd.merge(meta_sm, df, left_on='pmcid', right_on=['paper_id'])
merge2.head()
len(merge2)
#combine merged sha_id and pcmid dataset, remove the duplicate values based on file name
merge_final= merge2.append(merge1, ignore_index=True)
merge_final.drop_duplicates(subset ="title", keep = False, inplace = True)
len(merge_final)
merge_final.head()
#remove articles that are not related to COVID-19 based on publish time
corona=merge_final[(merge_final['publish_time']>'2019-11-01') & (merge_final['text_body'].str.contains('nCoV|Cov|COVID|covid|SARS-CoV-2|sars-cov-2'))]
corona.shape
import re 
def clean_dataset(text):
    text=re.sub('[\[].*?[\]]', '', str(text))  #remove in-text citation
    text=re.sub(r'^https?:\/\/.*[\r\n]*', '',text, flags=re.MULTILINE)#remove hyperlink
    text=re.sub(r'\\b[A-Z a-z 0-9._ - ]*[@](.*?)[.]{1,3} \\b', '', text)#remove email
    text=re.sub(r'^a1111111111 a1111111111 a1111111111 a1111111111 a1111111111.*[\r\n]*',' ',text)#have no idea what is a11111.. is, but I remove it now
    text=re.sub(r'  +', ' ',text ) #remove extra space
    text=re.sub('[,\.!?]', '', text)
    text=re.sub(r's/ ( *)/\1/g','',text) 
    text=re.sub(r'[^\w\s]','',text) #strip punctions (recheck)
    return text
import warnings
warnings.filterwarnings('ignore')
corona['text_body'] =corona['text_body'].apply(clean_dataset)
corona['title'] =corona['title'].apply(clean_dataset)
corona['abstract'] =corona['abstract'].apply(clean_dataset)
corona['text_body'] = corona['text_body'].map(lambda x: x.lower())
coro=corona.reset_index(drop=True)
coro.head()
coro['count_abstract'] = coro['abstract'].str.split().map(len)
coro['count_abstract'].sort_values(ascending=True)
#check word count
y = np.array(coro['count_abstract'])

sns.distplot(y);
coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)
#check word count
import seaborn as sns
import matplotlib.pyplot as plt

y = np.array(coro['count_abstract'])

sns.distplot(y);
coro['count_text'] = coro['text_body'].str.split().map(len)
coro['count_text'].sort_values(ascending=True)
coro['count_text'].describe()
y = np.array(coro['count_text'])

sns.distplot(y);
coro2=coro[((coro['count_text']>500)&(coro['count_text']<4000))]
coro2.shape
coro2.to_csv("corona.csv",index=False)
#split articles w/o abstarct as the test dataset

test=coro2[coro2['count_abstract']<5]
test.head()
test.shape
train= coro2.drop(test.index)

train.head()
train.shape
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
!pip install bert-extractive-summarizer
!pip install spacy
!pip install transformers==2.6.0
!pip install neuralcoref
# It seems there is something wrong with Bert Summarizer at the moment, if you want to see how it works, you can check out my last version

from summarizer import Summarizer
train['summary']=" "

for i in range(2):
    body=" "
    result=" " 
    full=" " 
    model = Summarizer()
    body=train['text_body'][i]
    result = model(body, min_length=200)
    full = ''.join(result)
    train['summary'][i]=full
     # print(i, train['summary'][i])
     # print("===next====")
#Bert does not work
# It seems there is something wrong with the environment at the moment, if you want to see how it works, you can check out my last version
train['summary'][0]
body=train['text_body'][0]
#GPT2
from summarizer import Summarizer,TransformerSummarizer
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
full = ''.join(GPT2_model(body, min_length=200))
print(full)
model = TransformerSummarizer(transformer_type="XLNet",transformer_model_key="xlnet-base-cased")
full = ''.join(model(body, min_length=60))
print(full)
!pip install spacy
!pip install transformers
import transformers
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
from transformers import pipeline

# load BART summarizer
summarizer = pipeline(task="summarization")
print(train['text_body'][0])
#I will redo this part

#remove stop words
import gensim
from gensim.parsing.preprocessing import remove_stopwords

my_extra_stop_words = ['preprint','paper','copyright','case','also','moreover','use','from', 'subject', 're', 'edu', 'use','and','et','al','medrxiv','peerreviewed','peerreview','httpsdoiorg','license','authorfunder','grant','ccbyncnd','permission','grant','httpsdoiorg101101202002']

train['text_body']=train['text_body'].apply(lambda x: ' '.join([word for word in x.split() if word not in (my_extra_stop_words) and word not in gensim.parsing.preprocessing.STOPWORDS and len(word)>3]))

coronaRe=train.reset_index(drop=True)
import spacy
nlp=spacy.load("en_core_web_sm",disable=['parser', 'ner'])

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    text_out=[]
    for word in texts:
      data=nlp(word)
      data=[word.lemma_ for word in data]
      text_out.append(data)
    return text_out
coronaRe['new_lem'] = lemmatization(coronaRe['text_body'],allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
from gensim.corpora import Dictionary
docs = coronaRe['new_lem']
dictionary = Dictionary(docs)

# Filter out words that occur less than 10 documents, or more than 50% of the documents
dictionary.filter_extremes(no_below=10, no_above=0.5)

# Create Bag-of-words representation of the documents
corpus = [dictionary.doc2bow(doc) for doc in docs]

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))
coronaRe.head()
import gensim.corpora as corpora
# Create Dictionary
dictionary = gensim.corpora.Dictionary(coronaRe['new_lem'])
count = 0
for k, v in dictionary.iteritems():
    #print(k, v)
    count += 1
#less than 15 documents (absolute number) or more than 0.5 documents (fraction of total corpus size, not absolute number).after the above two steps, keep only the first 4500 most frequent tokens.
dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=4500)
# Create Corpus
bow_corpus = [dictionary.doc2bow(doc) for doc in coronaRe
              ['new_lem']]
bow_corpus_id=[ id for id in coronaRe['cord_uid']]
# View
print(bow_corpus[:1])

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=bow_corpus,
                                       id2word=dictionary,
                                       num_topics=10, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)
from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]
lda_df = lda_model.get_document_topics(bow_corpus,minimum_probability=0)
lda_df = pd.DataFrame(list(lda_df))

num_topics = lda_model.num_topics

lda_df.columns = ['Topic'+str(i) for i in range(num_topics)]
for i in range(len(lda_df.columns)):
    lda_df.iloc[:,i]=lda_df.iloc[:,i].apply(lambda x: x[1])
lda_df['Automated_topic_id'] =lda_df.apply(lambda x: np.argmax(x),axis=1)
lda_df.head()
#coherence score https://stackoverflow.com/questions/54762690/coherence-score-0-4-is-good-or-bad
from gensim.models import CoherenceModel
# Compute Coherence Score
from tqdm import tqdm
coherenceList_cv=[]
num_topics_list = np.arange(5,26)
for num_topics in tqdm(num_topics_list):
  lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                         id2word=dictionary,
                                         num_topics=num_topics,
                                         random_state=100,
                                         chunksize=100,
                                         passes=10,
                                         alpha='auto',
                                         per_word_topics=True)
  coherence_model_lda = CoherenceModel(model=lda_model, texts=coronaRe['new_lem'], coherence='c_v')
  coherence_lda = coherence_model_lda.get_coherence()
  coherenceList_cv.append(coherence_lda)
print('\nCoherence Score: ', coherence_lda)
#re-do (not correct)
plotData = pd.DataFrame({'Number of topics':num_topics_list,
                         'CoherenceScore_cv':coherenceList_cv})
f,ax = plt.subplots(figsize=(10,6))
sns.set_style("darkgrid")
sns.pointplot(x='Number of topics',y= 'CoherenceScore_cv',data=plotData)

plt.title('Topic coherence')
#final model

Lda = gensim.models.LdaMulticore
lda_final= Lda(corpus=bow_corpus, num_topics=17,id2word = dictionary, passes=10,chunksize=100,random_state=100)
from pprint import pprint
# Print the Keyword in the 11 topics
pprint(lda_final.print_topics())
doc_lda = lda_final[corpus]
lda_df = lda_final.get_document_topics(bow_corpus,minimum_probability=0)
lda_df = pd.DataFrame(list(lda_df))
lda_id=pd.DataFrame(list(bow_corpus_id))
num_topics = lda_final.num_topics

lda_df.columns = ['Topic'+str(i) for i in range(num_topics)]

for i in range(len(lda_df.columns)):
    lda_df.iloc[:,i]=lda_df.iloc[:,i].apply(lambda x: x[1])

lda_df['Automated_topic_id'] =lda_df.apply(lambda x: np.argmax(x),axis=1)

lda_df['cord_uid']= lda_id
lda_df[39:40]
topic=lda_df[['Automated_topic_id','cord_uid']]
plot_topics=lda_df.Automated_topic_id.value_counts().reset_index()
plot_topics.columns=["topic_id","quantity"]
plot_topics
ax = sns.barplot(x="topic_id", y="quantity",  data=plot_topics)
coronaRe['topic_id']= topic['Automated_topic_id']
coronaRe.head()
#https://medium.com/@manivannan_data/how-to-train-ner-with-custom-training-data-using-spacy-188e0e508c6
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_ner_bionlp13cg_md-0.2.4.tar.gz
import spacy
from spacy import displacy
from collections import Counter

import en_ner_bionlp13cg_md
nlp = en_ner_bionlp13cg_md.load()
text = train['abstract'][2]
doc = nlp(text)
print(list(doc.sents))
print(doc.ents)
from spacy import displacy
displacy.render(next(doc.sents), style='dep', jupyter=True,options = {'distance': 110})
displacy.render(doc, style='ent')
#!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_sm-0.2.4.tar.gz
#!pip install en_core_web_sm
#!pip install git+https://github.com/NLPatVCU/medaCy.git@development
!pip install git+https://github.com/NLPatVCU/medaCy.git@development
!pip install git+https://github.com/NLPatVCU/medaCy_model_clinical_notes.git