import numpy as np
import torch
import tensorflow
import pandas as pd
import os
import json
import time
import glob
import re
import sys
import collections
from nltk import flatten
import dask
from dask import delayed,compute
import dask.dataframe as dd
from dask.multiprocessing import get
from tqdm._tqdm_notebook import tqdm_notebook
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm_notebook.pandas()
sys.path.insert(0, "../")

root_path = '/kaggle/input/CORD-19-research-challenge'

corona_features = {"doc_id": [None], "source": [None], "title": [None],
                  "abstract": [None], "text_body": [None]}
corona_df = pd.DataFrame.from_dict(corona_features)

json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)
def return_corona_df(json_filenames, df, source):
    
    for file_name in tqdm(json_filenames[20000:50000]):

        row = {}

        with open(file_name) as json_data:
            data = json.load(json_data)

            doc_id = data['paper_id']
            row['doc_id'] = doc_id
            row['title'] = data['metadata']['title']

            # Now need all of abstract. Put it all in 
            # a list then use str.join() to split it
            # into paragraphs. 
            try:
                abstract_list = [abst['text'] for abst in data['abstract']]
                abstract = "\n ".join(abstract_list)

                row['abstract'] = abstract
            except:
                row['abstract'] = np.nan
            # And lastly the body of the text. 
            body_list = [bt['text'] for bt in data['body_text']]
            body = "\n ".join(body_list)
            
            row['text_body'] = body
            
            # Now just add to the dataframe. 
            
            if source == 'b':
                row['source'] = "BIORXIV"
            elif source == "c":
                row['source'] = "COMMON_USE_SUB"
            elif source == "n":
                row['source'] = "NON_COMMON_USE"
            elif source == "p":
                row['source'] = "PMC_CUSTOM_LICENSE"
            
            df = df.append(row, ignore_index=True)
            
            
    return df
    
corona_df = return_corona_df(json_filenames,corona_df, 'b')
corona_df.dropna(subset=['text_body'],inplace=True)
corona_df.head()
#Install scispcy and spacy and pretrained model enc_core_sci_lg for analysis
!pip install -U spacy
!pip install scispacy
# !pip install https://med7.s3.eu-west-2.amazonaws.com/en_core_med7_lg.tar.gz
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_lg-0.2.4.tar.gz
import scispacy
import spacy
import en_core_sci_lg
## loading en_core_sci_lg model and disabling parser and ner, as these are not going to be used in EDA, disabling these functions from NLP pipeline can sometimes make a big difference and improve loading speed
nlp = en_core_sci_lg.load(disable=["parser", "ner"])
nlp.max_length = 2000000
# Function for cleaning data by using POS tagging and Lemmatization
def clean_text(sentence):
    if sentence:
        tokens = [word.lemma_ for word in nlp(str(sentence)) 
                  if not (word.like_num 
                          or word.is_stop
                          or word.is_punct
                          or word.is_space
                          or word.like_url
                          or word.like_email
                          or word.is_currency
                          or word.pos_ =='VBZ' 
                          or word.pos_ =='ADP'
                          or word.pos_ =='PRON'
                          or word.pos_ =='AUX'

                         )] 
    else :
        return np.nan
    return tokens
#cleaning of abstract data
corona_df["cleaned_abstract_tokens"] = corona_df['abstract'][22000:25000].progress_apply(lambda x: clean_text(x))

## this function can take take of lot of memory and execution time. uncomment to run and check output of frquency and word cloud
#cleaning of text body data

#corona_df["cleaned_text_body_tokens"] = corona_df['text_body'].progress_apply(lambda x: clean_text(str(x)))
# example for pos tagging
import spacy
from spacy import displacy
nlp = en_core_sci_lg.load()
## Create nlp object of spacy
doc = nlp(corona_df["title"].iloc[22000])
## Render output by using displacy module
displacy.render(doc, style="dep")
tokens_df= corona_df["cleaned_abstract_tokens"].dropna() #drop null values
# Get top 30 tokens based upon frequency in whole corpus
word_freq_top30 = pd.DataFrame(collections.Counter(flatten(tokens_df.to_list())).most_common(30),columns=['words',"frequency"])
##Plot bar graph of words wrt frequency 
sns.set(style="whitegrid")
sns.set(rc={'figure.figsize':(20,11)})
ax = sns.barplot(x=word_freq_top30['words'][1:], y=word_freq_top30['frequency'])
#ax.set_xticklabels(ax.get_xticklabels(),rotation=45)
plt.xticks(rotation=45)
from wordcloud import WordCloud, STOPWORDS
text = ' '.join(flatten(tokens_df.to_list())) ## text string of cleaned tokens 
stopwords = set(STOPWORDS) 
## add additional stopwwords based upon data, this is not final list, more words can added as per requiremet and data availability  
stopwords.update(["nan","find","show","conclusion","case","include","human","biorxib","day","total","author","funder",'virus','protein'])
wordcloud = WordCloud(stopwords=stopwords, background_color="white",width=1600,height=800).generate(text)

##Plotting
plt.figure(figsize=(20,10))
plt.axis("off")
# "bilinear" interpolation is used to make the displayed image appear more smoothly without overlapping of words  
# can change to some other interpolation technique to see changes like "hamming"
plt.imshow(wordcloud, interpolation="bilinear") 

plt.show()
## This function will extract all abbreviations and respective full form used within the text.
## This function has further scope of improvement
import re
def extract_abbreviations(text):
    '''
    Input: Text string
    Output: Dictionary of abbreviation and respective full form
    '''
    abbr_fullform_dict=dict()
    abbr = re.findall('\(([A-Z]{2,})\)',text)
    #print(abbr)
    if len(abbr)<10:
        for i in abbr:
            span_abbr = re.search(i,text).span()
            len_full_form = str(span_abbr[1]-span_abbr[0])
            try:
                full_form = re.search('(\w+\s)'+'{'+len_full_form+'}'+'\('+i+'\)',text).group(0)
                if full_form[0].lower()!= i.lower()[0]:
                    full_form = ' '.join(full_form.split()[1:])
                if full_form.lower()[-1]!= i.lower()[-1]:
                    full_form = ' '.join(full_form.split()[:-1])
                full_form = full_form.replace('('+i+')','').strip()
                abbr_fullform_dict[i] = full_form
            except AttributeError:
                print("error")
                pass
    
        return abbr_fullform_dict    
#corona_df["abbr_dict"]= pd.read_csv('/kaggle/input/abbreviations/abb.csv',names=['abb_dict'],header=None)
## calling function for extractons of abbreviations and respective full form from text
corona_df["abbr_dict"] = corona_df["text_body"].progress_apply(lambda x:extract_abbreviations(x))
#Function for replacing abbreviations by its full form
import ast
import numpy as np
def remove_abbriviation(text,abbr_dict):
    try :
       
        #abbr_dict = ast.literal_eval(abbr_dict)
    
        for key,value in abbr_dict.items():
                #print(key,value)
                text = text.replace(key,value)
    except AttributeError:            
          pass
    return text    
#Function for removing emailids from text
def remove_email(text):
    if text:
        text = text.lower()
        text = re.sub('([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})','*email*',str(text))
    else:
        pass
    return text
# Function for removing website links from text
def remove_weblink(text):
    if text:
        text=text.lower()
        text = re.sub('(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})','link',str(text))
    else:
        pass
    return text
# Function for removing paper refrences from text e.g [19],[1a,2b] or [19,14,15]
def remove_refrence(text):
    if text:
        text=text.lower()
        text = re.sub('\[\d+(,\s{0,}\d+){0,}\]','',str(text))   
    else:
        pass
    return text
# Function for removing non ASCII charecters like 'Ϫ','ó','ü','©','µ','▲','→'
# This function check if charecter hex value in range [\x00,\x7F] (in decimal [0,127] i.e range of ASCII charecters) and replace if it occurs outside limit
def remove_ghost_char(text):
    if text:
        text = re.sub(r'[^\x00-\x7F]+',' ', str(text))
    else:
        pass
    
    return text
# This function remove all bracktes with data
def remove_brackets(text):
    if text:
        text = re.sub('(\(.*?\))|(\[.*?\])','',str(text))   
    else:
        pass
    return text                  
# Function for removing multiple spaces 
def remove_extra_spaces(text):
    if text:
        text = re.sub(r'( +)',' ', str(text))
    else: 
        pass
   
    return text
def preprocess(df,dask_=False ,remove_abbr_=False,remove_email_=True,remove_weblink_=True,
               remove_refrence_=True,remove_brackets_=True,remove_ghost_char_=True,remove_extra_spaces_=True):
    start_time = time.time()
#     #series_text = corona_df['text_body']
#     series_abbr  = corona_df['abbr_dict']
#     _series = df["text_body"]
    
    if remove_abbr_: 
        print('Replacing abbreviations now!')
        _series = df.apply(lambda x: remove_abbriviation(x["text_body"],x["abbr_dict"]),axis=1)
    else:
        _series = df["text_body"]
                          
    if remove_email_: 
        print('Removing Email now!')
        if dask_:
            _series = (dd
                               .from_pandas(_series, npartitions=4)
                               .apply( lambda x: remove_email(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else:
            _series = _series.apply(lambda x: remove_email(x))
    if remove_weblink_: 
        print('Removing weblink now!') 
        if dask_:                       
            _series = (dd
                               .from_pandas(df["text_body"], npartitions=4)
                               .apply(lambda x: remove_weblink(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else:
            _series = _series.apply(lambda x:  remove_weblink(x))                
    if remove_refrence_: 
        print('Removing refrences now!')
        if dask_:                    
            _series = (dd
                               .from_pandas(_series, npartitions=4)
                               .apply(lambda x: remove_refrence(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else:
             _series = _series.apply(lambda x:  remove_refrence(x))                   
                          
    if remove_brackets_: 
        print('Removing brackets now!')
        if dask_:                        
            _series = (dd
                               .from_pandas(_series, npartitions=4)
                               .apply(lambda x: remove_brackets(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else :
            _series = _series.apply(lambda x:  remove_brackets(x))                 
    if remove_ghost_char_: 
        print('Removing bad charecters now!')
        if dask_:                       
            _series = (dd
                               .from_pandas(_series, npartitions=4)
                               .apply(lambda x: remove_ghost_char(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else:
            _series = _series.apply(lambda x: remove_ghost_char(x))                   
    if remove_extra_spaces_:
        print('Removing Extra spaces now!')
        if dask_:                       
            _series =(dd
                               .from_pandas(_series, npartitions=4)
                               .apply(lambda x: remove_extra_spaces(x),meta=('text_body', 'object'))
                               .compute(scheduler='processes')
                                )
        else:
             _series = _series.apply(lambda x: remove_extra_spaces(x))                 
    print ("completed preprocessing text in {:2f} minutes".format((time.time()-start_time)/60))
    
    return _series
# set "dask_ = True", to operate this function using Dask
# if good computing resources are available(cores>4 and RAM>16GB) then set it 'TRUE', don't try to run dask here, it will take all the RAM and freeze everything
corona_df["text_body"] = preprocess(corona_df,dask_= False)
## install required libraries for BERT
!pip install transformers
!wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar
!tar -xvf scibert_uncased.tar
## import libraries
import torch
from transformers import BertTokenizer, BertModel
# Let's load pretrained BERT model
model_version = 'scibert_scivocab_uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version,do_lower_case=do_lower_case)
## Take 100 sentences to demonstrate BERT embeddings
sent_series= corona_df["text_body"].progress_apply(lambda x:re.split('\.',x))
sent= flatten(sent_series.to_list())[0:100]
#tokenize the sentences -- break them up into word and subwords in the format BERT is comfortable with
tokenized = df["sent"].progress_apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
from tqdm import tqdm
max_len = 0
for i in tqdm(tokenized.values):
    if len(i) > max_len:
        max_len = len(i)

padded= np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
padded.shape
attention_mask = np.where(padded != 0, 1, 0)
attention_mask.shape
## Uncomment for BERT encoddings (takes lots of memory, therefore USE is used for completing tasks)
# input_ids = torch.tensor(padded)  
# attention_mask = torch.tensor(attention_mask)

# with torch.no_grad():
#     last_hidden_states = model(input_ids, attention_mask=attention_mask)
embeddings = last_hidden_states[0][:,0,:].numpy()
embeddings.shape
query = 'incubate with the membranes'
tok = torch.tensor(tokenizer.encode(query)).unsqueeze(0)
with torch.no_grad():
    last_hidden_states = model(tok )
embeddings_query = last_hidden_states[0][:,0,:].numpy()
#embeddings_query.shape
!pip install "tensorflow_hub>=0.6.0"
!pip install "tensorflow>=2.0.0"
import tensorflow as tf
import tensorflow_hub as hub

## Load USE model
module_url = "https://tfhub.dev/google/nnlm-en-dim128/2"
embed = hub.KerasLayer(module_url)
corona_df["text_body_sent"]= corona_df["text_body"].progress_apply(lambda x:re.split('\n',x))
import nltk
sent= nltk.flatten(corona_df["text_body_sent"].to_list())
df_sent = pd.DataFrame(columns=["sent","len"])
df_sent["sent"] = sent
df_sent["len"] = [len(str(i).split()) for i in sent]
df_sent['len'].describe()
## sentences those contains tokens more than 10
sent = [i for i in sent if len(str(i).split())>10] 
# create embeddings for sentences
embeddings = embed(sent)
embeddings.shape
query = [" transmission of virus in community"]
## embedding for query
embeddings_query = embed(query)
embeddings_query.shape
#Calculate cosine similarity of query with all sentences
def cosine_similarity_func(embeddings,embeddings_query):
    '''
    Input:
         embeddings: array or tensor of all sentence embeddings (nX128 for n sentences)
         embeddings_query: array or tensor of query embedding (1X128)
    Output:
         cosine_similarity: cosine similarity of query with each sentence (nX1) 
    '''
    # x.y
    dot_product = np.sum(np.multiply(np.array(embeddings),np.array(embeddings_query)),axis=1)
    
    #||x||.||y||
    prod_sqrt_magnitude = np.multiply(np.sum(np.array(embeddings)**2,axis=1)**0.5, np.sum(np.array(embeddings_query)**2,axis=1)**0.5)
    
    #x.y/(||x||.||y||)
    cosine_similarity  = dot_product/prod_sqrt_magnitude
    return cosine_similarity
# function for recommend text based upon query
def recommended_text(query,embeddings,sent,threshold_min=.95,threshold_max = 1):
    '''
    Input:
         query: list of queries
         embeddings: embeddings of all sentences
         sent:list all sentences
         threshold_min: lower limit of threshold for which sentence is supposed to be similar with query
         threshold_max: upper limit of threshold for which sentence is supposed to be similar with query
         
    Output:
          recommend_text: list of similar sentences with query
    '''
    recommend_text = []
    embeddings_query = embed(query) #create embedding for query
    
    cosine_similarity = cosine_similarity_func(embeddings,embeddings_query) # get cosine similarity with all sentences
    
    # standardize cosine similarity output, Range(0,1)
    standardize_cosine_simi  = (cosine_similarity-min(cosine_similarity))/(max(cosine_similarity)-min(cosine_similarity))
    
    #sort sent based upon cosine similarity score
    sent_prob = list(map(lambda x, y:(x,y), standardize_cosine_simi, sent)) 
    sent_prob.sort(key=lambda tup: tup[0], reverse=True)

    # select sentences by using upper and lower threshold
    for i,j in sent_prob:
        if (i >threshold_min) and (i<=threshold_max):
            recommend_text.append(j)
    return recommend_text  
query = ["range of incubation period of SARS"]
result = recommended_text(query,embeddings,sent,threshold_min=.95)
result
query = ["transmission of virus in community"]
result = recommended_text(query,embeddings,sent,threshold_min = .95)
result[0:30]     #top 30
# effect of environment factors on virus
# effect of weather on virus
# effect of climate on virus
#effect of environment factors on virus
query = ["effect of environment factors on virus"] 
result = recommended_text(query,embeddings,sent,threshold_min = .95)
result[0:50] #top 50 results 
#intervals of virus
#seasonal outbreaks
query = ["seasonal outbreaks" ]
result = recommended_text(query,embeddings,sent,threshold_min =.93)
result
query = ["adhesion to hydrophilic surfaces"]
result = recommended_text(query,embeddings,sent)
result
query = ["persistence of virus on different inanimate surfaces"]
result =recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.97)
result
query = ['implementation of diagnostics and products to improve clinical processes']
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
result[:30] #top 50 results
query = ['Physical science of the coronavirus']
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
result[:30]  # top 30 results
query = ['implementation of diagnostics to improve clinical processes']
results = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
results[0:30] #top 30 results
query = ["desease models of transmission, infection and disease"]
#query = ["desease models of transmission"]
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
result
query = ["tools and studies to monitor phenotypic change and potential adaptation of the virus"]
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
result[:30] # top 3 results
query = ["immune response and immunity"]
results = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.90)
results
query = ["effectiveness of movement control strategies to prevent secondary transmission in health care and community settings"]
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
result[:30] #top 30 results
query = ["effectiveness of personal protective equipment and its usefulness to reduce risk of transmission in health care and community settings"]
result = recommended_text(query,embeddings,sent,threshold_max=1,threshold_min=.95)
len(result)
result[0:30] #top 30 results
