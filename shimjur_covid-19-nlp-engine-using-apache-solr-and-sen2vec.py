import os
from nltk.corpus import stopwords
import os, json
import pandas as pd
from pandas.io.json import json_normalize
from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import SentencesDataset, LoggingHandler, SentenceTransformer
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import *
import logging
from datetime import datetime
import scipy
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import re
import gensim
from nltk import pos_tag, word_tokenize
import scipy
from scipy import spatial
import fasttext
import sent2vec
import os,json
from pandas.io.json import json_normalize
import pandas as pd
path_to_json = '/home/nlvm/kaggle/json_files/'
model = sent2vec.Sent2vecModel()
model.load_model('/home/nlvm/Covid19_Model.bin')
emptdf=pd.DataFrame()
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

fields=['paper_id', 'metadata', 'abstract', 'body_text', 'back_matter']
fields_v1=['cord_uid','title','doi','pmcid','pubmed_id','license','abstract','publish_time','authors','journal','has_full_text','full_text_file']

print('emptdf start')

for index, js in enumerate(json_files):
    with open(os.path.join(path_to_json, js)) as json_file:
        json_text = json.load(json_file)
        #print(index)
        #print(json_text)
        if str(type(json_text))!="<class 'list'>":

            paper_id=json_text["paper_id"]

            try:
                abstract=json_text["abstract"]
            except:
                abstract=""
            body_text=json_text["body_text"]
            title=json_text["metadata"]["title"]
            authors=""

            try:
                authors=json_text["metadata"]["authors"]

            except:
                pass
            xv=pd.DataFrame({'paper_id': [paper_id], 'abstract': [abstract], 'body_text':[body_text],'title':[title],'authors':[authors]})
            #print(xv)

            #xv=json_normalize(json_text)
            #xv=xv[['paper_id', 'abstract', 'body_text', 'back_matter','metadata.title', 'metadata.authors']]
            if not xv.empty:
                emptdf=pd.concat([emptdf,xv])
                
# function to extract text
def process_text(row):
    fullstr=""
    for i in row["body_text"]:
       
        if fullstr:
            fullstr=fullstr+"/n"+i['text']
        else:
            fullstr=fullstr+i['text']
          
    return fullstr
#function to extract abstract
def process_abstract(row):
    fullstr=""
    for i in row["abstract"]:
       
        if fullstr:
            fullstr=fullstr+" "+i['text']
        else:
            fullstr=fullstr+i['text']
          
    return fullstr
import re
emptdf["fullstr"]=emptdf.apply(process_text,axis=1)

emptdf["abstract_fullstr"]=emptdf.apply(process_abstract,axis=1)
# function to flag covid related articles 
def covid_flagfun(row):
    lis=["covid","2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19", "covid 19", "ncov-2019", "sars-cov-2"]
    fullstr=str(row["abstract_fullstr"])
    flag=0
    for i in lis:
        #print(fullstr)
        #print(pattern)
        if i=="covid":
            pattern = re.compile(r"{}".format(i))
        else :
            pattern = re.compile(r"{}*".format(i))
            
        
        if re.search(pattern,fullstr):
            return 1
            #print(1)
emptdf["covid_flagv1"]=emptdf.apply(covid_flagfun,axis=1)
emptdf["covid_flagv1_abs"]=emptdf.apply(covid_flagfun,axis=1)

keywords = [r"2019[\-\s]?n[\-\s]?cov", "2019 novel coronavirus", "coronavirus 2019", r"coronavirus disease (?:20)?19",
            r"covid(?:[\-\s]?19)?", r"n\s?cov[\-\s]?2019", r"sars-cov-?2", r"wuhan (?:coronavirus|cov|pneumonia)",
            r"rna (?:coronavirus|cov|pneumonia)", r"mers (?:coronavirus|cov|pneumonia)", r"influenza (?:coronavirus|cov|pneumonia)",
            r"sars (?:coronavirus|cov|pneumonia)", r"sars", r"mers", r"pandemic", r"pandemics"]

    # Build regular expression for each keyword. Wrap term in word boundaries
regex = "|".join(["\\b%s\\b" % keyword.lower() for keyword in keywords])

def tags(row):
    if re.findall(regex, str(row["fullstr"]).lower()):
        tags = "COVID-19"
    else:
        tags="NON COVID"
    return tags
emptdf["covid_flag"]=emptdf.apply(tags,axis=1)

# creating covid flag

covid_final_df=emptdf[(emptdf["covid_flag"]=="COVID-19") |(emptdf["covid_flagv1"]==1) | (emptdf["covid_flagv1_abs"]==1) ]

lis = list(covid_final_df.iloc[:,6])
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
import numpy as np
np.random.seed(400)
import nltk
nltk.download('wordnet')
stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

# Tokenize and lemmatize
def preprocess(text):
    result=[]
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            # TODO: Apply lemmatize_stemming on the token, then add to the results list
            result.append(lemmatize_stemming(token))
    return result
processed_docs = list(map(preprocess,lis))
dictionary = gensim.corpora.Dictionary(processed_docs)
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
from gensim import corpora, models
tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
#Creating LDA Model
lda_model = gensim.models.LdaMulticore(bow_corpus, 
                                       num_topics=5, 
                                       id2word = dictionary, 
                                       passes = 5,alpha=[0.01]*5,eta=[0.01]*len(dictionary.keys()) ,
                                       workers=2)
import pysolr

solr = pysolr.Solr('http://localhost:8983/solr/covid19_v4', always_commit=True)
import inflect
p = inflect.engine()
import en_core_sci_lg
nlp =  en_core_sci_lg.load()
def process_query_v3(text):
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    lis=['covid','novelcorona','coronavirus']
    for char in text.lower():
        if char not in punctuations:
            no_punct = no_punct + char
    no_punct = ''.join([i for i in no_punct if not i.isdigit()])
    tok=""
    
    doc = nlp(text.lower())
    
    taglis = []
    result=[]
    root_str = ''
    compound_str = ''
    num_nns = 0
    num_nn = 0
    noun_query = []
    noun_query_lis = []
    compound_lis = []
    
    for token in doc:
        taglis.append([token.text,token.dep_,token.tag_,
            token.head])
        
    #print(taglis)
    
    for tag_details in taglis:
        if tag_details[1] == 'compound':
            if tag_details[2] == 'NNS':
                compound_str_nns = tag_details[0]
                compound_str = tag_details[0]
                query_ph='fullstr:"' + tag_details[0] + '"'
                v=solr.search(query_ph)
                num_nns = v.raw_response['response']['numFound']
                
                query_ph='fullstr:"' + p.singular_noun(tag_details[0]) + '"'
                v=solr.search(query_ph)
                num_nn = v.raw_response['response']['numFound']
                if num_nns <= num_nn:
                    compound_str = p.singular_noun(tag_details[0])
                    idx = 0
                    for i in taglis:
                        for a, b in enumerate(i):
                            if str(b) == compound_str_nns:
                                taglis[idx][a] = compound_str
                        idx += 1
                    #taglis[idx] = root_str
            else:
                compound_str = tag_details[0]
            compound_lis.append(compound_str) 
        
    compound_lis = [i for n, i in enumerate(compound_lis) if i not in compound_lis[:n]]
    
    # Adding compound  to the main list (noun_query_lis) 
    for compound in range(len(compound_lis)):
        query_ph='fullstr:"' + ' '.join(x for x in compound_lis[0:len(compound_lis)-compound]) +'"~1000'
        v=solr.search(query_ph)
        #print(query_ph)
        if v.raw_response['response']['numFound'] > 0:
            #print(query_ph)
            for lis_ij in compound_lis[0:len(compound_lis)-compound]:
                noun_query_lis.append(lis_ij)
            break;
    
    #return noun_query_lis
    
            
    noun_query_lis = [i for n, i in enumerate(noun_query_lis) if i not in noun_query_lis[:n]]

    mod_lis=[]
    #return noun_query_lis
    for tag_details in taglis:
        if tag_details[1] == 'nmod' or tag_details[1] == 'amod':
            if tag_details[2] == 'NNS':
                mod_str_nns = tag_details[0]
                mod_str = tag_details[0]
                query_ph='fullstr:"' + tag_details[0] + '"'
                v=solr.search(query_ph)
                num_nns = v.raw_response['response']['numFound']

                query_ph='fullstr:"' + p.singular_noun(tag_details[0]) + '"'
                v=solr.search(query_ph)
                num_nn = v.raw_response['response']['numFound']
                if num_nns <= num_nn:
                    compound_str = p.singular_noun(tag_details[0])
                    idx = 0
                    for i in taglis:
                        for a, b in enumerate(i):
                            if str(b) == mod_str_nns:
                                taglis[idx][a] = mod_str
                        idx += 1
                    #taglis[idx] = root_str
            else:
                mod_str = tag_details[0]
            mod_lis.append(mod_str)

    mod_lis = [i for n, i in enumerate(mod_lis) if i not in mod_lis[:n]]
    #print(mod_lis)
    
    
    # Adding nmod and amods  to the main list (noun_query_lis) 
    
    for mod in range(len(mod_lis)):
        query_ph='fullstr:"' + ' '.join(x for x in mod_lis[0:len(mod_lis)-mod]) +'"~1000'
        v=solr.search(query_ph)
        #print(query_ph)
        if v.raw_response['response']['numFound'] > 0:
            #print(query_ph)
            for lis_ij in mod_lis[0:len(mod_lis)-mod]:
                noun_query_lis.append(lis_ij)
            break;
    noun_query_lis = [i for n, i in enumerate(noun_query_lis) if i not in noun_query_lis[:n]]
    #print(noun_query_lis)
    #return noun_query_lis
 
    head_lis = []
    head_lis_final = []
    noun_head_lis_final = []
    vb_jj_head_lis_final = []
    noun_vb_jj_lis_final = []
    
    
    for tag_details in taglis:
        head_lis.append(str(tag_details[3]))
    
    head_lis_final = [i for n, i in enumerate(head_lis) if i not in head_lis[:n]]
    
    #print('head_lis_final',head_lis_final)
    
    
    # Adding verbs  to the main list (noun_query_lis) 
    
    for head in head_lis_final:
        noun_head_lis = []
        vb_jj_head_lis = []
        for tag_details in taglis:
            if str(tag_details[1]) != 'ROOT' and 'NN' in str(tag_details[2]) and str(tag_details[3]) == str(head) and str(tag_details[0]) != str(head):
                noun_head_lis.append(tag_details[0])
            elif (str(tag_details[2]) == 'VB' or str(tag_details[2]) == 'JJ') and str(tag_details[3]) == str(head) and str(tag_details[0]) != str(head):
                vb_jj_head_lis.append(tag_details[0])
        if noun_head_lis:
            noun_vb_jj_lis_final.append([head,noun_head_lis])
        
        if vb_jj_head_lis:
            noun_vb_jj_lis_final.append([head,vb_jj_head_lis])
    
    for i in noun_vb_jj_lis_final:
        if len(i[1]) == 0:
            noun_vb_jj_lis_final.remove(i)
    
    #print('noun_vb_jj_lis_final', noun_vb_jj_lis_final)
    for i in noun_vb_jj_lis_final:
        for j in range(len(i[1])):
            query_ph='fullstr:"'+ ' '.join(x for x in i[1][0:len(i[1])-j]) + ' '  + str(i[0]) +'"~1000'
            v=solr.search(query_ph)
            #print(query_ph)
            if v.raw_response['response']['numFound'] > 0:
                #print(query_ph)
                
                for lis_ij in i[1][0:len(i[1])-j]:
                    noun_query_lis.append(lis_ij)
                noun_query_lis.append(str(i[0]))
                
                break;
    # Adding root  to the main list (noun_query_lis) 
    for tag_details in taglis:
        if tag_details[1] == 'ROOT':
            if tag_details[2] == 'NNS':
                root_str_nns = tag_details[0]
                root_str = tag_details[0]
                query_ph='fullstr:"' + tag_details[0] + '"'
                v=solr.search(query_ph)
                num_nns = v.raw_response['response']['numFound']
                
                query_ph='fullstr:"' + p.singular_noun(tag_details[0]) + '"'
                v=solr.search(query_ph)
                num_nn = v.raw_response['response']['numFound']
                if num_nns <= num_nn:
                    root_str = p.singular_noun(tag_details[0])
                    idx = 0
                    for i in taglis:
                        for a, b in enumerate(i):
                            if str(b) == root_str_nns:
                                taglis[idx][a] = root_str
                        idx += 1
                    #taglis[idx] = root_str
            else:
                root_str = tag_details[0]
        noun_query.append(root_str) 
        
        break;
        
    #print(taglis)
    noun_query_lis.append(root_str) 
    
    #return (noun_query_lis)
    
    for noun in range(len(noun_query_lis)):
        noun_query_temp = noun_query_lis
        query_ph='fullstr:"' + ' '.join(x for x in noun_query_temp[0:len(noun_query_temp)-noun]) +'"~1000'
        v=solr.search(query_ph)
        #print(query_ph)
        if v.raw_response['response']['numFound'] > 0:
            #print(query_ph)
            break;
            
    noun_query_lis = noun_query_temp[0:len(noun_query_temp)-noun]
    
    noun_query_lis = [i for n, i in enumerate(noun_query_lis) if i not in noun_query_lis[:n]]
    #print(stop_words)   
    
    noun_query_lis = [i for i in noun_query_lis if not i.lower() in gensim.parsing.preprocessing.STOPWORDS and len(i) > 2]
    
    # Returning keywords 
    return ' '.join(x for x in noun_query_lis)

def query_formation(query_str,dictionary):
    query_str_orig = query_str
    def set_tf_idf(row):
        
        try:
            token_id=dictionary.token2id
            idx=token_id[row["processed"]]
            for i in corpus_tfidf[0]:
                if i[0]==idx:
                    return i[1]
        except:
            return 0
    global str    
 
    def preprocess(text):
        result=[]
        wnl = WordNetLemmatizer()
        for word, tag in pos_tag(word_tokenize(query_str)):
            wntag = tag[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
            if word not in gensim.parsing.preprocessing.STOPWORDS and len(word) > 2:
                if not wntag:
                    result.append(word)
                else:
                    result.append(wnl.lemmatize(word, wntag))
        #print(result)
#         for token in gensim.utils.simple_preprocess(text) :
#             if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2:
#                 # TODO: Apply lemmatize_stemming on the token, then add to the results list
#                 result.append(wnl.lemmatize(token,)
        return result
    def lemmatize_stemming_df(row):
        return stemmer.stem(WordNetLemmatizer().lemmatize(row["original"], pos='v'))
    query_lis=[]
    query_lis_final=[]
    #for i in nlp(query_str).ents:
        #query_lis.append(str(i))
    query_lis = preprocess(query_str)
    query_str=" ".join(query_lis)
    #print (query_str)
    query_lis_final.append(query_str)
    df=pd.DataFrame( gensim.utils.simple_preprocess(query_lis_final[0]))
    df.columns=["original"]
    df['processed'] = df.apply(lemmatize_stemming_df,axis=1) 
    a=[list(df["processed"])]
    bow_query = [dictionary.doc2bow(doc) for doc in a]
    corpus_tfidf = tfidf[bow_query]
    df['tf_idf'] = df.apply(set_tf_idf,axis=1)
    df_filtered=df[df['tf_idf']>0]
    df_filtered.sort_values(by=['tf_idf'], ascending=False,inplace=True)
    df_filtered["boost"]=df_filtered["tf_idf"]/df_filtered["tf_idf"].sum()*1000
    df_filtered.reset_index(drop=True)
    phrase_query=""
    #print(df_filtered)
    for i in range(len(df_filtered)):
        phrase_query=phrase_query+" "+df_filtered.iloc[i,0]
        if df_filtered.iloc[0:i,2].sum()>.95*df_filtered["tf_idf"].sum():
            break
    
    # Construction of Phrase Query
    phrase_query=phrase_query.strip()
    phrase_list=phrase_query.strip().split()
    lis=[]
    for i in range(len(phrase_query.strip().split())):
        #print(i,len(phrase_query.strip().split())," ".join(phrase_list[0:-1-i]))
        if i+2==len(phrase_query.strip().split()):
            break
        else:
            lis.append(" ".join(phrase_list[0:-1-i]) ) 
    lis.insert(0,process_query_v3(query_str_orig))    
    
    # Construction of term Query
    term_lis=[]
    term_lis=[str(x[0]) + "*^"  +str(int(x[1])) for x in zip(list(df_filtered["processed"]),list(df_filtered["boost"]))]
    return lis,term_lis

def getSolr_results(query_string,dictionary):
    lis=["covid","2019-ncov", "2019 novel coronavirus", "coronavirus 2019", "coronavirus disease 19", "covid-19", "covid 19", "ncov-2019", "sars-cov-2"]
    result_list = []
    idx = 0
    query_string=query_string.replace("corona virus", "coronavirus", 1)
    query_string=query_string.replace("covid-19", " ", 1)
    for i in lis:
        query_string=query_string.replace(i, "", 1)

    phrase,term = query_formation(query_string,dictionary)
    #topic = query_topic(query_string,dictionary_title,model)
    
    #print(term)
    for i in range(math.ceil(len(phrase)/2)):
        if i==0:
            for k in lis:
                phrase_query = 'fullstr:"' + phrase[i] + " " + k+'"~1000'
                results = solr.search(phrase_query,rows=3)

                for result in results:
                    result_list.append([result['paper_id'][0],idx])
                #print(phrase_query)
                #print(len(results))
                if(len(result_list))>=3:
                    query_string_new=query_string.lower()+" "+ k
                    #print(query_string_new)
                    break




        elif len(result_list) ==0 or i!=0:
            phrase_query = 'fullstr:"' + phrase[i] +'"~1000'
            results = solr.search(phrase_query,rows=3)
            query_string_new=query_string.lower()


        idx += 1
        for result in results:
            result_list.append([result['paper_id'][0],idx])
            #print(result_list)

    if len(result_list) < 50:
        term_query = ""
        for i in term:
            if term_query:
                term_query = term_query + ", " + i
            else:
                term_query = i
        term_query = 'fullstr:(' + term_query + ')'
        #print(term_query)
        results = solr.search(term_query,rows=3)
        for result in results:
            result_list.append([result['paper_id'][0],idx+1])
            #print(result_list)
    query_string_new=query_string_new.replace("covid", "covid-19", 1)
    articles_final=selec_articles(query_string_new,result_list) 
    para_final=selec_para(query_string_new,articles_final)
    final_summary=final_return(query_string_new,articles_final,para_final)
    return final_summary

def selec_articles(query_str,articles_lis):
    def cosine_test(row):
        query_vec=model.embed_sentence(query_str)
        #print(type(row["para_vector"]))
        distances = scipy.spatial.distance.cdist(query_vec, row["para_vect_fullstr"], "cosine")[0]
        return distances[0]
    rank1=[]
    rank_others=[]
    dist=[]
    dist1=[]
    query_vec=model.embed_sentence(query_str)
    #print(query_vec.shape)
    for i in articles_lis:
        if i[1]==0:
            rank1.append(i[0])
        else :
            rank_others.append(i[0])
    rank1_artc=covid_final_df[covid_final_df.paper_id.isin(rank1)]

    #rank1_array=np.array(rank1_artc.iloc[:,[0,10]])
    #print(rank1_array[0][1])
    #rank1_array[2]=scipy.spatial.distance.cdist(query_vec, rank1_array[1], "cosine")[0]
    #print(rank1_array[1].shape)
    #print(rank1_array)
    if rank1_artc.shape[0]<3:
        rankothers_artc=covid_final_df[covid_final_df.paper_id.isin(rank_others)]
    #rank1_artc["test_distance"]= rank1_artc.apply(cosine_test,axis=1)
    #return rankothers_artc
    for h in range(rank1_artc.shape[0]):
        dist.append([scipy.spatial.distance.cdist(query_vec, rank1_artc.iloc[h,10], "cosine")[0][0],rank1_artc.iloc[h,0]])
    if rank1_artc.shape[0]<3:
        for k in range(rankothers_artc.shape[0]):
            dist1.append([scipy.spatial.distance.cdist(query_vec, rankothers_artc.iloc[k,10], "cosine")[0][0],rankothers_artc.iloc[k,0]])
    #rank1_artc=rank1_artc.sort_values(by=['test_distance'])
    #a=rank1_artc.iloc[:,0]
    dist=sorted(dist, key = lambda x: x[0])
    if dist1:
        sorted(dist1, key = lambda x: x[0])



    return dist+dist1

for index, row in emptdf.iterrows():
    x={}
    for i in row["body_text"]:
        if i['text']:
            x["paper_id"]=row["paper_id"]
            x["paragraph"]=i['text']
            x_df=pd.DataFrame.from_dict(x,orient='index').T
            consolidated_df=pd.concat([consolidated_df, x_df])

consolidated_df.to_json("/home/nlvm/kaggle/consolidated_df.json",orient='records')
    
import json
with open("/home/nlvm/kaggle/consolidated_df.json") as json_file:
    s=json.load(json_file)
para_df=json_normalize(s)
para_df.drop_duplicates(subset='paragraph', keep="last",inplace=True)

#Extracting Sentences from Paragraph
def sent_token(row):
    sentences = nltk.sent_tokenize(row["paragraph"])
    return sentences

#Creating Sentence Vector list
def sent_vector_lis(row):
    lis=[]
    for i in row["sent_tok"]:
        vec=model.embed_sentence(i)
        lis.append(vec)
    return lis
para_df['sent_tok'] = para_df.apply(sent_token,axis=1)
para_df["sent_vector_lis"]= para_df.apply(sent_vector_lis,axis=1)

#Aggregating sentence vector to paragraph by taking Mean of sentence vectors 
def sent_vector(row):
    lis=[]
    for i in row["sent_tok"]:
        vec=model.embed_sentence(i)
        lis.append(vec)
    lis_arr=np.array(lis)
    lis_arr_mean=lis_arr.mean(axis=0)
    return lis_arr_mean
para_df["sent_vect_mean"]= para_df.apply(sent_vector,axis=1)

para_df.reset_index(drop=True,inplace=True)
para_df=para_df.reset_index()

#Function returns the best paragrapgh using cosine distance
def selec_para(query_str,articles_shortlisted):


    para_article=[]
    dist_final=[]
    query_vec=model.embed_sentence(query_str)
    #print(query_vec.shape)
    #print(articles_shortlisted)
    for i in articles_shortlisted:

        para_artc=para_df[para_df.paper_id==i[1]]

        dist=[]
        try:
            for h in range(para_artc.shape[0]):
                dist.append([scipy.spatial.distance.cdist(query_vec, para_artc.iloc[h,6], "cosine")[0][0],para_artc.iloc[h,0]])
            dist=sorted(dist, key = lambda x: x[0])
            dist_final.append(dist[0])
        except:
            pass
    dist_final=sorted(dist_final, key = lambda x: x[0])    
    return dist_final
    #return para_artc.shape
    #rank1_array=np.array(rank1_artc.iloc[:,[0,10]])
    #print(rank1_array[0][1])
    #rank1_array[2]=scipy.spatial.distance.cdist(query_vec, rank1_array[1], "cosine")[0]
    #print(rank1_array[1].shape)
    #print(rank1_array)
    #rank1_artc["test_distance"]= rank1_artc.apply(cosine_test,axis=1)
 
    
    #rank1_artc=rank1_artc.sort_values(by=['test_distance'])
    #a=rank1_artc.iloc[:,0]


    return dist_final

#Function returns the best sentences using Cosine distance
def best_sent(query_str,para_results):
    query_vec=model.embed_sentence(query_str)

    para_id=[]
    for j in para_results:
        para_id.append(j[1])
    #print(para_id)
    pararesults_df=para_df[para_df["index"].isin(para_id)]
    #print(pararesults_df)
    idx=0
    final_highlight=[]
    try:
        for index, row in pararesults_df.iterrows():
            lis_para=[]
            #print("parse:",idx)
            #print("sent: ",row["sent_tok"])
            #print("sent_vect",len(row["sent_vector_lis"]))
            for i in range(len(row["sent_vector_lis"])):

                lis_para.append([scipy.spatial.distance.cdist(query_vec, row["sent_vector_lis"][i], "cosine")[0][0],i])

            lis_para=sorted(lis_para, key = lambda x: x[0])
            #print(lis_para)
            #print(len(row["sent_vector_lis"]))
            templis=[]
            for l in range(math.ceil(len(lis_para)/3)):
                #print("ceil:",len(lis_para))
                #regex = re.compile('.*({}).*'.format(row["sent_tok"][lis_para[l][1]]))
                #print(row["sent_tok"][lis_para[l][1]])
                c=row["paragraph"].find(row["sent_tok"][lis_para[l][1]])
                templis.append([c,len(row["sent_tok"][lis_para[l][1]])])
            templis=sorted(templis, key = lambda x: x[0])
            #print(templis)
            final_highlight.append([row["paper_id"],[templis[0][0],templis[-1][0]+templis[-1][1]]])
            idx+=1
        return final_highlight
    except:
        return final_highlight
with open("/home/nlvm/kaggle/covid_summarize_auth_insti_final.json") as json_file:
    s=json.load(json_file)

covid_summarize_auth_insti_final=json_normalize(s)

#Function to fetch all the results converted to JSON and feed to API.
def final_return(query_str,articles_results,para_results):
    articles_id=[]
    para_id=[]
    sumr_str=[]
    for i in articles_results:
        articles_id.append(i[1])
    for j in para_results:
        para_id.append(j[1]) 
    #print(para_id)
    article_df=covid_final_df[covid_final_df.paper_id.isin(articles_id)].iloc[:,[0,3,6]] 
    pararesults_df=para_df[para_df["index"].isin(para_id)].iloc[:,[1,2]]
    auth_results = covid_summarize_auth_insti_final[covid_summarize_auth_insti_final['paper_id'].isin(articles_id)]
    sent_highlights=best_sent(query_str,para_results)

    if sent_highlights:
        sent_high_df=pd.DataFrame(sent_highlights,columns=["paper_id","pos"])
#     for index, row in article_df.iterrows():
#         if row["abstract_fullstr"]=="":
#             sumr_str.append([row["fullstr"],row["paper_id"]])
#         else:
#             sumr_str.append([row["abstract_fullstr"],row["paper_id"]])
            

    final_summary1=pd.merge(article_df,pararesults_df,on="paper_id",how="inner")
    final_summary2=pd.merge(final_summary1,auth_results,on="paper_id",how="left")
    if sent_highlights:
        final_summary=pd.merge(final_summary2,sent_high_df,on="paper_id",how="left")
    else:
        final_summary=final_summary2

#     idx=0
#     for sumr in sumr_str:
#         #print(sumr[1])
#         #summarize( sumr[0], word_count=300)
#         print(sumr)
    return final_summary.to_json(orient="records")
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_cors import CORS, cross_origin
app = Flask(__name__)
app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:5000"}})

@app.route('/covid',methods=['POST'])
@cross_origin(origin='localhost',headers=['Content- Type','Authorization'])
def results():
    data=request.get_json(force=True)
    query=str(data["query"])
    covid_json = getSolr_results(query,dictionary)
    #covid_final_df = covid_final(query,dictionary,dictionary_title,lda_model)
    #covid_json = covid_final_df.to_json(orient="records")
    return covid_json


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)