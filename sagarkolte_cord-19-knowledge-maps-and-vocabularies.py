import pandas as pd

import numpy as np

from os import listdir

from os.path import isfile, join

import json

import re

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

from nltk import word_tokenize

stop_words = set(stopwords.words('english'))

from nltk.tokenize import RegexpTokenizer

import spacy

nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])

nlp.max_length = 1500000

from collections import Counter

import seaborn as sns

import matplotlib.pyplot as plt

import networkx as nx

from itertools import combinations 

import functools

import operator

stop_words = set(stopwords.words('english'))

import os

import pathlib

from shutil import copyfile

from shutil import copy

from nltk import sent_tokenize, word_tokenize

pd.options.display.max_colwidth = 2000
#All the file path information

mypath = '../input/CORD-19-research-challenge/'

docdir1= 'biorxiv_medrxiv/biorxiv_medrxiv/pdf_json/'

docdir2 = 'comm_use_subset/comm_use_subset/pdf_json/'

docdir3 = 'custom_license/custom_license/pdf_json/'

docdir4 = 'noncomm_use_subset/noncomm_use_subset/pdf_json/'

root = './'

dir_list = [docdir1,docdir2,docdir3,docdir4]



#####################################################-------------------------------->Extracting Relevant Metadata from a given json



def create_refkeys(n):#-------------------------------->Used to extract references

    key_list = ['BIBREF'+str(i) for i in range(n) ]

    return key_list



def journals_reffered(p_dict):#------------------------->Gets all the journal names in the bibliography

    try:

        n = len(p_dict['bib_entries'])

        ref_list = create_refkeys(n)

        j_ref = [p_dict['bib_entries'][ref]['venue'] for ref in ref_list]

    except:

        j_ref = 'NA'

    return j_ref



def get_sha(p_dict):#------------------------------------>Gets the SHA of the article

    try:

        sha = p_dict['paper_id']

    except:

        sha = 'NA'

    return sha



def get_title(p_dict):#---------------------------------->Gets the title

    try:

        title = p_dict['metadata']['title']

    except:

        title = 'NA'

    return title



def get_doi_meta(text):#---------------------------------->Guesses the doi of the article from the text

    try:

        #file_str = str(json_to_dict(sub_dir,file_name)).replace('\'','')

        doi = re.findall('doi.org/(.*?)[\s]',text)[0]

    except:

        doi = 'NA'

    return doi



def get_first_author(p_dict):#------------------------------>Gets the first author

    try:

        first = p_dict['metadata']['authors'][0]['first']

        last =  p_dict['metadata']['authors'][0]['last']

        name = first+' '+last

    except:

        name = 'NA'

    return name



def get_institution(p_dict):#-------------------------------->Gets the institution to which the first author is affiliated

    try:

        inst = p_dict['metadata']['authors'][0]['affiliation']['institution']

    except:

        inst = 'NA'

    return inst



def text_to_vec_meta(text,task):#----------------------------->Creates a vector given an abstract and a vocabulary(task)

    l = tok_pun_lem(text)

    c_dict = Counter(l)

    c_dict = {k:[c_dict[k]] for k in task }

    df = pd.DataFrame(c_dict)

    return df



def get_abstract(p_dict):#------------------------------------>Gets the abstract of the article

    try:

        abstract = p_dict['abstract'][0]['text']

    except:

        abstract = 'NA'

    return abstract



def create_row_meta(subdir,file_name,task):#------------------->Puts all the metadata into a single row with the word counts

    #print('running create_row_meta...')

    p_dict = json_to_dict(subdir,file_name)

    text =  str(p_dict).replace('\'','')

    sha = get_sha(p_dict)

    title = get_title(p_dict)

    author = get_first_author(p_dict)

    institution = get_institution(p_dict)

    doi = get_doi_meta(text)

    abstract = get_abstract(p_dict)

    p_dict = get_abstract(p_dict)

    text =  str(p_dict).replace('\'','')

    df1 = text_to_vec_meta(text,task)

    df2 = pd.DataFrame({'SHA':[sha], 

                        'Title':[title], 

                        'First Author':[author], 

                        'Institution':[institution], 

                        'DOI':[doi], 

                        'Abstract':[abstract]})

    df = pd.concat([df2,df1],axis = 1)

    return df



def sha_to_meta(sha):#------------------->Puts all the metadata into a single row with the word counts

    file_path = sha_to_file_path(sha)

    p_dict = json_to_dict_1(file_path)

    text =  str(p_dict).replace('\'','')

    sha = get_sha(p_dict)

    title = get_title(p_dict)

    author = get_first_author(p_dict)

    institution = get_institution(p_dict)

    doi = get_doi_meta(text)

    abstract = get_abstract(p_dict)

    p_dict = get_abstract(p_dict)

    #text =  str(p_dict).replace('\'','')

    #df1 = text_to_vec_meta(text,task)

    df2 = pd.DataFrame({'SHA':[sha], 

                        'Title':[title], 

                        'First Author':[author], 

                        'Institution':[institution], 

                        'DOI':[doi], 

                        'Abstract':[abstract]})

    #df = pd.concat([df2,df1],axis = 1)

    return df2.T







def meta_pipeline(task):#--------------------------------------------------------------->Gives the final dataframe for all articles which have an abstract

    task_number = task[0]

    subtask_number = task[1]

    task = task[2]

    g = lambda subdir,task:pd.concat([create_row_meta(subdir,file_name,task) for file_name in get_file_list(subdir,mypath)])

    df_list = list(map(g,dir_list,[task]*len(dir_list)))

    df_final = pd.concat(df_list)

    o = create_task_dir(task_number,subtask_number)

    dest_file = root+o+'/'+'MetaTable_'+str(task_number)+'_'+str(subtask_number)+'.csv'

    try:

        os.makedirs(root+o)

    except:

        df_final.to_csv(dest_file)    

    df_final.to_csv(dest_file)

    return df_final





def word_selector(df,word_list):#------------------------------------------------------->Returns a data frame containing articles with words specified in the list

    df['flag'] = df.apply(lambda row:np.prod(row[word_list]), axis=1)

    df = df[df['flag']!=0]

    df = df.drop('flag',axis = 1)

    df = df.loc[:, (df != 0).any(axis=0)]

    df = df.sort_values(word_list,ascending = False)

    df = df[0:15]

    df = df.loc[:, (df != 0).any(axis=0)]

    return df



def get_top_matches(df,word_list,task):

    task_number = task[0]

    subtask_number = task[1]

    #df['sum'] = df.apply(lambda row:np.sum(row[word_list]),axis=1)

    #df = df.sort_values('sum', ascending=False)

    #df = df.head(5)

    o = create_task_dir(task_number,subtask_number)

    dest_file = root+o+'/'+'TopMatches_'+str(task_number)+'_'+str(subtask_number)+'.csv'

    df.to_csv(dest_file)

    return df

    



########################################################-------------------------------->Extracting Relevant Metadata from a given json
#######################################################--------------------------------->Natural Language Processing module

def get_file_list(sub_dir,mypath=mypath):#---------------------------------------------->fetches the list of files from a subdirectory.

    #print('running get_file_list...')

    #print(mypath+sub_dir)

    #print(listdir(mypath+sub_dir))

    onlyfiles = [f for f in listdir(mypath+sub_dir) if isfile(join(mypath+sub_dir, f))]

    #print(len(onlyfiles))

    return onlyfiles



def get_file_list_1(sub_dir,mypath):#---------------------------------------------->fetches the list of files from a subdirectory.

    (curdir,dir_names, onlyfiles) = next(os.walk(mypath+sub_dir+'/pdf_json'))

    print('running get_file_list...')

    print(mypath+sub_dir)

    print(listdir(mypath+sub_dir))

    #onlyfiles = [f for f in listdir(mypath+sub_dir) if isfile(join(mypath+sub_dir, f))]

    #print(len(onlyfiles))

    return onlyfiles







def json_to_dict(sub_dir,filename,mypath=mypath):#--------------------------------------->Converts json flies to dict.

    full_path = mypath+sub_dir+filename

    with open(mypath+sub_dir+filename, "r") as read_file:

        data = json.load(read_file)

    return data



def json_to_dict_1(full_path):#---------------------------------------------------------->Converts json flies to dict.

    with open(full_path, "r") as read_file:

        data = json.load(read_file)

    return data





def get_doi(sub_dir,file_name):#---------------------------------------------------------->Guesses the doi of the article from the text

    try:

        file_str = str(json_to_dict(sub_dir,file_name)).replace('\'','')

        doi = re.findall('doi.org/(.*?)[\s]',file_str)[0]

    except:

        doi = 'NA'

    return doi



def tok_pun_lem(text):#------------------------------------------------------------------->Tokenizes, Removes Punctuation and Lemmatizes

    tokenizer = RegexpTokenizer(r'\w+')

    result = tokenizer.tokenize(text.lower())

    new_sentence = list(filter(lambda x:False if x in stop_words else True,result))

    new_sentence_s = " ".join(new_sentence)

    doc = nlp(new_sentence_s)

    new_sentence_l = [token.lemma_ for token in doc]

    return new_sentence_l



def abst_to_vec(sub_dir,task):#----------------------------------------------------------->Creates a vector given an abstract and a vocabulary(task)

    

    file_list = get_file_list(sub_dir)

    print('old len',len(file_list))

    file_list = filter(lambda file_name: True if len(json_to_dict(sub_dir,file_name)['abstract'])>0 else False, file_list) 

    text_list = [json_to_dict(sub_dir,file_name)['abstract'][0]['text'].lower() for file_name in file_list]

    print('new len',len(text_list))

    l_list = [tok_pun_lem(text) for text in text_list]

    c_dict_list = [Counter(l) for l in l_list]

    c_dict_list = [{k:[c_dict[k]] for k in task } for c_dict in c_dict_list]

    df_list = [pd.DataFrame(c_dict) for c_dict in c_dict_list]

    final_df = pd.concat(df_list)

    print('final len',len(final_df))

    return final_df



def text_to_vec(sub_dir,task):#----------------------------------------------------------->Creates a vector given an abstract and a vocabulary(task)

    file_list = get_file_list(sub_dir)

    print('old len',len(file_list))

    text_list = [str(json_to_dict(sub_dir,file_name)).lower().replace('\'','') for file_name in file_list]

    print('new len',len(text_list))

    l_list = [tok_pun_lem(text) for text in text_list]

    c_dict_list = [Counter(l) for l in l_list]

    c_dict_list = [{k:[c_dict[k]] for k in task } for c_dict in c_dict_list]

    df_list = [pd.DataFrame(c_dict) for c_dict in c_dict_list]

    final_df = pd.concat(df_list)

    print('final len',len(final_df))

    return final_df

####################################################--------------------------------------->Natural language processing module

def abst_pipeline(task):

    df_list = [abst_to_vec(subdir,task) for subdir in dir_list]

    df = pd.concat(df_list)

    return df



def text_pipeline(task):

    df_list = [abst_to_vec(subdir,task) for subdir in dir_list]

    len_list = [len(x) for x in df_list]

    print(len_list)

    df = pd.concat(df_list)

    print(len(df))

    return df

###############################################--------------------------------------------->Data Extraction Pipeline

def get_full_paths(root):

    path_list = [str(i) for i in list(pathlib.Path('../').glob('**/*'))]    

    return path_list



def sha_to_file_path(sha):

    path_list = get_full_paths(root)

    g = lambda x: sha in x 

    file_path = list(filter(g,path_list))

    return file_path[0]



def get_sentences(sha,word):

    text = sha_to_text(sha)

    text = re.sub('[{:\}]', '', text) 

    sentences = sent_tokenize(text) 

    g = lambda x : True if word in word_tokenize(x) else False

    sentences = list(filter(g,sentences))

    #sentences = [i for i in sent_tokenize(text) if "word" in word_tokenize(i)]

    

    return sentences



def sha_to_text(sha):#---------------------------------------------------------------------->Gets the full text of the article given the SHA.

    full_path = sha_to_file_path(sha)

    with open(full_path, "r") as read_file:

        data = json.load(read_file)

    text = str(data).lower().replace('\'','')

    return text







def save_sha(sha,task_number,subtask_number):

    o = create_task_dir(task_number,subtask_number)

    file_name = sha_to_file_path(sha)

    dest_file = root+o+'/'+sha+'.json'

    os.makedirs(os.path.dirname(dest_file), exist_ok=True)

    try:

        copy(file_name,dest_file)#----copyfile

    except:

        return None

    return None



def create_task_dir(task_number,subtask_number):

    o = 'Task_'+str(task_number)+'/'+'Sub_Task_'+str(subtask_number)

    return o



def sha_pipeline(df,task_number,subtask_number):

    #sha_list = list(df['SHA'])

    sha_list = df.index.values.tolist()

    l = list(map(save_sha,sha_list,[task_number]*len(sha_list),[subtask_number]*len(sha_list)))

    return None

##################################################------------------------------------------>Data Extraction Pipeline
task11 = [1,1,['incubation','age','convalescent','transmission','contagious','carrier','convalescence']]

task12 = [1,2,['transmission','asymptomatic','shedding','shed']]

task13 = [1,3,['transmission','seasonality','seasonal']]

task14 = [1,4,['charge','distribution','coronavirus','persistence','protein','surface','adhesion']]

task15 = [1,5,['persistence','stability','sputum','urine','feces','fecal','blood','nasal','discharge','coronavirus']]

task16 = [1,6,['persistence','surface','virus']]

task17 = [1,7,['history','virus','shed','shedding','natural']]

task18 = [1,8,['diagnosis','diagnostic','clinical','products','process']]

task19 = [1,9,['disease','model','infection','transmission']]

task110 = [1,10,['phenotypic','change','adapt','adaptation','virus','morphology','gene','genetic']]

task111 = [1,11,['immune','immunity','response','virus','cronavirus']]

task112 = [1,12,['protective','protect','equipment','risk','transmission','health','care','community']]

task113 = [1,13,['environment','transmission','virus']]
task = task15#---------------------->Change the task here
task_number = task[0]

subtask_number = task[1]

df = meta_pipeline(task)#----------->Create the main dataframe for the subtask
cluster_df = df[task[2]]

cluster_df['sum'] = cluster_df.apply(lambda row:sum(np.array(row)),axis = 1)

cluster_df = cluster_df[cluster_df['sum']>=2]

cluster_df = cluster_df.drop('sum',axis=1)
g = sns.clustermap(cluster_df,cmap='PuBu_r',metric = 'cosine')

o = create_task_dir(task_number,subtask_number)

dest_file = root+o+'/'+'MainClusterMap_'+str(task_number)+'_'+str(subtask_number)+'.png'

g.savefig(dest_file, dpi=400,bbox_inches='tight')
word_list = ['persistence','coronavirus']#------------->study map above and select words here

df1 = word_selector(df,word_list)
df1_cluster = df1.set_index("SHA", inplace = False) 

df1_cluster = df1_cluster.drop(['Title','First Author','Institution','DOI','Abstract'],axis=1)

g = sns.clustermap(df1_cluster,cmap='PuBu_r',metric = 'cosine',yticklabels=True)

o = create_task_dir(task_number,subtask_number)

dest_file = root+o+'/'+word_list[0]+'_'+word_list[-1]+'_'+str(task_number)+'_'+str(subtask_number)+'.png'

g.savefig(dest_file, dpi=400,bbox_inches='tight')
df_save = get_top_matches(df1_cluster,word_list,task)

sorted_sha_list = list(df_save.sort_index().index.values)

meta_df_list = list(map(sha_to_meta,sorted_sha_list))

meta_df_final = pd.concat(meta_df_list)

task_number = task[0]

subtask_number = task[1]

o = create_task_dir(task_number,subtask_number)

dest_file_html = root+o+'/'+'sha_df'+str(task_number)+'_'+str(subtask_number)+'.html'

meta_df_final.to_html(dest_file)

print ("please find the list of relevant papers saved as:"+dest_file_html)
print('\n\n'.join(get_sentences('4bc6d312effedb4dafb7bca22236c006ad1f7136','children')))