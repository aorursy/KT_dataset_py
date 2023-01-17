# Installation of package including BERT method 

!pip install transformers
## DO NOT RUN HERE

# This code has been use to prepare the df_covid data frame representing the 27 March 2020 version of the COVID-19 Open Research Dataset Challenge (CORD-19) data base.



# Process all json information and update df_covid data frame from the input path in the working path

import os

import json

import glob as gl

import sys

import numpy as np

import pandas as pd

sys.path.insert(0, "../")



class FileReader:

    def __init__(self, file_path):

        with open(file_path) as file:

            content = json.load(file)

            self.paper_id = content['paper_id']

            self.abstract = []

            self.body_text = []

            # Abstract

            try:

                for entry in content['abstract']:

                    self.abstract.append(entry['text'])

                self.abstract = '\n'.join(self.abstract)

            except:

                self.abstract.append(['Not provided.'])

            # Body text

            try:

                for entry in content['body_text']:

                    self.body_text.append(entry['text'])

                self.body_text = '\n'.join(self.body_text)

            except:

                self.body_text.append(['Not provided.'])    

    def __repr__(self):

        return f'{self.paper_id}: {self.abstract[:200]}... {self.body_text[:200]}...'

    

def get_breaks(content, length):

    data = ""

    words = content.split(' ')

    total_chars = 0



    # add break every length characters

    for i in range(len(words)):

        total_chars += len(words[i])

        if total_chars > length:

            data = data + "<br>" + words[i]

            total_chars = 0

        else:

            data = data + " " + words[i]

    return data



root_path = '/kaggle/input/CORD-19-research-challenge/'

# Just set up a quick blank dataframe to hold all these medical papersb. 



json_filenames = gl.glob(f'{root_path}/***/**/*.json', recursive=True)



metadata_path = f'{root_path}/metadata.csv'

meta_df = pd.read_csv(metadata_path, dtype={

    'pubmed_id': str,

    'Microsoft Academic Paper ID': str, 

    'doi': str

})



all_json = json_filenames



dict_ = {'paper_id': [], 'abstract': [], 'body_text': [], 'authors': [], 'title': [], 'journal': [], 'abstract_summary': [],'publish_time':[]}

for idx, entry in enumerate(all_json):

    if idx % (len(all_json) // 10) == 0:

        print(f'Processing index: {idx} of {len(all_json)}')

    content = FileReader(entry)

    

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    # no metadata, skip this paper

    if len(meta_data) == 0:

        continue

    

    dict_['paper_id'].append(content.paper_id)

    dict_['abstract'].append(content.abstract)

    dict_['body_text'].append(content.body_text)

    

    # also create a column for the summary of abstract to be used in a plot

    if len(content.abstract) == 0: 

        # no abstract provided

        dict_['abstract_summary'].append("Not provided.")

    elif len(content.abstract.split(' ')) > 100:

        # abstract provided is too long for plot, take first 300 words append with ...

        info = content.abstract.split(' ')[:100]

        summary = get_breaks(' '.join(info), 40)

        dict_['abstract_summary'].append(summary + "...")

    else:

        # abstract is short enough

        summary = get_breaks(content.abstract, 40)

        dict_['abstract_summary'].append(summary)

        

    # get metadata information

    meta_data = meta_df.loc[meta_df['sha'] == content.paper_id]

    

    try:

        # if more than one author

        authors = meta_data['authors'].values[0].split(';')

        if len(authors) > 2:

            # more than 2 authors, may be problem when plotting, so take first 2 append with ...

            dict_['authors'].append(". ".join(authors[:2]) + "...")

        else:

            # authors will fit in plot

            dict_['authors'].append(". ".join(authors))

    except Exception as e:

        # if only one author - or Null valie

        dict_['authors'].append(meta_data['authors'].values[0])

    

    # add the title information, add breaks when needed

    try:

        title = get_breaks(meta_data['title'].values[0], 40)

        dict_['title'].append(title)

    # if title was not provided

    except Exception as e:

        dict_['title'].append(meta_data['title'].values[0])

    

    # add the publish time information, add breaks when needed

    try:

        publish_time = meta_data['publish_time'].values[0]

        dict_['publish_time'].append(publish_time)

    # if title was not provided

    except Exception as e:

        dict_['publish_time'].append(meta_data['publish_time'].values[0])

        

    # add the journal information

    dict_['journal'].append(meta_data['journal'].values[0])

    

df_covid = pd.DataFrame(dict_, columns=['paper_id', 'abstract', 'body_text', 'authors', 'title', 'journal', 'abstract_summary','publish_time'])

df_covid.to_csv('df_covid.csv')
import os

import json

import glob as gl

import sys

import numpy as np

import pandas as pd

sys.path.insert(0, "../")



sys.path.insert(0, "../")

root_path = '/kaggle/input/'



## /kaggle/input/covid-19-nlp-tasks/df_covid.csv --> Problème de localisation du fichier dans ma version



# Import the 27 March 2020 version of the COVID-19 Open Research Dataset Challenge (CORD-19) data base Version 27 March 2020=> 27 678 articles.

df_covid = pd.read_csv(f'{root_path}df-covid/df_covid.csv')
import torch

from transformers import *



# Transformers has a unified API

# for 10 transformer architectures and 30 pretrained weights.

#          Model          | Tokenizer          | Pretrained weights shortcut

MODELS = 'bert-base-uncased'



# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`



# Load pretrained model/tokenizer

tokenizer = BertTokenizer.from_pretrained(MODELS)

model = BertModel.from_pretrained(MODELS)

## DO NOT RUN HERE, PROCESS EXTREMELY SLOW

# USE INSTEAD the ArticleParaEmb.csv file to have the signature of 27678 articles.



# # BERT PARAMETERS

# block_size = 512

# Vector_size = 768 



# for a in range(df_covid.shape[0]):

#     print(a,'/',len(df_covid['body_text'])) # Processing of the text

        

#     # Article body text

#     text = df_covid['body_text'].iloc[a]

    

#     # Encode text

#     input_ids_1 = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.



#     # Result table

#     result = torch.tensor(np.empty(list(input_ids_1.size())+[Vector_size]))   



#     # In case oh the number of words in the sentence is superior of block_size of BERT: 

#     if input_ids_1.shape[1] > block_size:

#         nbatch = round(input_ids_1.shape[1]/block_size)

#         if nbatch < input_ids_1.shape[1]/block_size:

#             last_size = input_ids_1.shape[1]-block_size*nbatch

#         else:

#             nbatch = nbatch-1

#             last_size = input_ids_1.shape[1]-block_size*nbatch

#         iter_=0

#         with torch.no_grad():

#             while iter_ != nbatch:

#                 result[:,(iter_*block_size):((iter_+1)*block_size),:] = model(input_ids_1[:,(iter_*block_size):((iter_+1)*block_size)])[0]

#                 iter_ += 1

#             result[:,(iter_*block_size):((iter_*block_size)+last_size),:] = model(input_ids_1[:,(iter_*block_size):((iter_*block_size)+last_size)])[0]

#     else:

#         with torch.no_grad():

#             result = model(input_ids_1)[0]

#     result = result.numpy()[0,:,:]



#     # Compute the mean of all vocabulary BERT representation of the body text

#     result = result.mean(dim=1)



#     # Save all article BERT representation

#     if( a == 0):

#         ArticleParaEmb = result[np.newaxis]

#     else:

#         ArticleParaEmb = np.concatenate((ArticleParaEmb,result[np.newaxis]))
###### Hyperparameters #######

nArticleKeywords = 100

nArticleClusters = 60

nTopArticleCluster = 100

nClusterKeywords = 100
import random

import re

from scipy.spatial.distance import cdist

from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans

from sklearn.feature_extraction.text import CountVectorizer

random.seed(12)



# Article embedding load from the 27 March 2020 version of the COVID-19 Open Research Dataset Challenge (CORD-19) data base Version 27 March 2020=> 27 678 articles embedding corresponding to the df_covid data base.

ArticleParaEmb_np =  pd.read_csv(f'{root_path}articlebertembedding768d/ArticleParaEmb.csv')





###### Data Processing ##########

# Function to supress non-pertinent article

def TitleToDel(x):

    if isinstance(x, str):

        x = x.split(' ')

        k = 0

        flag=False

        while (k < len(x))&(flag==False):

            flag = x[k] in ['Index','Subject','Cumulative']

            k=k+1

        return(flag)

    else:

        return(True)





# Mask to delete non informative article of the data base

sub_df_covid = df_covid.copy()

ind_to_del = sub_df_covid['title'].apply(lambda x: TitleToDel(x))

mask = np.ones(len(sub_df_covid['title']), dtype=bool)

mask[ind_to_del] = False



ArticleParaEmb_np = ArticleParaEmb_np.to_numpy()[mask,1:]

sub_df_covid = sub_df_covid.iloc[mask,:]







###### Top Keywords Article ##########



### Delete punctuation and lowercase the text

df = sub_df_covid.copy()

df['body_text'] = df['body_text'].map(lambda x: re.sub("[,\.!?]", "", x))

df['body_text'] = df['body_text'].map(lambda x: x.lower())



# Word tokenizer

count_vectorizer = CountVectorizer(stop_words='english')

df['nKeywords'] = ""

df['nCount'] = ""



### Remove stopwords and add 10 most used words for each paper

for index, row in df.iterrows():

    row['body_text'] = [row['body_text']]

    count_data = count_vectorizer.fit_transform(row['body_text'])

    words = count_vectorizer.get_feature_names()

    total_counts = np.zeros(len(words))

    for t in count_data:

        total_counts += t.toarray()[0]



    count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[:nArticleKeywords]

    words = [w[0] for w in count_dict]

    counts = [w[1] for w in count_dict]

    df.at[index, 'nKeywords'] = words

    df.at[index, 'nCount'] = counts



sub_df_covid['nKeywords'] = df['nKeywords']

sub_df_covid['nCount'] = df['nCount']

###### Article Clustering ##########

clust = MiniBatchKMeans(n_clusters=nArticleClusters,random_state=0,batch_size=round(ArticleParaEmb_np.shape[0]*0.4)).fit(ArticleParaEmb_np)







###### Top Keywords Cluster  ##########



catch = re.compile(r'/*[.-;\*()\[§\]]')

catch1 = re.compile(r'/*[.]') #-> punctuation to separate each sentence 

catch2 = re.compile(r'\n') #-> separator of section 

catch3 = re.compile(r'/*[(]') #-> open parenthesis/citation

catch4 = re.compile(r'/*[)]') #-> close parenthesis/citation

catch5 = re.compile(r'\[') #-> open parenthesis/citation

catch6 = re.compile(r'\]') #-> close parenthesis/citation

catch7 = re.compile(r'\s\[\s') #-> punctuation to separate each sentence 

catch8 = re.compile(r'\s\]\s') #-> punctuation to separate each sentence 

catch9 = re.compile(r'\s\(\s') #-> punctuation to separate each sentence 

catch10 = re.compile(r'\s\)\s') #-> punctuation to separate each sentence 

match1 = re.compile(r'([0-9]\.[0-9])') #-> punctuation to separate each sentence 

match2 = re.compile(r'Fig.') #-> punctuation to separate each sentence 

match3 = re.compile(r'<br>') #-> punctuation to separate each sentence 

match4 = re.compile(r'##') #-> punctuation to separate each sentence

match5 = re.compile(r',') #-> punctuation to separate each sentence 

match6 = re.compile(r'/*[-;\'%"+=£«^»]') #-> punctuation to separate each sentence

match7 = re.compile(r'\[\d\]') #-> punctuation to separate each sentence 

match8 = re.compile(r'\s\w\s') #-> punctuation to separate each sentence 





stop_words = set(stopwords.words('english'))

ClusterKeyWord = list()

for a in set(clust.labels_):

    if(((a+1) % 10) == 0):

        print((a+1),'/',len(set(clust.labels_)))

    clust_dist = ArticleParaEmb_np[clust.labels_==a,:]

    clust_abs = sub_df_covid.iloc[clust.labels_==a,:]

    all_dist = np.array([cdist(clust.cluster_centers_[a][np.newaxis],clust_dist[i][np.newaxis])[0][0].tolist() for i in range(clust_dist.shape[0])])

    index_sort = np.argsort(all_dist)

    if len(index_sort)>nTopArticleCluster:

        abstract = clust_abs.iloc[index_sort[:nTopArticleCluster],:]

    else:

        abstract = clust_abs.iloc[:,:]

        

    text = abstract['abstract_summary'].copy()



    study = abstract.loc[abstract['abstract_summary'] == "Not provided.",:]

    replace_text = study.copy()

    for k in range(study.shape[0]):

        if isinstance(study['title'].iloc[k], str):

            replace_text['abstract_summary'].iloc[k] = '. '.join([study['title'].iloc[k],catch2.sub(" \n ",study['body_text'].iloc[k]).split(' \n ')[0]])

        else:

            replace_text['abstract_summary'].iloc[k] = catch2.sub(" \n ",study['body_text'].iloc[k]).split(' \n ')[0]

        

    text.loc[abstract['abstract_summary'] == "Not provided."]=replace_text['abstract_summary']

    text = text.map(lambda x: x.lower())

    text = ". ".join(text.iloc[:])

    text = catch.sub("", text)

    text = match3.sub(" ", text)

    text = match5.sub("", text)

    text = match6.sub("", text)

    text = match7.sub("", text)

    text = catch2.sub(" ", text)

    

    text = text.split(" ")

    text = [w for w in text if not w in stop_words]

    text = " ".join(text) 

    text = match8.sub("", text)

    

    count_data = count_vectorizer.fit_transform([text])

    words = count_vectorizer.get_feature_names()

    total_counts = np.zeros(len(words))

    for t in count_data:

        total_counts += t.toarray()[0]



    count_dict = (zip(words, total_counts))

    count_dict = sorted(count_dict, key=lambda x: x[1], reverse=True)[:nClusterKeywords]

    words = [w[0] for w in count_dict]

    ClusterKeyWord.append(words)
########### Preprocessing for plotting results ###########



import colorcet

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE  



colors = [colorcet.glasbey[i] for i in range(0,len(colorcet.glasbey) ,round(len(colorcet.glasbey)/len(set(clust.labels_))))]



match10 = re.compile(r"(?:ed$)")

i=0

ListKey = []

for i in range(len(ClusterKeyWord)):

    tmp = [w for w in ClusterKeyWord[i] if not w in stop_words]

    tmp = [match10.sub("", w) for w in tmp]

    ListKey.append(" ; ".join(tmp))





pca = PCA(n_components=100)

pca_result = pca.fit_transform(ArticleParaEmb_np)



  

X_pca = pca_result[:,:(np.where(np.cumsum(pca.explained_variance_ratio_)>= 0.9)[0][0])]

X_embedded = TSNE(n_components=2,verbose=1, perplexity=100).fit_transform(X_pca)





X_par = sub_df_covid.copy()

X_par['X'] = X_embedded[:,0]

X_par['Y'] = X_embedded[:,1]

X_par['Cluster'] = clust.labels_

X_par['Colors'] = [colors[i] for i in clust.labels_]
########### Plotting results ###########



# BERT PARAMETERS

block_size = 512

Vector_size = 768



from IPython.display import display

from bokeh.io import push_notebook, show, output_notebook

from bokeh.plotting import figure,Figure

from bokeh.layouts import column, row

from bokeh.models import ColumnDataSource,Panel, Tabs,TextInput,CustomJS, Slider

from ipywidgets import interact

from scipy.spatial.distance import cosine

match10 = re.compile(r"/*[0-9-&\/\\#,+()$~%.'\":*?<>{}]")



###################### CLUSTERING BY BERT REPRESENTATION AND PCA&t-SNE ######################

output_notebook()

source1 = ColumnDataSource(data=X_par[["X","Y","Colors","Cluster","title","publish_time","nKeywords"]])

sourceUpdate = ColumnDataSource(data=X_par[["X","Y","Colors","Cluster","title","publish_time","nKeywords"]])

sourceKeywords = ColumnDataSource(data=pd.DataFrame(ListKey,columns=['Keywords']))

# sourceParameters = ColumnDataSource(data=pd.DataFrame([nArticlePlot]))



# Java script code for update filtering text box

ClusterKeywordSearch = CustomJS(args=dict(source1= source1,sourceUpdate=sourceUpdate,sourceKeywords=sourceKeywords), code="""

    const reducer = (accumulator, currentValue) => accumulator + currentValue;

    var data = sourceUpdate.data;

    const data2 = source1.data

    const returnedTarget1 = Object.assign(data, data2)

    var text = cb_obj.value

    if (text == "") {

        const newdata = source1.data

        const returnedTarget1 = Object.assign(data, newdata)

        sourceUpdate.change.emit();

    } else {

        var textlist = text.split(';')

        const newdata = {"index":[],"X":[],"Y":[],"Colors":[],"title":[],"publish_time":[]}

        const keys1 = Object.keys(newdata)

        var cluster = []

        for (var i = 0; i < sourceKeywords.data['Keywords'].length; i++) {

            var keywords = sourceKeywords.data['Keywords'][i].split(" ; ")

            for (var t = 0; t < textlist.length; t++) {

                text = textlist[t]

                text = text.toLowerCase()

                text = text.replace(/[0-9-&\/\\#,+()$~%.'":*?<>{}]/g,'')

                if(t==0){

                    var tab = [keywords.includes(text)]

                }

                if(t>0){

                    tab.push(keywords.includes(text))

                }

            }

            if(tab.reduce(reducer)==(textlist.length)){

                cluster.push(i)

            }

        }

        var indices = []

        for (var i = 0; i < data2["Cluster"].length; i++) {

            if (cluster.includes(data2["Cluster"][i])) 

                indices.push(i)

        }

        if (indices.length > 0) {

            for (var i = 0; i < keys1.length; i++) {

                for (var k = 0; k < indices.length; k++) {

                    newdata[keys1[i]].push(data2[keys1[i]][indices[k]])

                }

            }

            const returnedTarget2 = Object.assign(data, newdata)

        }

        if (indices.length == 0) {

            const newdata = source1.data

            const returnedTarget2 = Object.assign(data, newdata)

        }

        sourceUpdate.change.emit();

    }""")



ArticleKeywordSearch = CustomJS(args=dict(source1= source1,sourceUpdate=sourceUpdate), code="""

    const reducer = (accumulator, currentValue) => accumulator + currentValue;

    var data = sourceUpdate.data;

    const data2 = source1.data

    const returnedTarget1 = Object.assign(data, data2)

    var text = cb_obj.value

    if (text == "") {

        const newdata = source1.data

        const returnedTarget1 = Object.assign(data, newdata)

        sourceUpdate.change.emit();

    } else {

        var textlist = text.split(';')

        const newdata = {"index":[],"X":[],"Y":[],"Colors":[],"title":[],"publish_time":[]}

        const keys1 = Object.keys(newdata)

        var indices = []

        for (var i = 0; i < data2["nKeywords"].length; i++) {

            for (var t = 0; t < textlist.length; t++) {

                text = textlist[t]

                text = text.toLowerCase()

                text = text.replace(/[0-9-&\/\\#,+()$~%.'":*?<>{}]/g,'')

                if(t==0){

                    var tab = [data2["nKeywords"][i].includes(text)]

                }

                if(t>0){

                    tab.push(data2["nKeywords"][i].includes(text))

                }

            }

            if(tab.reduce(reducer)==(textlist.length)){

                indices.push(i)

            }

        }

        if (indices.length > 0) {

            for (var i = 0; i < keys1.length; i++) {

                for (var k = 0; k < indices.length; k++) {

                    newdata[keys1[i]].push(data2[keys1[i]][indices[k]])

                }

            }

            const returnedTarget2 = Object.assign(data, newdata)

        }

        if (indices.length == 0) {

            const newdata = source1.data

            const returnedTarget2 = Object.assign(data, newdata)

        }

        sourceUpdate.change.emit();

    }""")





TOOLTIPS=[

    ("Title", "@title"),

    ("Publishdate", "@publish_time")

]



p = figure(title='t-SNE representation of the article BERT-768d-embedding reduced by PCA',tooltips=TOOLTIPS)



p.circle(x='X',y='Y',size=5,color='Colors', source=sourceUpdate)



text_clusterkeyword = TextInput(value="", title="Cluster Keywords filtering\n\n(TIPS: ';' to separate multiple words, entry with nothing to reset):")

text_clusterkeyword.js_on_change('value', ClusterKeywordSearch)



text_articlekeyword = TextInput(value="", title="Article Keywords filtering \n (TIPS: ';' to separate multiple words, entry with nothing to reset):")

text_articlekeyword.js_on_change('value', ArticleKeywordSearch)



###################### TOPIC RESEARCH SENTENCES BY BERT REPRESENTATION ######################

def TopicSearch(Topics,nTopArticle,nTopSentences,dateOrder):

    if(Topics != ''):

        if(len(Topics.split(' '))>6):

            nTopArticle = int(nTopArticle)

            nTopSentences = int(nTopSentences)

            print('In Process of ',Topics)

            

            # Topic BERT encoding and representation

            input_ids_1 = torch.tensor([tokenizer.encode(Topics, add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.

            with torch.no_grad():

                result = model(input_ids_1)[0]

            result = result.numpy()[0,:,:]

            if(result.shape[1]>1):

                result = result.mean(axis=0)

            

            # Topic filtering and processing for best article keyword matching

            x = Topics.lower()

            x = x.split(" ")

            x = [match10.sub("", i) for i in x]

            x = [w for w in x if not w in stop_words]

            x = [x[i] for i in range(len(x)) if x[i] != '']

            

            #Matching

            list_indices = []

            for i in range(X_par.shape[0]):

                list_indices.append(sum([k in X_par['nKeywords'].iloc[i] for k in x]))

            

            # Take nTopArticle best articles on the n words from the topic matched on each keyword articles

            First = len(list_indices)-1

            sort_indices = np.sort(list_indices)

            index_sort = np.argsort(list_indices)

            while sort_indices[First] == max(list_indices):

                First=First-1

            First = First+1

            if(len(range(First,X_par.shape[0])) > nTopArticle):

                newindices = index_sort[First:]

            else:

                newindices = index_sort[nTopArticle:]  

            

            # Rordering on most current Dates

            if(dateOrder):

                date_ = []

                for i in newindices:

                    tmp = sub_df_covid['publish_time'].iloc[i].split(' ')

                    if(len(tmp)>1):

                        date_.append(int(tmp[0]))

                    else:

                        date_.append(int(sub_df_covid['publish_time'].iloc[i].split('-')[0]))

                index_sort = np.argsort(date_)

                newindices = np.flip(newindices[index_sort[(len(index_sort)-nTopArticle):]])

                

            # Article BERT sentence representation research loop

            for a in newindices:

                ## Body text preocessing 

                # Special character spliting and elimination of all "." not used as sentence separator

                text = match2.sub("Fig", sub_df_covid['body_text'].iloc[a])

                text = catch1.sub(" . ", text)

                text = catch2.sub(" \n ", text)

                text = catch3.sub(" ( ", text)

                text = catch4.sub(" ) ", text)

                text = catch5.sub(" [ ", text)

                text = catch6.sub(" ] ", text)

                text = text.split(" . ")

                

                # Loop to collapse all part separated part of a parenthesis

                h=0

                n=len(text)

                i=0

                while i < n:

                    if(h==1):

                        i = i-1

                    tmp = text[i].split(' ')

                    h=0

                    for k in range(len(tmp)):

                        if tmp[k] in ['(','[']:

                            h=1

                        if ((tmp[k] in [')',']'])&(h==1)):

                            h=0

                        if((k==(len(tmp)-1))&(h==1)):

                            if(i != (len(text)-1)):

                                text[i] = text[i]+text[i+1]

                                del text[i+1]

                    i=i+1

                    n= len(text)

                text = " . ".join(text)  

                text = catch7.sub("(", text)

                text = catch8.sub(")", text)

                text = catch9.sub("[", text)

                text = catch10.sub("]", text)

                text = text.split(" . ")

                text = [text[i] for i in range(len(text)) if text[i] != '']

                

                # Encode vocabulary body text

                input_ids_1 = torch.tensor([tokenizer.encode(" . ".join(text), add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.



                # vocabulary BERT tokenized

                old_text_token = tokenizer.tokenize(" . ".join(text),add_special_tokens=True)

                old_text_token.insert(0,"<CLS>")

                old_text_token.insert(len(old_text_token),"<CLS>")

                

                # loop to collapse word separated in the BERT tokenized representation

                text_token = [match4.sub(' ',old_text_token[k]) for k in range(len(old_text_token)) ]

                new_text_token = []

                k=0

                h=0

                first=0

                list_index = np.empty(len(text_token))

                while k < len(text_token):

                    if(len(text_token[k].split(' '))>1):

                        if first == 0:

                            list_index[h-1] = 1

                            first = 1

                        new_text_token.append(''.join([new_text_token[k-1],text_token[k].split(' ')[1]]))

                        del text_token[k]

                        del new_text_token[k-1]

                        list_index[h] = 2

                    else:

                        first=0

                        new_text_token.append(text_token[k])

                        list_index[h] = 0

                        k=k+1

                    h=h+1

                

                # Sentence reconstruction with the body text tokenized reconstructed

                concatnex = []

                old_i = 0

                for i in range(len(new_text_token)):

                    if ((new_text_token[i]=='.')|(i==(len(new_text_token)-1))):

                        if old_i == 0:

                            concatnex = [' '.join(new_text_token[old_i:i])]

                        else:

                            concatnex.append(' '.join(new_text_token[old_i:i]))

                        old_i=i+1



                

                result_bis = torch.tensor(np.empty(list(input_ids_1.size())+[Vector_size]))   

                

                # In case oh the number of words in the sentence is superior of block_size of BERT: 

                if input_ids_1.shape[1] > block_size:

                    nbatch = round(input_ids_1.shape[1]/block_size)

                    if nbatch < input_ids_1.shape[1]/block_size:

                        last_size = input_ids_1.shape[1]-block_size*nbatch

                    else:

                        nbatch = nbatch-1

                        last_size = input_ids_1.shape[1]-block_size*nbatch

                    iter_=0

                    

                # Vocabulary BERT representation

                    with torch.no_grad():

                        while iter_ != nbatch:

                            result_bis[:,(iter_*block_size):((iter_+1)*block_size),:] = model(input_ids_1[:,(iter_*block_size):((iter_+1)*block_size)])[0]

                            iter_ += 1

                        result_bis[:,(iter_*block_size):((iter_*block_size)+last_size),:] = model(input_ids_1[:,(iter_*block_size):((iter_*block_size)+last_size)])[0]

                else:

                    with torch.no_grad():

                        result_bis = model(input_ids_1)[0]

                

                result_bis = result_bis.numpy()[0,:,:]

                

                # Sentence BERT representation

                old_i = 0

                for i in range(len(old_text_token)):

                    if ((old_text_token[i]=='.')|(i==(len(old_text_token)-1))):

                        if old_i == 0:

                            result_bis_bis = result_bis[old_i:i,:].mean(axis=0)[np.newaxis]

                        else:

                            result_bis_bis = np.concatenate((result_bis_bis,result_bis[old_i:i,:].mean(axis=0)[np.newaxis]))

                        old_i=i

                # Cosinus distance between Topic and Sentences

                all_dist = np.array([cosine(result[np.newaxis],result_bis_bis[i,:]) for i in range(result_bis_bis.shape[0])])

                index_sort = np.argsort(all_dist)

                concatnex = np.array(concatnex)[index_sort[:nTopSentences]]

                print('\n\n Title: ',sub_df_covid['title'].iloc[a],'\n Sentences: ','. '.join(concatnex.tolist()))

            display(sub_df_covid[['title',"publish_time", 'paper_id']].iloc[newindices,:])





# Set up layouts and add to document

inputs = row(column(text_clusterkeyword,text_articlekeyword),p)

show(inputs, notebook_handle=True)
interact(TopicSearch, Topics='What do we know about COVID-19 risk factors ?',nTopArticle='10',nTopSentences='5',dateOrder=True)
interact(TopicSearch, Topics='What do we know about COVID-19 risk factors ?',nTopArticle='10',nTopSentences='5',dateOrder=False)