import json

import nltk

import matplotlib.pyplot as plt

import pandas as pd

df=pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")

df.journal.isnull().value_counts()

df.publish_time.isnull().value_counts()
# 20 most popular journals

df.journal.value_counts()[0:20].plot(kind='bar')

plt.grid()

plt.show()
biorxiv=pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

comm_use=pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

noncomm_use=pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

pmc=pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")

import nltk

from nltk.corpus import stopwords



stop_words=stopwords.words("english")



#some preprocessing

def preprocess_text(text):

    #lower

    text=str(text).lower()

    #tokenize

    token = nltk.RegexpTokenizer(r'\w+')

    tk = token.tokenize(text)

    #remove numbers

    no_num = [word for word in tk if not word.isnumeric()]

    #remove stopwords

    no_stop = [word for word in no_num if word not in stop_words]

    #lemmatize

    lemmatizer=nltk.stem.WordNetLemmatizer()

    lem = [lemmatizer.lemmatize(word) for word in no_stop]

    return lem



#example

text=biorxiv["abstract"][4]



preprocess_text(text)[:4]
def search_by_word(csv,word_list):

    paper_list=[]

    

    lemmatizer=nltk.stem.WordNetLemmatizer()

    words = [lemmatizer.lemmatize(word) for word in word_list]

    

    for index, paper in csv.iterrows():

        if all([word in preprocess_text(paper.abstract) for word in words]):

            paper_list.append(paper.paper_id)

            

    return paper_list



search = search_by_word(biorxiv,["proton"])

print(len(search))
br_surf = search_by_word(biorxiv,["decontamination"])

#noncomm_use_surf=search_by_word(comm_use,["adhesion"])

#comm_use_surf=search_by_word(noncomm_use,["adhesion"])

pmc_surf=search_by_word(pmc,["decontamination"])

print("number of articles for keyword surface in biorxiv are:",len(br_surf))

#print("number of articles for keyword surface in non_comm_use are:",len(noncomm_use_surf))

#print("number of articles for keyword surface in non_comm_use are:",len(comm_use_surf))

print("number of articles for keyword surface in non_comm_use are:",len(pmc_surf))

#keep only papers of interest



def papers_of_interest(csv,word_list):

    paper_list=search_by_word(csv,word_list)

    poi = pd.DataFrame.copy(csv)

    for index,paper in csv.iterrows():

        if paper.paper_id not in paper_list:

            poi=poi.drop(index)

    return poi

poi = papers_of_interest(biorxiv,["proton"])

print(poi)
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer

import matplotlib.pyplot as plt



def word_bar_graph_function(df,column,title):

    # adapted from https://www.kaggle.com/benhamner/most-common-forum-topic-words

    topic_words = [ z.lower() for y in

                       [ preprocess_text(x) for x in df[column] if isinstance(x, str)]

                       for z in y]

    word_count_dict = dict(Counter(topic_words))

    #filtering additional stopwords. 

    # TODO: Find better way than hardcode

    add_stopwords = ["conclusion","preprint","http","doi","biorxiv","medrxiv"]

    stop_words = list(set(stopwords.words("english"))|set(add_stopwords))

    popular_words = sorted(word_count_dict, key = word_count_dict.get, reverse = True)

    popular_words_nonstop = [w for w in popular_words if w not in stop_words]

    plt.barh(range(50), [word_count_dict[w] for w in reversed(popular_words_nonstop[0:50])])

    plt.yticks([x + 0.5 for x in range(50)], reversed(popular_words_nonstop[0:50]))

    plt.title(title)

    plt.show()





            



plt.figure(figsize=(10,10))

word_bar_graph_function(poi,"abstract", "Most common words in abstracts with environment & transmission")



from gensim.models import Word2Vec
def gen_train(): #ttype - texttype : choose abstract or text (for the full text)

    journals = [biorxiv]#,comm_use,noncomm_use,pmc]

    train_set = []

    for element in journals:

        for index,paper in element.iterrows():

            doc = preprocess_text(paper.text)

            train_set.append(doc) 

    return train_set



# generate train set

train = gen_train()

model = Word2Vec(train,min_count=1,window=10,size=100)
model.most_similar("charge")

import numpy as np

import gensim

import time



def similar_count_dic(model,word):

    similars = model.most_similar(word)

    sim_new = {}

    sim_new[word] = 0

    for i in range(len(similars)):

        sim_new[ similars[i][0] ] = 0

        

    return sim_new



def get_tk2id(paper_text):

    

    text_tk = preprocess_text(paper_text)

    dictionary = gensim.corpora.Dictionary([text_tk])#wo_stop)

    corpus = dictionary.doc2bow(text_tk)#title) for title in wo_stop]

    try:

        dictionary[0]

    except KeyError:

        pass

    tk2id=dictionary.token2id

    

    return tk2id,corpus



#note that only the paper_id is output without the information where it can be found (e.g. biorxiv)



def search_by_similarity(model,word):

    sim_count = similar_count_dic(model,word)

    results={}

    journals = [biorxiv,noncomm_use,comm_use,pmc]

    for element in journals:

        for index,paper in element.iterrows():

            tk2id , corpus = get_tk2id(paper.text)

            counter = 0

            for word in sim_count.keys():

                try: 

                    if corpus[tk2id[word]][1]>0:

                        counter += corpus[tk2id[word]][1]

                except KeyError:

                    pass

            if counter > 0:

                results[paper.paper_id] = counter

    return results

        

t=time.time()

results = search_by_similarity(model,"charge")

t1=time.time()

print(t1-t)
def filter_results(results):

    del_list = [ key for key in results if results[key]<140 ]

    for key in del_list:

        del results[key]

        

    return results



def id_from_meta(paper_id):

    journals = [biorxiv,noncomm_use,comm_use,pmc]

    subset_names = ["biorxiv","noncomm_use","comm_use","pmc"]

    res = []

    for element in journals:

        for index, paper in element.iterrows():

            if paper.paper_id == paper_id:

                return (subset_names[journals.index(element)],index)

                break

    print("paper_id not found")

        

        

            

r=filter_results(results)

id_from_meta(list(r.keys())[0])
with open("../input/CORD-19-research-challenge/biorxiv_medrxiv/biorxiv_medrxiv/"+"e7c3ec3dcb3469ee4d608029e7aa3068a4c90c54"+".json") as f:

    data=json.load(f)

data["metadata"]
#compare with most similar words

#model.most_similar("charge")
def highlight(text):

    return " \033[1;41m " + text + " \033[m "  



def mark_passages(text,pass_nums):

    

    text = nltk.sent_tokenize(text)

    for i in range(len(text)):

        if i in pass_nums:

            text[i] = highlight( text[i] )

        print(text[i])



def preprocess_doc2vec(text):

    text = nltk.sent_tokenize(text)

    token = nltk.RegexpTokenizer(r'\w+')

    out = [token.tokenize(line) for line in text]

    return out



train_ex = preprocess_doc2vec(biorxiv["text"][631])

train_ex[:2];
import nltk

from nltk.corpus import stopwords

from gensim.models.doc2vec import Doc2Vec,TaggedDocument



#some preprocessing



#train_set = gen_train_doc()



def tagged_arts(train):

    tagged_arts = []

    for j in range(len(train)):

        tagged_doc = [TaggedDocument(

                     words=[word for word in document],

                     tags=[i]

                 ) for i, document in enumerate(train)]

        tagged_arts.append(tagged_doc)

    return tagged_arts



tagged_arts = tagged_arts(train_ex)





model = Doc2Vec(tagged_arts[0], vector_size=50, window=10, min_count=1, workers=4,train_epochs=50)
train_ex[0];

inferred_vector = model.infer_vector(nltk.sent_tokenize("ion misses force to push into gate"))

sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))

sims

train_ex[14];

#biorxiv["abstract"][0]
mark_passages(biorxiv["text"][631],[10])