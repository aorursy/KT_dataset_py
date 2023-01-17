# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import json



path = "/kaggle/input/CORD-19-research-challenge/"



def Load_articles(path):

    titles=[]

    abstracts=[]

    texts=[]

    #['biorxiv_medrxiv','comm_use_subset','custom_license','noncomm_use_subset'] u can add all of them

    for directory in ['biorxiv_medrxiv'] : 

        dir_list = os.listdir(path+directory+'/'+directory+'/')

        for dr in dir_list:

            j= json.load(open(path+directory+'/'+directory+'/'+dr,'rb'))

            titles.append(j['metadata']['title'])

            abstrs=""

            for abstract in j['abstract'] :

                abstrs+=str(abstract['text']+'\n')

            abstracts.append(abstrs)

            txts=""

            for text in j['body_text'] :

                txts+=str(text['text']+'\n')

            texts.append(txts)

    print('get_all_articles_texts accomlished')

    return {'title':titles,'abstract':abstracts,'body_text':texts}

from nltk.tokenize import RegexpTokenizer

from nltk.corpus import stopwords



def DataPreprecessing(lst_of_lst):

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_-]+')

    stop_words = set(stopwords.words('english'))

    lst_result=[]

    for lst in lst_of_lst:

        txt=' '.join(lst)

        lst_no_punc=tokenizer.tokenize(txt)

        lst_no_punc_no_SW = [word for word in lst_no_punc if word not in stop_words]

        lst_result.append(lst_no_punc_no_SW )

    print('No_StopWords_No_punctuation accomplshed')

    return (lst_result)



def QstPreprocessing(txt):

    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9_-]+')

    stop_words = set(stopwords.words('english'))

    items = [word for word in [ item.lower() for item in tokenizer.tokenize(txt)] if word not in stop_words]

    return (" ".join(items))



#print(QstPreprocessing("What is known about transmission, incubation, and environmental stability?"))
import nltk

from nltk.corpus import stopwords

from gensim.models import Word2Vec

import os

import gensim



def creat_model(path,Model_Name1,Model_Name2):

#     if not(os.path.isfile(path+Model_Name)) :

    df=pd.DataFrame(Load_articles(path)) #DataDrame presenting

    corpus=df["title"].values.tolist()+df["abstract"].values.tolist()+df["body_text"].values.tolist()

    del df

    tok_corp=[nltk.word_tokenize(str(sent).lower()) for sent in corpus] #tokenize the full text

    del corpus

    print('step tokenize accomplished')

    tok_corp=DataPreprecessing(tok_corp) #Data PreProcessing

    print('step remove Stop Words with Punctuation accomplished')

    model1=gensim.models.Word2Vec(tok_corp,min_count=6,size=32, iter=10, sg=0, workers=14)

    model1.save("../working/"+Model_Name1)#save the first model in "../working/" path

    del model1

    model2=gensim.models.Word2Vec(tok_corp,min_count=6,size=32, iter=10, sg=1, workers=14)

    model2.save("../working/"+Model_Name2) #save the second model in "../working/" path

    del model2

    del tok_corp

    

creat_model("/kaggle/input/CORD-19-research-challenge/","model_CBOW","model_SG") # Models Creation



sg_model = gensim.models.Word2Vec.load("../working/model_SG") # load Skip-Gram model from "../working/model_SG" path after saved there

cbow_model = gensim.models.Word2Vec.load("../working/model_CBOW") # load CBOW model from "../working/model_CBOW" path 
# load model

def most_similar(model,word):

    similarities = model.wv.most_similar(word,topn=10)

    key=[similar_word for (similar_word , score) in similarities]

    value=[score for (similar_word , score) in similarities]

    return pd.DataFrame({'key':key,'value':value}, columns=["key", "value"])

most_similar(sg_model,'covid-19') 

"""exmample of applying the similar word function for the 'covid-19' word on the skip-gram model """
most_similar(cbow_model,'covid-19') 

"""exmample of applying the similar word function for the 'covid-19' word on the skip-gram model """
def Qst_Enhancement(model,qst,threshold):

    qst=QstPreprocessing(qst)

    items=[]

    for word in qst.split():

        items.append(word)

        try:

            similarities = model.wv.most_similar(word,topn=10)

            for similar_word , score in similarities:

                if score > threshold:

                    items.append(similar_word)

        except : 

            pass

    return " ".join(items)

       

#Qst_Enhancement(sg_model,"What is known about transmission, incubation, and environmental stability?",0.80)

Qst_Enhancement(cbow_model ,"What is known about transmission, incubation, and environmental stability?",0.80)
import re

path="/kaggle/input/CORD-19-research-challenge/"

def re_search(model,qst):

    qst=Qst_Enhancement(model,qst,0.8)

    lst=qst.split()

    df=pd.DataFrame(Load_articles(path))

    columns = list(df)

    articles=[]

    for i in range(len(df)):

        article = (df['title'][i])+"\n"+(df['abstract'][i])

        rank=2

        if(re.search("^.*"+"*.*|".join(lst)+"*$",article)):

            for item in lst:

                if len(re.findall(item,article)) > 0:

                    rank+= 1

                    articles.append([rank,article])

    articles.sort(reverse=True)

    for i,j in articles :

        print('score ======'+str(i))

        print(j)    
re_search(sg_model,'What is known about transmission, incubation, and environmental stability?')

re_search(cbow_model,'What is known about transmission, incubation, and environmental stability?')