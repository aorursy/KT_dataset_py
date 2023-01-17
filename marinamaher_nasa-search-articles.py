!python -m spacy download en_core_web_lg

!pip install simpleneighbors

!pip install chart-studio 

import argparse

import requests

import cv2

import matplotlib.pyplot as plt

import os

from datetime import datetime

import numpy as np

import matplotlib.pyplot as plt

import matplotlib.animation as ani

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



# loop the URLs

import time

import matplotlib.pyplot as plt

import numpy as np

import os

import PIL

import tensorflow as tf

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib

import matplotlib.pyplot as plt

from ipywidgets import interact, widgets # this is what makes the dataframe interactive



from skimage import io

import matplotlib.pyplot as plt

from IPython.core.display import HTML



import matplotlib.animation as animation

from tensorflow import keras

from tensorflow.keras import layers

from tensorflow.keras.models import Sequential

start_time = time.time() # To know code run time

import requests

from bs4 import BeautifulSoup

import pprint

!pip install sentence-transformers

import requests

from bs4 import BeautifulSoup

import pickle as pkl

import pandas as pd

import numpy as np 

from sentence_transformers import SentenceTransformer

from scipy.spatial.distance import jensenshannon

from IPython.display import HTML, display







import nltk

import os

from nltk.stem.lancaster import LancasterStemmer

import numpy as np

import tensorflow as tf

import random

import json

import pickle

import numpy as np

import pandas as pd

import json

from pandas.io.json import json_normalize

import requests as req

import re

from PIL import Image

import requests

from io import BytesIO

import urllib.request

!pip install transformers

!wget -O scibert_uncased.tar https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar

!tar -xvf scibert_uncased.tar

import torch

from transformers import BertTokenizer, BertModel
NASA =  pd.read_pickle("../input/nasa-data-preprocessing/NASA.pkl")

NASA.head(1)
img_scores=np.zeros(NASA.shape[0])



def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)

    err /= float(imageA.shape[0] * imageA.shape[1])

    return err



def image_search(input_image):

    for dirname, _, filenames in os.walk('../input/nasa-data-preprocessing'):

        for filename in filenames:

            if("jpg" in filename ):

                pa=os.path.join(dirname, filename)

            

                original = cv2.imread(input_image)

                contrast = cv2.imread(pa)



                mini0=min(original.shape[0],contrast.shape[0])

                mini1=min(original.shape[1],contrast.shape[1])



                original.resize((mini0, mini1,3))

                contrast.resize((mini0, mini1,3))



                imageA = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

                imageB = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)

                m = mse(imageA, imageB)

                

                num=[int(s) for s in filename.split() if s.isdigit()]

                n0=num[0]

                img_scores[n0]=m

    maxi=np.max(img_scores)+10

    l=(maxi-img_scores)/maxi

    NASA["img_scores"]=l
title = NASA['title'].tolist()

description = NASA['description'].tolist()
model_bert = SentenceTransformer('bert-base-nli-max-tokens')

bert_text_vec =model_bert.encode(description)

bert_vectors = np.array(bert_text_vec.tolist())



sci_bert_version = 'scibert_scivocab_uncased'

sci_tokenizer = BertTokenizer.from_pretrained(sci_bert_version, do_lower_case=True)

encoded_title = sci_tokenizer.encode(title )
from scipy import spatial

def score(model_vectors,model_encode,size):

    score = []

    for i in range(size):

        result = 1 - spatial.distance.cosine(model_vectors[i],model_encode)

        score.append(result)

    return score
def average (model1_score,model2_score):

    l=[sum(n) for n in zip(*[model1_score,model2_score])]

    final_score = [x * 0.5 for x in l]

    return final_score 



def average3 (model1_score,model2_score,model3_score):

    l=[sum(n) for n in zip(*[model1_score,model2_score,model3_score])]

    final_score = [x * 0.3 for x in l]

    return final_score 
date="2020-04-04"

date_object_check = datetime.strptime(date, '%Y-%m-%d').date()

for i in range(NASA.shape[0]):

    date_str = NASA["date"][i]

    date_object = datetime.strptime(date_str, '%Y-%m-%d').date()

    if( date_object > date_object_check ):

        print(NASA["date"][i])
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



def Text(list):  

    str1 = ""    

    for element in list:  

        str1 += element 

    return str1

def getwordcloud(text,img): 

    wordcloud = WordCloud(max_font_size=256, max_words=500, background_color="black",stopwords = set(STOPWORDS),random_state=42, width=500, height=500).generate(text)

    plt.figure(figsize=(10,5))

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis("off")

    plt.show()


def make_clickable(link):

   

    return f'<a target="_blank" href="{link}">{link}</a>'



NASA['url'] = NASA['url'].apply(make_clickable)



NASA.head()
l=list()



for i in range(NASA.shape[0]):

    l.append(str(datetime.strptime(NASA['date'][i], '%Y-%m-%d').date().year))



NASA['year']=l
fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)



def nice_axes(ax):

    ax.set_facecolor('.8')

    ax.tick_params(labelsize=8, length=0)

    ax.grid(True, axis='x', color='white')

    ax.set_axisbelow(True)

    ax.set_ylabel('Year')

    ax.set_xlabel('Number of papers in precentage')

    ax.set_title('precentage of articles in each year',color='r')

    [spine.set_visible(False) for spine in ax.spines.values()]

    

nice_axes(ax)

def draw_fun (df):

    NASA_years = df[['year','tag']].sort_values(by=['year'], ascending=True)

    NASA_years['combine']=NASA_years['year']

    y=NASA_years['combine'].value_counts()

    y=y/y.sum()*100

    y=y.to_frame()

    index=list(y.index)

    counts=list(y['combine'])

    draw=pd.DataFrame()

    draw['index']=index

    draw['counts']=counts

    nums=list()

    for i in range(draw.shape[0]):

        c=int(re.search(r'\d+', draw.iloc[i,0]).group())

        nums.append(c)

    draw["years"]=nums

    draw = draw.sort_values(by=['years'], ascending=True)

    return draw
def path_to_image_html(path):

    return '<img src="'+ path + '" width="160" >'



date="NULL"

from ipywidgets import interact, widgets # this is what makes the dataframe interactive

default_question = 'temperature'

img="NULL"



pd.set_option('display.float_format', lambda x: '%.3f' % x)

pd.set_option('max_colwidth', 180)



results=[]

total_docs=NASA.shape[0]

@interact

def search_articles( Ur_query=default_question,num_results=[2, 4, 5],Ur_image=img,begin_date=date):



    sci_encode =sci_tokenizer.encode([Ur_query])

    bert_encode = model_bert.encode([Ur_query])

    

    if(Ur_image != "NULL"):

        image_search(Ur_image)

        imgs=NASA["img_scores"]

        bert_score=score(bert_vectors,bert_encode,total_docs)

        sci_score=score(encoded_title,sci_encode,total_docs)

        f_score=average3(bert_score,sci_score,imgs)

        NASA["score"] = f_score

        select_cols = ['title', 'description', 'tag','score','url','date','year','image']

        results = NASA[select_cols].sort_values(by=['score'], ascending=False).head(num_results)

        results = results.dropna(subset=['title'])

        d=draw_fun(results)

        if(d.shape[0]!=0):

            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)

            colors = plt.cm.Dark2(range(6))

            ax.barh(y=d['index'].values, width=d['counts'].values, color=colors);

            ax.set_ylabel('Year')

            ax.set_xlabel('Number of papers in precentage')

            ax.set_title('precentage of articles in each year',color='r')

        return (HTML(results.to_html(escape=False, formatters=dict(image=path_to_image_html))))



     

    elif (True):

        bert_score=score(bert_vectors,bert_encode,total_docs)

        sci_score=score(encoded_title,sci_encode,total_docs)

        f_score=average(bert_score,sci_score)

        NASA["score"] = f_score

        select_cols = ['title', 'description', 'tag','score','url','date','year']

        results = NASA[select_cols].sort_values(by=['score'], ascending=False).head(num_results)

        results = results.dropna(subset=['title'])

       

        

        if(begin_date != "NULL"):

            num_results=5

            l=list()

            date_object_check = datetime.strptime(begin_date, '%Y-%m-%d').date()

            for i in range(results.shape[0]):

                date_str = results.iloc[i,5]

                date_object = datetime.strptime(date_str, '%Y-%m-%d').date()

                if( date_object > date_object_check ):

                    l.append(results.iloc[i,:])

            results=pd.DataFrame(l)

             

        else:

            j=0

            for i in range(num_results):

                Description=Text(results["description"][i:i+1].tolist())

                common_abstract_word=getwordcloud(Description , str(results["tag"][i:i+1])+"_"+str(j))

                j=j+1

            

            

    if (len(results.index) == 0):

        print('NO RESULTS')

        return None

    else:

        

        top_row = results.iloc[0]



        print('TOP RESULT OUT OF ' + str(total_docs) + ' DOCS FOR QUESTION:\n' + Ur_query + '\n')

        print('TITLE: ' + str(top_row['title']) + '\n')



        d=draw_fun(results)

        if(d.shape[0]!=0):

            fig, ax = plt.subplots(figsize=(4, 2.5), dpi=144)

            colors = plt.cm.Dark2(range(6))

            ax.barh(y=d['index'].values, width=d['counts'].values, color=colors);

            ax.set_ylabel('Year')

            ax.set_xlabel('Number of papers in precentage')

            ax.set_title('precentage of articles in each year',color='r')

        

        return HTML(results[select_cols].to_html(escape=False))
Description=Text(NASA["description"][0:1].tolist())

common_abstract_word=getwordcloud(Description,"")
Tags=Text(NASA["tag"][:5].tolist())

common_abstract_word=getwordcloud(Tags,"")
