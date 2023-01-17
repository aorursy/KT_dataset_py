!pip install p_tqdm

!pip install dateparser

!pip install tqdm

!pip install pandarallel

!pip install nltk

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

json_files = [] #We store all the publications based json files into here.

for dirname, _, filenames in os.walk('/kaggle/input/CORD-19-research-challenge'):

    for filename in filenames:

        filepath = os.path.join(dirname, filename)

        extension = os.path.splitext(filepath)[1]

        if os.path.splitext(filepath)[1] == '.json':

            json_files.append(filepath)



# Any results you write to the current directory are saved as output.
import os

import json



from p_tqdm import p_map



from dateparser.search import search_dates

from pandarallel import pandarallel

pandarallel.initialize()



#1. Process CORD19 Metadata

cord19_metadata = pd.read_csv("../input/CORD-19-research-challenge/metadata.csv")



def extractDateTime(text):

    if text == "nan":

        text = str(2030)               #For publication with missing date, return 2030

    thedate = search_dates(text)

    return thedate[0][1]               #Return datetime



cord19_metadata['publish_time_str'] = cord19_metadata['publish_time'].astype(str)

cord19_metadata['publish_datetime'] = cord19_metadata['publish_time_str'].parallel_map(extractDateTime)



#2. Process every json file into a new list of dataframe

publications_df = pd.DataFrame()



# to process all files, uncomment the next line and comment the line below

# selected_json_files = json_files

selected_json_files = json_files[0:5000]



selected_columns = ['paper_id', 'metadata.title', 'body_text', 'abstract']



def newDF(file):

    with open(file) as json_file:

        json_data = json.load(json_file)

        json_data_df = pd.io.json.json_normalize(json_data)

        json_data_df_selected = json_data_df[selected_columns]

        return json_data_df_selected



list_df = p_map(newDF, selected_json_files)             #Append every dataframe into a list

publications_df = pd.concat(list_df)                    #Merge every dataframe in one big dataframe



#3. Join publications_df + cord19_metdata based on sha (paper_id)

publications_published_time_df = publications_df.merge(cord19_metadata[['sha','publish_datetime']], how='inner', left_on="paper_id", right_on='sha')



#Extract abstract and body text.

publications_published_time_df['abstract_text'] = publications_published_time_df['abstract'].parallel_apply(lambda x: x[0]['text'] if x else "")

publications_published_time_df['all_body_text'] = publications_published_time_df['body_text'].parallel_apply(lambda x: " ".join([(t['text']) for t in x]))
import pickle

#publications_published_time_df.to_csv("all_publications_time_text.csv")

pickle.dump( publications_published_time_df, open( "publications.p", "wb" ) )
publications_published_time_df.head()
# timeunit = 1 #Represent one year, for now, forget about this

publications_published_time_df['year'] = publications_published_time_df['publish_datetime'].parallel_apply(lambda x: x.year)



#Text preprocessing

#In the paper, we used nltk, but in Kaggle, we used spacy.

from spacy.tokenizer import Tokenizer

from spacy.lang.en import English

nlp = English()

tokenizer = nlp.Defaults.create_tokenizer(nlp)



def textToListProcessing(text):

    new_text = text.lower()

    tokens = tokenizer(new_text)

    tokens_list = [t.text for t in tokens]

    return tokens_list





publications_published_time_df['body_list_of_terms'] = publications_published_time_df['all_body_text'].parallel_apply(textToListProcessing)

publications_published_time_df.head()





#Let's compare two years



list_of_texts_to_2010 = publications_published_time_df[publications_published_time_df['year'] < 2010 ]

list_of_texts_to_2020 = publications_published_time_df[publications_published_time_df['year'] < 2020 ]



list_of_texts_to_2010 = list_of_texts_to_2010["body_list_of_terms"].tolist()

list_of_texts_to_2020 = list_of_texts_to_2020["body_list_of_terms"].tolist()



#Word2Vec

from gensim.test.utils import common_texts, get_tmpfile

import multiprocessing

from gensim.models import Word2Vec

from time import time  # To time our operations



w2v_model = Word2Vec(min_count=10,

             window=2,

             size=300,

             sample=6e-5, 

             alpha=0.03, 

             min_alpha=0.0007, 

             negative=10,

             workers=multiprocessing.cpu_count())



#Update vocabulary

w2v_model.build_vocab(list_of_texts_to_2010)

w2v_model.build_vocab(list_of_texts_to_2020, update=True)



w2v_model.train(list_of_texts_to_2010, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

w2v_model.save("2010.w2v")

w2v_model.train(list_of_texts_to_2020, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)

w2v_model.save("2020.w2v")

import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from collections import defaultdict

stop_words = list(set(stopwords.words('english')))

stop_words.extend(["et", "al", "de", "fig", "en", "use"])



#%% Get embedding vector

from scipy import spatial



def semanticDivergence(a, b):

    cos_lib = 1 - spatial.distance.cosine(a, b)

    return cos_lib



model_2010 = Word2Vec.load("2010.w2v")

model_2020 = Word2Vec.load("2020.w2v")



list_frequent_terms = model_2020.wv.index2entity

list_frequent_terms = [w for w in list_frequent_terms if not w in stop_words]

top1000terms = list_frequent_terms[:1000]



top1000terms_vector_2010 = [(w,model_2010.wv[w]) for w in top1000terms]

top1000terms_vector_2020 = [(w,model_2020.wv[w]) for w in top1000terms]



#Get semantic divergence

word_DV = []

for wv1, wv2 in zip(top1000terms_vector_2010, top1000terms_vector_2020):

    dv = semanticDivergence(wv1[1], wv2[1])

    word_DV.append((wv1[0], dv))



#Sort based on highest divergence to lowest

sorted_word_DV = sorted(word_DV, key=lambda tup: tup[1], reverse=True)

sorted_word_DV[:10]



import matplotlib

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

from sklearn.manifold import TSNE

import math



def visualizeclusterClosestWords_tsne(word, models, years):

    Colors = []

    Labels = []

    Xs = []

    Ys = []

    

    list_color = ['g','r']

    for model, year, cr in zip(models, years, list_color):

        vector_dim = model.vector_size

        arr = np.empty((0,vector_dim), dtype='f')

        theword = word + "\n(" + str(year) + ")"

        word_labels = [theword]

    

        # get close words

        close_words = model.wv.similar_by_word(word, topn=3)

        

        # add the vector for each of the closest words to the array

        arr = np.append(arr, np.array([model[word]]), axis=0)

        for wrd_score in close_words:

            wrd_vector = model[wrd_score[0]]

            word_labels.append(wrd_score[0])

            arr = np.append(arr, np.array([wrd_vector]), axis=0)

            

        # find tsne coords for 2 dimensions

        tsne = TSNE(n_components=2, random_state=0)

        np.set_printoptions(suppress=True)

        Y = tsne.fit_transform(arr)

    

        x_coords = Y[:, 0]

        y_coords = Y[:, 1]

        

        colors = [ cr for i in range(len(x_coords))]

        # colors[0] = 'r'

        

        #Append to list

        Labels.append(word_labels)

        Xs.append(x_coords)

        Ys.append(y_coords)

        Colors.append(colors)

    

    title = 'The semantic divergence for word "' + word + '"'

    plt.title(title)

    for xs, ys, labels, clrs in zip(Xs, Ys, Labels, Colors): 

        plt.scatter(xs, ys, color=clrs, s=300)

        for label, x, y in zip(labels, xs, ys):

            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points') 

        plt.show()

    

visualizeclusterClosestWords_tsne(sorted_word_DV[1][0], [model_2010, model_2020], [2010, 2020] )



#%% Cluster top m where m = 100   

from sklearn.cluster import KMeans

import numpy as np



terms = [(w,model_2020.wv[w]) for w in top1000terms[:100]]

terms_w =  [w for w,v in terms]

terms_wv = [v for w,v in terms]



kmeans = KMeans(n_clusters=4)

kmeans.fit(terms_wv)

y_kmeans = kmeans.predict(terms_wv)



cluster_to_list_terms_dict = defaultdict(list)

for k, term in zip(y_kmeans,terms_w):

    cluster_to_list_terms_dict[k].append(term)



for k, v in cluster_to_list_terms_dict.items():

    print("Cluster: ", k,v)

    