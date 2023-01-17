# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm.notebook import tqdm

import os

import json

from pprint import pprint

from copy import deepcopy



# define global variable to print logs

loglevel ='INFO'
def format_name(author):

    middle_name = " ".join(author['middle'])

    

    if author['middle']:

        return " ".join([author['first'], middle_name, author['last']])

    else:

        return " ".join([author['first'], author['last']])





def format_affiliation(affiliation):

    text = []

    location = affiliation.get('location')

    if location:

        text.extend(list(affiliation['location'].values()))

    

    institution = affiliation.get('institution')

    if institution:

        text = [institution] + text

    return ", ".join(text)



def format_authors(authors, with_affiliation=False):

    name_ls = []

    

    for author in authors:

        name = format_name(author)

        if with_affiliation:

            affiliation = format_affiliation(author['affiliation'])

            if affiliation:

                name_ls.append(f"{name} ({affiliation})")

            else:

                name_ls.append(name)

        else:

            name_ls.append(name)

    

    return ", ".join(name_ls)



def format_body(body_text):

    texts = [(di['section'], di['text']) for di in body_text]

    texts_di = {di['section']: "" for di in body_text}

    

    for section, text in texts:

        texts_di[section] += text



    body = ""



    for section, text in texts_di.items():

        body += section

        body += "\n\n"

        body += text

        body += "\n\n"

    

    return body



def format_bib(bibs):

    if type(bibs) == dict:

        bibs = list(bibs.values())

    bibs = deepcopy(bibs)

    formatted = []

    

    for bib in bibs:

        bib['authors'] = format_authors(

            bib['authors'], 

            with_affiliation=False

        )

        formatted_ls = [str(bib[k]) for k in ['title', 'authors', 'venue', 'year']]

        formatted.append(", ".join(formatted_ls))



    return "; ".join(formatted)
def get_filenames(dirname):

    filenames = os.listdir(dirname)

    raw_files = []



    for filename in tqdm(filenames):

        filename = dirname + filename

        file = json.load(open(filename, 'rb'))

        raw_files.append(file)

    

    return raw_files



def create_df(all_files,article_type):

    cleaned_files = []

    

    for file in tqdm(all_files):

        features = [

            file['paper_id'],

            file['metadata']['title'],

            format_authors(file['metadata']['authors']),

            format_authors(file['metadata']['authors'], 

                           with_affiliation=True),

            format_body(file['abstract']),

            format_body(file['body_text']),

            format_bib(file['bib_entries']),

            file['metadata']['authors'],

            file['bib_entries'],

            article_type

        ]



        cleaned_files.append(features)



    col_names = ['paper_id', 'title', 'authors',

                 'affiliations', 'abstract', 'text', 

                 'bibliography','raw_authors','raw_bibliography','article_type']



    clean_df = pd.DataFrame(cleaned_files, columns=col_names)

    clean_df.head()

    

    return clean_df
dirnames = [{'dir':'/kaggle/input/CORD-19-research-challenge/2020-03-13/biorxiv_medrxiv/biorxiv_medrxiv/','type':'biorxiv'},{'dir':'/kaggle/input/CORD-19-research-challenge/2020-03-13/pmc_custom_license/pmc_custom_license/','type':'PMC'},

            {'dir':'/kaggle/input/CORD-19-research-challenge/2020-03-13/comm_use_subset/comm_use_subset/','type':'Comm'},{'dir':'/kaggle/input/CORD-19-research-challenge/2020-03-13/noncomm_use_subset/noncomm_use_subset/','type':'NonComm'}]
dataframe = pd.DataFrame()

for item in dirnames:

    print(item['dir'],item['type'])

    dirname = item['dir']

    types = item['type']

    filenames = get_filenames(dirname)

    # extract article type and article body text

    df = create_df(filenames,types)[['article_type','text']]

    dataframe= pd.concat((dataframe,df))
# library to clean data 

import re  

  

# Natural Language Tool Kit 

import nltk  



# to remove stopword 

from nltk.corpus import stopwords 

  

# for Stemming propose  

from nltk.stem.porter import PorterStemmer 
# sample https://www.geeksforgeeks.org/python-nlp-analysis-of-restaurant-reviews/

loglevel ='INFO'

def clean_article_text(texts):

    """

    clean review columns from row and reconstruct and return

    """

    if loglevel == 'DEBUG':

        print(f"intital quote\n {reviews}")

    # column : "Review" with only a-zA-Z char

    text = re.sub('[^a-zA-Z]', ' ', texts[0])  

    if loglevel == 'DEBUG':

        print(f"step1 quote\n {text}")

      

    # convert all cases to lower cases 

    text = text.lower()

    if loglevel == 'DEBUG':

        print(f"step3 quote\n {text}")

      

    # split to array(default delimiter is " ") 

    text = text.split()  

    if loglevel == 'DEBUG':

        print(f"step4 quote\n {review}")

      

    # creating PorterStemmer object to 

    # take main stem of each word 

    ps = PorterStemmer()  

      

    # loop for stemming each word 

    # in string array at ith row   

    # remove words thats are in stopwords file

    text = [ps.stem(word) for word in text 

                if not word in set(stopwords.words('english'))]  

    

    if loglevel == 'DEBUG':

        print(f"step5 quote\n {text}")

    # rejoin all string array elements 

    # to create back into a string 

    text = ' '.join(text)   

    if loglevel == 'DEBUG':

        print(f"step6 quote\n {text}")



    return text
%%time

dataframe.loc[:,'clean_text'] = np.apply_along_axis(clean_article_text,1,dataframe[['text']])
import gensim 

from gensim.models import Doc2Vec

from sklearn.decomposition import PCA



# create contains that can be used in Dc2Vec



LabeledSentence1 = gensim.models.doc2vec.TaggedDocument

all_content_train = []

j=0

for em in dataframe['clean_text'].values:

    all_content_train.append(LabeledSentence1(em,[j]))

    j+=1

print("Number of texts processed: ", j)



# create document vectors

d2v_model = Doc2Vec(all_content_train, vector_size = 100, window = 10, min_count = 500, workers=7, dm = 1,alpha=0.025, min_alpha=0.001)

d2v_model.train(all_content_train, total_examples=d2v_model.corpus_count, epochs=10, start_alpha=0.002, end_alpha=-0.016)



# Now lets use  Kmean clustering

kmeans_model = KMeans(n_clusters=10, init='k-means++', max_iter=200) 

X = kmeans_model.fit(d2v_model.docvecs.vectors_docs)

# get reviews labels

labels=kmeans_model.labels_.tolist()

#l = kmeans_model.fit_predict(d2v_model.docvecs.vectors_docs)

# using PCA to project vectors to 2D for plotting

pca = PCA(n_components=2).fit(d2v_model.docvecs.vectors_docs)

datapoint = pca.transform(d2v_model.docvecs.vectors_docs)





                      

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(20,10))

#label1 = ['#FFFF00', '#008000', '#0000FF', '#800080']

#color = [label1[i] for i in labels]

plt.scatter(datapoint[:, 0], datapoint[:, 1], c=labels)

# show cluser centers on plot

centroids = kmeans_model.cluster_centers_

centroidpoint = pca.transform(centroids)

plt.scatter(centroidpoint[:, 0], centroidpoint[:, 1], marker='^', s=150, c='#000000')

plt.show()





# add clusters labels to review so that we can sample reviews in a clusters



df_review.loc[:,'cluster'] = labels