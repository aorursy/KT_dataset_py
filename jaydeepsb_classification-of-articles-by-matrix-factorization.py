# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import glob

import json



#sys.path.insert(0, "../")



root_path = '/kaggle/input/CORD-19-research-challenge/2020-03-13'



json_filenames = glob.glob(f'{root_path}/**/*.json', recursive=True)

print(len(json_filenames))



#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
all_articles_df = pd.DataFrame(columns=["source", "title", "doc_id",  "abstract", "text_body"])



#all_articles_df = pd.DataFrame.from_dict(all_articles_df)
def return_corona_df(json_filenames, df, source):



    for file_name in json_filenames:



        row = {"doc_id": None, "source": None, "title": None,

              "abstract": None, "text_body": None}



        with open(file_name) as json_data:

            data = json.load(json_data)



            row['doc_id'] = data['paper_id']

            row['title'] = data['metadata']['title']



            # Now need all of abstract. Put it all in 

            # a list then use str.join() to split it

            # into paragraphs. 



            abstract_list = [data['abstract'][x]['text'] for x in range(len(data['abstract']) - 1)]

            abstract = "\n ".join(abstract_list)



            row['abstract'] = abstract



            # And lastly the body of the text. For some reason I am getting an index error

            # In one of the Json files, so rather than have it wrapped in a lovely list

            # comprehension I've had to use a for loop like a neanderthal. 

            

            # Needless to say this bug will be revisited and conquered. 

            

            body_list = []

            for _ in range(len(data['body_text'])):

                try:

                    body_list.append(data['body_text'][_]['text'])

                except:

                    pass



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



all_articles_df = return_corona_df(json_filenames, all_articles_df, 'b')

all_articles_df_out = all_articles_df.to_csv('kaggle_covid-19_open_csv_format.csv')
all_articles_df.head()
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.decomposition import NMF, LatentDirichletAllocation

titles = all_articles_df['title']

titles.fillna("",inplace=True)

titles.iloc[:5]
# Fit model
tfidf = TfidfVectorizer(max_features=10000, stop_words='english')

X_tfidf = tfidf.fit_transform(titles)

tfidf_feature_names = tfidf.get_feature_names()



vectorizer = CountVectorizer(stop_words='english', max_features=1000)

X_tf = vectorizer.fit_transform(titles)

tf_feature_names = vectorizer.get_feature_names()
tfidf_feature_names[500:510]
clustered = KMeans(n_clusters=6, random_state=0).fit_predict(X_tfidf)



all_articles_df['cluster_abstract']=clustered



grouped=all_articles_df.groupby('cluster_abstract')
grouped.count()
import pylab as plt

from numpy import arange

plt.figure()

for i in arange(500):

    plt.plot(i,len(titles.iloc[i]), 'ro')

plt.show()
n_topics = 15



# Run NMF

nmf = NMF(n_components=n_topics).fit(X_tfidf)



# Run LDA

lda = LatentDirichletAllocation(n_components=n_topics).fit(X_tf)
for j in arange(10):

    print("==============")

    for i in nmf.components_[j].argsort()[:-10:-1]:

        print(tfidf_feature_names[i])
#extract topics

def display_topics(model, feature_names, no_top_words):

    topics=[]

    for topic_idx, topic in enumerate(model.components_):

        #rint ("Topic %d:" % (topic_idx))

        topic_words=" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]])

        #rint(topic_words)

        topics.append(topic_words)

    return topics



no_top_words = 5

#rint("NMF: ")

topics_nmf=display_topics(nmf, tfidf_feature_names, no_top_words)

#rint("\nLDA: ")

topics_lda=display_topics(lda, tf_feature_names, no_top_words)



#rint(topics_nmf)

#rint(topics_lda)



pred_lda=lda.transform(X_tf)

pred_nmf=nmf.transform(X_tfidf)



res_lda=[topics_lda[np.argmax(r)] for r in pred_lda]

res_nmf=[topics_nmf[np.argmax(r)] for r in pred_nmf]
all_articles_df['topic_lda']=res_lda

all_articles_df['topic_nmf']=res_nmf
all_articles_df.head()
grouped=all_articles_df.groupby('topic_nmf')
grouped.count()