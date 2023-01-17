# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df1=pd.read_csv("/kaggle/input/unstructured-l0-nlp-hackathon/data.csv")
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

import nltk
df1.head()
df1['text'] =df1['text'].str.replace("[^a-zA-Z#]", " ")

stopwords_list = stopwords.words('english')

punctuations = list(set(string.punctuation))



def clean_text_initial(text):

    text = ' '.join([x.lower() for x in word_tokenize(text) if x.lower() not in stopwords_list and len(x)>1])

    text = ' '.join([x.lower() for x in word_tokenize(text) if x.lower() not in punctuations and len(x)>3])

    text = ' '.join([x.lower() for x in word_tokenize(text) if nltk.pos_tag([x])[0][1].startswith("NN") or nltk.pos_tag([x])[0][1].startswith("JJ")])

    return text.strip()



df1["clean_text"]=df1.text.apply(lambda text:clean_text_initial(str(text)))

df1.head()
from gensim.corpora.dictionary import Dictionary



from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()



from gensim.models import ldamodel
cleaned_text_list=df1.clean_text.apply(lambda clean_text:[lemmatizer.lemmatize(tokenized_text) for tokenized_text in word_tokenize(clean_text)])



gensim_dict=Dictionary(cleaned_text_list)



doc_term_matrix = [gensim_dict.doc2bow(text) for text in cleaned_text_list]



LDA = ldamodel.LdaModel
from sklearn.model_selection import GridSearchCV

from gensim.sklearn_api import LdaTransformer
num_topics = 5



# Define Search Param

search_params = {'alpha':np.arange(0,1,0.1) , 'eta': np.arange(0,1,0.1)}



# Init the Model

lda = LdaTransformer(num_topics=num_topics,id2word=gensim_dict, iterations=10, random_state=1)



# Init Grid Search Class

model = GridSearchCV(lda, param_grid=search_params)



# Do the Grid Search

model.fit(doc_term_matrix)
# Best Model

best_lda_model = model.best_estimator_



# Model Parameters

print("Best Model's Params: ", model.best_params_)



# Log Likelihood Score

print("Best Log Likelihood Score: ", model.best_score_)
num_topics = 5



# Running and Training LDA model on the document term matrix.

lda_model = LDA(corpus=doc_term_matrix, num_topics=num_topics, id2word = gensim_dict, passes=10,random_state=1,alpha=0.1,eta=0.4)
def get_lda_topics(model, num_topics):

    word_dict = {}

    topics = model.show_topics(num_topics,10)

    word_dict = {'Topic '+str(i):[x.split('*') for x in words.split('+')] \

                 for i,words in model.show_topics(num_topics,10)}

    return pd.DataFrame.from_dict(word_dict)



get_lda_topics(lda_model, 5)
df_doc_top = pd.DataFrame()

final_list = []

for index in range(len(df1.clean_text)):

    word_id_dict = dict(lda_model.get_document_topics(doc_term_matrix[index]))

    word_score_list = []

    for index in range(num_topics):

        try:

            value = word_id_dict[index]

        except:

            value = 0

        word_score_list.append(value)

    final_list.append(word_score_list)
df_doc_top = pd.DataFrame(final_list)

df_doc_top.columns = ['Topic ' + str(i) for i in range(1, num_topics+1)]

df_doc_top.index = ['Document ' + str(i) for i in range(1, len(df1.clean_text)+1)]

df_doc_top.head()
df_doc_top["Dominant_Topic"] = df_doc_top.idxmax(axis=1).tolist()

df_doc_top["Topic_Probability"] = df_doc_top.max(axis=1).tolist()

document_df = df_doc_top.reset_index().rename(columns={"index":"Document"})[["Document","Dominant_Topic","Topic_Probability"]]

document_df
initial_submission=pd.concat([df1.Id,document_df.Dominant_Topic],axis=1)
initial_submission.Dominant_Topic=initial_submission.Dominant_Topic.replace({"Topic 5":"Automobiles","Topic 4":"room_rentals",

                                           "Topic 3":"glassdoor_reviews","Topic 2":"sports_news",

                                          "Topic 1":"tech_news"})
initial_submission=initial_submission.set_index("Id").rename(columns={"Dominant_Topic":"topic"})
initial_submission.to_csv("initial_submission.csv")