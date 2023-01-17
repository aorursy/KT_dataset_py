# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



# There are too many paths and printing them takes

# up too much space so I don't do this normally

if 1==0: 

    for dirname, _, filenames in os.walk('/kaggle/input'):

        for filename in filenames:

            pass

            #print(os.path.join(dirname, filename))
# Progress bar

import tqdm



# Word2Vec

from gensim.models.word2vec import Word2Vec



from nltk.tokenize import word_tokenize 

from scipy.spatial.distance import cdist
biorxiv_clean = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/biorxiv_clean.csv")

clean_comm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_comm_use.csv")

clean_noncomm_use = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_noncomm_use.csv")

clean_pmc = pd.read_csv("../input/cord-19-eda-parse-json-and-generate-clean-csv/clean_pmc.csv")



all_data = pd.concat([biorxiv_clean, clean_comm_use, clean_noncomm_use, clean_pmc]).reset_index(drop=True)



all_data.head()
print("Number of Rows in Table: %i" % len(all_data))

print("Number of Titles: %i " % all_data['title'].count())

print("Number of Abstracts: %i " % all_data['abstract'].count())

print("Number of Texts: %i " % all_data['text'].count())
# replace newlines with empty strings

all_text = all_data.text.str.replace('\n\n', ' ')

# remove citations

ptn = r'\[[0-9]{1,2}\]'

all_text = all_text.str.replace(ptn,'')
all_text_list = list(

    map(

        lambda x: word_tokenize(x), all_text.values

    ))
model = Word2Vec(all_text_list, size=300, iter=10)
document_dict = {}

for idx, text_df in tqdm.tqdm(all_data[["paper_id", "text"]].iterrows()):

    text = text_df['text'].replace("\n\n", ' ')

    

    word_vector = (

        list(

            map( # this part takes every word and converts it into a vector

                lambda x: model.wv[x], 

                filter( #this part checks that word is in the model

                    lambda x: x in model.wv, 

                    text.split(" ")

                )

            )

        )

    )

    # here we are making a dictory where the document is a key and the value

    # is the average of the word vectors in that document

    if len(word_vector) > 0:

        document_dict[text_df['paper_id']] = pd.np.stack(word_vector, axis=1).mean(axis=1)
print(len(document_dict))
print(list(document_dict.values())[0].shape)
document_embeddings_df = pd.DataFrame.from_dict(document_dict, orient="index")

document_embeddings_df.head()
def cos_sim(text):

    word_vector_list = (

        list(

            map( # this part takes every word and converts it into a vector

                lambda x: model.wv[x], 

                filter( #this part checks that word is in the model

                    lambda x: x in model.wv, 

                    text.split(" ")

                )

            )

        )

    )

    mean_word_vec = pd.np.stack(word_vector_list, axis=1).mean(axis=1)

    



    

    # compute similarity

    doc_sim = (

        1-cdist(

            document_embeddings_df.values,

            [mean_word_vec],

            'cosine'

        )

    )

    # convert result to a date frame

    document_sim_df = (

        pd.DataFrame(doc_sim, columns=["cos_sim"])

        .assign(document_id=list(document_embeddings_df.index))

    )

    # sort from most similar to least

    document_sim_df = document_sim_df.sort_values("cos_sim", ascending=False)

    

    # perform left-join to get information about the documents

    doc_sim_meta_df = document_sim_df.merge(all_data,

                      how='left',

                     left_on='document_id',

                     right_on='paper_id')

    return(doc_sim_meta_df)
pd.options.display.max_colwidth = 100

cos_sim('airborne').loc[range(0,10),"title"]
pd.options.display.max_colwidth = 100

cos_sim('non-pharmaceutical interventions').loc[range(0,10),"title"]