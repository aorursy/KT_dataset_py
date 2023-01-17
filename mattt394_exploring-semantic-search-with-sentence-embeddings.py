!pip install -U sentence-transformers
import pickle

import numpy as np

import pandas as pd

from sentence_transformers import SentenceTransformer

from annoy import AnnoyIndex
model = SentenceTransformer('bert-large-nli-mean-tokens')
annoy_index = AnnoyIndex(1024, 'angular')

annoy_index.load('/kaggle/input/my-coronavirus-kernel/annoy_index')
df_covid = pd.read_pickle('/kaggle/input/my-coronavirus-kernel/df_covid.pkl')

sent_list = pd.read_pickle('/kaggle/input/my-coronavirus-kernel/sent_list.pkl')
pickle_in = open('/kaggle/input/my-coronavirus-kernel/sent_embeddings.pkl',"rb")

sent_embeddings = pickle.load(pickle_in)
sent_concat = np.concatenate(sent_list.values)
from IPython.display import Markdown, display

def printmd(string):

    display(Markdown(string))
def semantic_search(input_sentence):

    # convert to list

    input_list = [input_sentence]

    #get converted vector

    input_vector = model.encode(input_list)

    #perform nearest neighbours search

    nearest_neighbours = annoy_index.get_nns_by_vector(input_vector[0],5)

    nn_list = []

    for i in nearest_neighbours:

        l = []

        l.append(i-1)

        l.append(i)

        l.append(i+1)

        nn_list.append(l)



    #print nearest neighbours

    pd.options.display.width = 750

    pd.options.display.max_colwidth = 1500



    sentinel = []

    for sublist in nn_list:

        search_sents = []

        for i, j in enumerate(sublist):

            if i==1:

                search_sents.append(('**' + sent_concat[j] + '**'))

            else:

                search_sents.append((sent_concat[j]))

        sent = ' '.join(search_sents)

        printmd(sent)

        sentinel.append(sent)



    return pd.DataFrame(sentinel)

semantic_search('The risk factors associated with coronavirus');
_ = semantic_search('Are smokers at increased risk from coronavirus')
_ = semantic_search('pre-existing pulmonary disease exacerbates coronavirus')
_ = semantic_search('Co-infections as risk factors for viral pneumonia')
_ = semantic_search('determine whether co-existing respiratory/viral infections make the COVID-19 more transmissible or virulent')
_ = semantic_search('Identify co-morbidities as risk factors for coronavirus')
_ = semantic_search('potential risk factors for neonates')
_ = semantic_search('Socio-economic factors in the economic impact of the coronavirus')
_ = semantic_search('R0 basic reproductive number of novel coronavirus')
_ = semantic_search('incubation period of coronavirus')
_ = semantic_search('the serial interval of coronavirus')
_ = semantic_search('the various modes of transmission for coronavirus')
_ = semantic_search('environmental factors in the transmission of coronavirus')
_ = semantic_search('risk of fatality among symptomatic hospitalized patients of coronavirus')
_ = semantic_search('risk of fatality among high-risk patient groups of coronavirus')
_ = semantic_search('Susceptibility of different populations to coronavirus')
_ = semantic_search('Public health mitigation measures that could be effective for controlling coronavirus')