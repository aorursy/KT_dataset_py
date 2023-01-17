!pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer

import pickle

import sklearn

import numpy as np
embedder = SentenceTransformer('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Trained_Model/training_stsbenchmark_continue_training-nli_allenai-scibert_scivocab_uncased-2020-04-24_07-41-01')
# Queries sentences

queries = ['What is known about COVID-19 transmission, incubation, and environmental stability?', 

           'What do we know about natural history, transmission, and diagnostics for the COVID-19 virus?',

           'What have we learned about COVID-19 infection prevention and control?',

           'Range of incubation periods for the COVID-19 disease in humans (and how this varies across age and health status) and how long individuals are contagious, even after recovery.',

           'COVID-19 prevalence of asymptomatic shedding and transmission (e.g., particularly children).',

           'COVID-19 seasonality of transmission.',

           'Physical science of the COVID-19 coronavirus (e.g., charge distribution, adhesion to hydrophilic/phobic surfaces, environmental survival to inform decontamination efforts for affected areas and provide information about viral shedding).',

           'COVID-19 coronavirus persistence and stability on a multitude of substrates and sources (e.g., nasal discharge, sputum, urine, fecal matter, blood).',

           'Persistence of COVID-19 coronavirus on surfaces of different materials (e,g., copper, stainless steel, plastic).',

           'Natural history of the COVID-19 virus and shedding of it from an infected person.',

           'Implementation of diagnostics and products to improve clinical processes for COVID-19.',

           'COVID-19 disease models, including animal models for infection, disease and transmission.',

           'Tools and studies to monitor phenotypic change and potential adaptation of the COVID-19 coronavirus.',

           'COVID-19 immune response and immunity.',

           'Effectiveness of movement control strategies to prevent secondary transmission in health care and community settings.',

           'Effectiveness of personal protective equipment (PPE) and its usefulness to reduce risk of transmission in health care and community settings.',

           'Role of the environment in transmission.']



# Create Queries Embeddings

query_embeddings = embedder.encode(queries)
# Number of similarity sentences from top

top_n = 10
# Read Corpus

with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_title.pickle', 'rb') as f:

    corpus = pickle.load(f)



# Read Corpus Embedding

with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_title_emb.pickle', 'rb') as f:

    corpus_embeddings = pickle.load(f)
# Calculate cosine similarity.

for query, query_embedding in zip(queries, query_embeddings):

    distances = sklearn.metrics.pairwise.cosine_similarity([query_embedding], corpus_embeddings, "cosine")[0]

    print('\nQuery: {}'.format(query))

    print('Top {} similarity sentences out of {} sentences'.format(top_n, len(distances)))

    for i in range(top_n):

        print('(Similarity:{:.2f})'.format(np.sort(distances)[::-1][i]), corpus[np.argsort(distances)[::-1][i]])
# Read Corpus

with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_abstract.pickle', 'rb') as f:

    corpus = pickle.load(f)



# Read Corpus Embedding

with open('/kaggle/input/cord19-model-and-emb/My_Model_and_Corpus/Corpus/Corpus_and_Embeddings/corpus_abstract_emb.pickle', 'rb') as f:

    corpus_embeddings = pickle.load(f)
# Calculate cosine similarity.

for query, query_embedding in zip(queries, query_embeddings):

    distances = sklearn.metrics.pairwise.cosine_similarity([query_embedding], corpus_embeddings, "cosine")[0]

    print('\nQuery: {}'.format(query))

    print('Top {} similarity sentences out of {} sentences'.format(top_n, len(distances)))

    for i in range(top_n):

        print('(Similarity:{:.2f})'.format(np.sort(distances)[::-1][i]), corpus[np.argsort(distances)[::-1][i]])