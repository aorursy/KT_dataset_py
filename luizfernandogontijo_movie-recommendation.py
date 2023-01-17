import numpy as np

import pandas as pd

import spacy

from sklearn.feature_extraction.text import TfidfVectorizer

from spacy.lang.en.stop_words import STOP_WORDS

from sklearn.metrics.pairwise import cosine_similarity 
md = pd.read_csv('../input/wikipedia-movie-plots/wiki_movie_plots_deduped.csv')
md.head()
md.describe()
md_plot = md['Plot']
md_plot.head()
md_nan = md_plot.isna()

md_nan.sum()
nlp = spacy.load('en_core_web_sm') 
doc = nlp(md_plot[0]) 

print(doc) 
lemmas = [token.lemma_ for token in doc] 

print(lemmas)
a_lemmas = [lemma for lemma in lemmas 

            if lemma.isalpha() or lemma not in STOP_WORDS] 



print(a_lemmas)
print(' '.join(a_lemmas))
def preprocess(text):

    doc = nlp(text)

    lemmas = [token.lemma_ for token in doc]

    a_lemmas = [lemma for lemma in lemmas 

            if lemma.isalpha() and lemma not in STOP_WORDS]

    

    return ' '.join(a_lemmas)
preprocess(md_plot[0]) # verificar o resultado da função

print(md_plot[0])
md_plot_test = [[md_plot[0]], [md_plot[1]], [md_plot[2]]] #selecionar apenas algumas linhas

md_plot_test = pd.DataFrame(md_plot_test, columns = ['Plot']) 

    

md_plot_test['test'] = md_plot_test['Plot'].apply(lambda x: preprocess(x))



md_plot_test #cria uma nova coluna
md_half = md[:len(md)//2] 
md_half['Plot_lemma'] = md_half['Plot'].apply(lambda x: preprocess(x)) 
md_half #verificar a nova coluna
md_half.head()
np.savez_compressed('md_half')

md_half.to_csv('csv_to_submit.csv', index = False)
vectorizer = TfidfVectorizer()
md_half_plot_lemma = md_half['Plot_lemma'] 

md_half_plot_lemma.head()

md_half_plot_lemma.shape
tfidf_matrix_teste = vectorizer.fit_transform(md_plot_test['test'])

print(tfidf_matrix_teste) 
tfidf_matrix_half = vectorizer.fit_transform(md_half['Plot_lemma']) #criar matriz de TFIDF
md_half_plot_lemma.shape
print(tfidf_matrix_half) 
cosine_sim_test = cosine_similarity(tfidf_matrix_teste, tfidf_matrix_teste)
cosine_sim_half = cosine_similarity(tfidf_matrix_half, tfidf_matrix_half)
print(cosine_sim_half)
cosine_sim_half.shape
indices_half = pd.Series(md_half.index, index=md_half['Title']).drop_duplicates() #pegar os nomes de cada filme

indices_half


def get_recommendations(title, cosine_sim, indices):

    idx = indices[title]

    sim_scores = list(enumerate(cosine_sim[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[1:6]

    movie_indices = [i[0] for i in sim_scores]

    return md_half['Title'].iloc[movie_indices]
print(get_recommendations('The Godfather', cosine_sim_half, indices_half))