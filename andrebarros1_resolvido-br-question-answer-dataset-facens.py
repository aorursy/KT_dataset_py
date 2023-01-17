import pandas as pd

import numpy as np

import string

df_S08 = pd.read_csv('../input/S08_question_answer_pairs.txt', sep="\t", header=0)

df_S09 = pd.read_csv('../input/S09_question_answer_pairs.txt', sep="\t", header=0)

df_S10 = pd.read_csv('../input/S10_question_answer_pairs.txt', sep="\t", header=0, encoding = "ISO-8859-1")
df_list = ['df_S08','df_S09','df_S10']

questions = []

answer = []

for d in df_list:

    d = eval(d)

    d['Question'] = d['ArticleTitle'] + " " + d['Question'] # concatena nome artigo em pergunta para melhorar referencia

    d = d[['Question','Answer']] #remove colunas indesejadas

    d = d.dropna() # Remove NaN

    d['Answer'] = d['Answer'].str.lower() # deixa tudo minusculo

    d['Answer'] = d['Answer'].str.replace('[{}]'.format(string.punctuation), '') # remove pontuacao

    d['Answer'] = d['Answer'].str.replace(r'\\t', '') #remove qualquer "\t" que existir em respostas

    d['Question'] = d['Question'].str.replace(r'_', ' ') #remove qualquer "_" que existir em perguntas

    d = d.sort_values('Question').drop_duplicates(subset=['Question', 'Answer'], keep='last') #remove dados duplicados

    questions = np.append(questions, d['Question'].values) #cria np array de questoes

    answer = np.append(answer, d['Answer'].values) #cria np array de respostas
perguntas = tuple(questions)

pergunta = ["Does Arabic language have many words borrowed by European languages?"]



from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()

tfidf_matrix = tfidf_vectorizer.fit_transform(perguntas)



from sklearn.metrics.pairwise import cosine_similarity

query_vect = tfidf_vectorizer.transform(pergunta)

similariedade = cosine_similarity(query_vect, tfidf_matrix) # calcula e cria um vetor com os valores de similariedade



indice = np.unravel_index(np.argmax(similariedade, axis=None), similariedade.shape) #encontra o indice do valor com maior similariedade



print("RESPOSTA: {}".format(answer[indice[1]]))

print("PERGUNTA ORIGINAL: {}".format(questions[indice[1]]))
display(df_S09.Question[100])

display(df_S09.Answer[100])